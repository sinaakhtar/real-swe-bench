"""Step 4: Embed correlated Q&A + diff pairs with `gemini-embedding-2`, deduplicate semantic clusters, and build the stratified `agents-cli eval` dataset.

Methodology implemented:
1. Loads correlated PR sessions (`passed_survival_filter == True`) from `data/correlated_pr_sessions.json`.
2. Calls `gemini-embedding-2` on Vertex AI (`location="global"`)
   to generate 3072-dimensional embeddings for each correlated SWE task.
3. Computes the pairwise cosine similarity matrix and groups near-duplicate tasks (`similarity >= 0.78`)
   within the same complexity tier so repetitive bugfixes do not skew the evaluation suite.
4. Performs stratified selection across `tier_1_simple`, `tier_2_medium`, and `tier_3_complex`.
5. Outputs the `agents-cli eval` compatible `EvaluationDataset` JSON to:
   - `tests/eval/datasets/swe_routing_dataset.json`
   - `tests/eval/datasets/basic-dataset.json`
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import google.auth
import numpy as np
from google import genai

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "tests" / "eval" / "datasets"

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
EMBEDDING_MODEL = "gemini-embedding-2"
DEDUP_SIMILARITY_THRESHOLD = 0.78


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    return np.dot(normalized, normalized.T)


def curate_eval_dataset() -> None:
    correlated = json.loads((DATA_DIR / "correlated_pr_sessions.json").read_text(encoding="utf-8"))
    candidates = [r for r in correlated if r["passed_survival_filter"]]
    print(f"[Step 4.1] Loaded {len(candidates)} golden PR candidates (after filtering low edit-survival PRs).")

    # Build text representations combining developer prompt, target file, and solution diff
    texts_to_embed = [
        f"Developer Task: {c['user_prompt']}\nFile: {c['file_path']}\nMerged Solution Diff:\n{c['merged_pr_diff']}"
        for c in candidates
    ]

    print(f"\n[Step 4.2] Computing 3072-d embeddings via Vertex AI `{EMBEDDING_MODEL}` (project={PROJECT_ID}, location=global)...")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
    raw_vecs = []
    for text in texts_to_embed:
        emb_resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        raw_vecs.append(emb_resp.embeddings[0].values)
    vectors = np.array(raw_vecs, dtype=np.float64)
    print(f"  Generated embeddings matrix of shape {vectors.shape}.")

    sim_matrix = cosine_similarity_matrix(vectors)
    pr_labels = [f"PR#{c['pr_number']}" for c in candidates]

    print("\n[Step 4.3] Pairwise Cosine Similarity Matrix (`gemini-embedding-2`):")
    header = "        " + "  ".join(f"{lbl:>7}" for lbl in pr_labels)
    print(header)
    for i, lbl in enumerate(pr_labels):
        row_str = "  ".join(f"{sim_matrix[i, j]:7.3f}" for j in range(len(pr_labels)))
        print(f"{lbl:>7} {row_str}")

    # Greedy clustering / deduplication within complexity tiers
    visited = set()
    clusters: list[dict] = []
    for i, cand in enumerate(candidates):
        if i in visited:
            continue
        cluster_members = [i]
        visited.add(i)
        for j in range(i + 1, len(candidates)):
            if j in visited:
                continue
            if sim_matrix[i, j] >= DEDUP_SIMILARITY_THRESHOLD:
                cluster_members.append(j)
                visited.add(j)

        # Select the member with highest edit_survival_rate (and earliest PR number as tie-breaker) as medoid
        best_idx = max(cluster_members, key=lambda idx: (candidates[idx]["edit_survival_rate"], -candidates[idx]["pr_number"]))
        clusters.append({
            "cluster_id": f"cluster_{len(clusters) + 1}",
            "medoid_pr": candidates[best_idx]["pr_number"],
            "member_prs": [candidates[m]["pr_number"] for m in cluster_members],
            "dropped_duplicates": [candidates[m]["pr_number"] for m in cluster_members if m != best_idx],
            "max_intra_cluster_similarity": float(
                max([sim_matrix[best_idx, m] for m in cluster_members if m != best_idx], default=1.0)
            ),
            "selected_candidate": candidates[best_idx],
        })

    print(f"\n[Step 4.4] Semantic Deduplication & Stratified Selection (threshold={DEDUP_SIMILARITY_THRESHOLD}):")
    for cl in clusters:
        sel = cl["selected_candidate"]
        dropped = cl["dropped_duplicates"]
        dup_info = f"dropped near-duplicate PRs {dropped} (sim={cl['max_intra_cluster_similarity']:.3f})" if dropped else "unique task"
        print(
            f"  {cl['cluster_id']}: Selected PR #{sel['pr_number']} [{sel['complexity_tier']}] "
            f"({sel['file_path']}) — {dup_info}"
        )

    # Save similarity & clustering report
    curation_report = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(vectors.shape[1]),
        "dedup_similarity_threshold": DEDUP_SIMILARITY_THRESHOLD,
        "pr_labels": pr_labels,
        "cosine_similarity_matrix": np.round(sim_matrix, 4).tolist(),
        "clusters": [
            {k: v for k, v in cl.items() if k != "selected_candidate"}
            for cl in clusters
        ],
    }
    (DATA_DIR / "embedding_similarity_report.json").write_text(
        json.dumps(curation_report, indent=2), encoding="utf-8"
    )

    # Build canonical agents-cli EvaluationDataset
    eval_cases = []
    for cl in clusters:
        c = cl["selected_candidate"]
        case_id = f"pr_{c['pr_number']}_{c['complexity_tier']}"
        prompt_text = (
            f"[CASE_ID={case_id}] [TARGET_FILE={c['file_path']}] [PYTEST_TARGET={c['pytest_target']}] "
            f"[COMPLEXITY_TIER={c['complexity_tier']}]\n\n"
            f"{c['user_prompt']}\n\n"
            "Use `read_repo_file` to inspect the current file, then use `test_and_verify_patch` to verify your fix "
            "against the unit test suite, and include the final unified diff in your response."
        )
        eval_cases.append({
            "eval_case_id": case_id,
            "prompt": {
                "role": "user",
                "parts": [{"text": prompt_text}],
            },
            "reference": {
                "response": {
                    "role": "model",
                    "parts": [{"text": c["merged_pr_diff"]}],
                }
            },
        })

    dataset_obj = {"eval_cases": eval_cases}
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    swe_dataset_path = DATASETS_DIR / "swe_routing_dataset.json"
    basic_dataset_path = DATASETS_DIR / "basic-dataset.json"
    swe_dataset_path.write_text(json.dumps(dataset_obj, indent=2), encoding="utf-8")
    basic_dataset_path.write_text(json.dumps(dataset_obj, indent=2), encoding="utf-8")

    print(f"\n[Step 4.5] Wrote {len(eval_cases)} stratified eval cases to:")
    print(f"  - {swe_dataset_path}")
    print(f"  - {basic_dataset_path}")


if __name__ == "__main__":
    curate_eval_dataset()
