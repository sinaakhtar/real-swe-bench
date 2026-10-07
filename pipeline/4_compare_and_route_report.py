"""Step 5: Compare `gemini-3.8-flash` vs `claude-sonnet-5-5` evaluation results via `agents-cli eval compare` and build a Stratified Model Routing & Cost Optimization Report."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_PATH = ROOT_DIR / "tests" / "eval" / "datasets" / "swe_routing_dataset.json"

# Official Vertex AI Standard PayGo per-1M token pricing (USD, Global endpoint)
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
MODEL_PRICING_PER_1M = {
    "claude-opus-5-5 (Current Baseline)": {
        "input": 4.00,
        "output": 20.00,
        "regional_input": 4.40,
        "regional_output": 22.00,
        "opus_4_6_input": 5.00,
        "opus_4_6_output": 25.00,
    },
    "claude-sonnet-5-5": {
        "input": 2.00,
        "output": 10.00,
        "regional_input": 2.20,
        "regional_output": 11.00,
        "sonnet_4_6_input": 3.00,
        "sonnet_4_6_output": 15.00,
    },
    "gemini-3.8-flash": {
        "input": 0.75,
        "output": 3.75,
        "standard_2027_input": 1.50,
        "standard_2027_output": 7.50,
        "gemini_3_flash_preview_input": 0.50,
        "gemini_3_flash_preview_output": 3.00,
    },
}


def latest_result_json(subdir: str) -> Path:
    folder = ARTIFACTS_DIR / "grade_results" / subdir
    files = sorted(folder.glob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"No results_*.json found in {folder}")
    return files[-1]


def extract_case_metrics(results_path: Path, dataset_cases: list[dict]) -> list[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("eval_case_results", []):
        idx = item["eval_case_index"]
        case_meta = dataset_cases[idx] if idx < len(dataset_cases) else {}
        case_id = case_meta.get("eval_case_id", f"case_{idx}")
        m_res = item["response_candidate_results"][0]["metric_results"]

        rows.append({
            "eval_case_id": case_id,
            "pytest_unit_test_pass": m_res.get("pytest_unit_test_pass", {}).get("score"),
            "patch_semantic_equivalence": m_res.get("patch_semantic_equivalence", {}).get("score"),
            "multi_turn_task_success": m_res.get("multi_turn_task_success_v1", {}).get("score"),
            "multi_turn_tool_use_quality": m_res.get("multi_turn_tool_use_quality_v1", {}).get("score"),
            "agent_tool_call_count": m_res.get("agent_tool_call_count", {}).get("score"),
        })
    return rows


def generate_routing_report() -> None:
    import os

    flash_json = latest_result_json("gemini_3_8_flash")
    sonnet_json = latest_result_json("claude_sonnet_5_5")

    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    print(f"[Step 5.1] Running `agents-cli eval compare`:\n  Baseline:  {sonnet_json}\n  Candidate: {flash_json}\n")
    proc = subprocess.run(
        ["agents-cli", "eval", "compare", str(sonnet_json), str(flash_json)],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    raw_out = proc.stdout
    json_start = raw_out.find("{")
    compare_json = json.loads(raw_out[json_start:] if json_start != -1 else raw_out)
    summary_diff = compare_json.get("differences", {}).get("summary_metrics", {})
    print("Summary Metrics Comparison (Baseline=claude-sonnet-5-5, Candidate=gemini-3.8-flash):")
    print(json.dumps(summary_diff, indent=2))
    print()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["eval_cases"]
    flash_cases = extract_case_metrics(flash_json, dataset)
    sonnet_cases = extract_case_metrics(sonnet_json, dataset)

    print("[Step 5.2] Per-Case Stratified Evaluation Breakdown (Gemini 3.8 Flash vs Claude Sonnet 5.5):")
    print(
        f"{'Eval Case ID':<26} | {'Flash Pytest':>12} | {'Sonnet Pytest':>13} | "
        f"{'Flash Diff(1-5)':>15} | {'Sonnet Diff(1-5)':>16} | {'Flash TaskSucc':>14} | {'Sonnet TaskSucc':>15}"
    )
    print("-" * 124)

    combined_cases = []
    for f_row, s_row in zip(flash_cases, sonnet_cases):
        cid = f_row["eval_case_id"]
        print(
            f"{cid:<26} | {f_row['pytest_unit_test_pass']:>12.2f} | {s_row['pytest_unit_test_pass']:>13.2f} | "
            f"{f_row['patch_semantic_equivalence']:>15.2f} | {s_row['patch_semantic_equivalence']:>16.2f} | "
            f"{f_row['multi_turn_task_success']:>14.2f} | {s_row['multi_turn_task_success']:>15.2f}"
        )
        combined_cases.append({
            "eval_case_id": cid,
            "gemini_3_8_flash": f_row,
            "claude_sonnet_5_5": s_row,
        })

    report = {
        "models_compared": ["claude-sonnet-5-5", "gemini-3.8-flash"],
        "baseline_generation_model": "claude-opus-5-5",
        "embedding_model": "gemini-embedding-2",
        "claude_sonnet_5_5_results_file": str(sonnet_json.relative_to(ROOT_DIR)),
        "gemini_3_8_flash_results_file": str(flash_json.relative_to(ROOT_DIR)),
        "summary_metrics_comparison": summary_diff,
        "per_case_comparison": combined_cases,
        "pricing_per_1m_tokens_usd": MODEL_PRICING_PER_1M,
    }
    out_path = DATA_DIR / "model_routing_comparison_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved structured comparison report to {out_path}")


if __name__ == "__main__":
    generate_routing_report()
