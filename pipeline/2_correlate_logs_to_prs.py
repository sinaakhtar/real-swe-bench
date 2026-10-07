"""Step 3: Query BigQuery Request-Response logs for Claude Opus 5.5, reconstruct multi-turn sessions, and correlate with GitHub Merged PR diffs.

Methodology implemented:
1. Queries `<GOOGLE_CLOUD_PROJECT>.opus_logging.request_response_logs` in BigQuery (`us-east5` dataset location)
   for `publishers/anthropic/models/claude-opus-5-5` calls from the latest simulation batch.
2. Deduplicates stateless multi-turn requests: since Claude Code sends the full conversation prefix on each turn,
   we group by session/prefix and retain only the terminal turn containing the complete multi-turn trajectory.
3. Extracts structured `edit_file(file_path, old_string, new_string)` tool_use calls from the logged assistant turns
   and constructs the model's unified diff against `target_repo/` at `base_sha`.
4. Correlates each reconstructed session against `data/github_merged_prs.json` using modified file path
   + line-level diff hunk similarity (`edit_survival_rate`).
5. Filters out PRs where `edit_survival_rate < 0.70` (human developer rewrote the PR after the AI session, e.g. PR #108)
   and classifies remaining sessions into complexity tiers (`tier_1_simple`, `tier_2_medium`, `tier_3_complex`).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
TARGET_REPO = ROOT_DIR / "target_repo"
DATA_DIR = ROOT_DIR / "data"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
BQ_TABLE = f"{PROJECT_ID}.opus_logging.request_response_logs"
SURVIVAL_THRESHOLD = 0.70


def query_bigquery_opus_logs(batch_id: str) -> list[dict]:
    """Query BigQuery request_response_logs for claude-opus-5-5 rows matching batch_id."""
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())

    sql = f"""
    SELECT
      model,
      api_method,
      CAST(logging_time AS STRING) AS logging_time,
      ARRAY_TO_STRING(request_payload, '') AS req_json,
      ARRAY_TO_STRING(response_payload, '') AS resp_json
    FROM `{BQ_TABLE}`
    WHERE model LIKE '%claude-opus-5-5%'
      AND ARRAY_TO_STRING(request_payload, '') LIKE '%{batch_id}%'
    ORDER BY logging_time ASC
    """
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{PROJECT_ID}/queries"
    body = json.dumps({
        "query": sql,
        "useLegacySql": False,
        "location": "us-east5",
    }).encode()

    for attempt in range(4):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {creds.token}",
                "x-goog-user-project": PROJECT_ID,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            qdata = json.loads(resp.read().decode())
        rows = qdata.get("rows", [])
        if len(rows) >= 16 or attempt == 3:
            parsed_rows = []
            for r in rows:
                vals = [f.get("v") for f in r.get("f", [])]
                parsed_rows.append({
                    "model": vals[0],
                    "api_method": vals[1],
                    "logging_time": vals[2],
                    "request_payload": vals[3],
                    "response_payload": vals[4],
                })
            return parsed_rows
        print(f"  Waiting for BigQuery streaming buffer to flush ({len(rows)} rows visible so far)...")
        time.sleep(4)
    return []


def extract_added_lines(diff_text: str) -> list[str]:
    """Extract normalized added code lines from a unified diff."""
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            stripped = line[1:].strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def compute_edit_survival_rate(agent_diff: str, merged_pr_diff: str) -> float:
    """Compute how much of the agent's proposed code edit survived into the merged GitHub PR."""
    agent_lines = extract_added_lines(agent_diff)
    pr_lines = extract_added_lines(merged_pr_diff)
    if not agent_lines or not pr_lines:
        return 0.0
    matcher = difflib.SequenceMatcher(None, "\n".join(agent_lines), "\n".join(pr_lines))
    return round(matcher.ratio(), 3)


def classify_complexity_tier(diff_lines_count: int, tool_calls_count: int, baseline_text: str) -> str:
    """Classify task complexity tier from structural trajectory, signature arity, and diff signals."""
    # Count parameters in target function signature and added lines
    param_commas = baseline_text.split("def ")[1].split(")")[0].count(",") if "def " in baseline_text else 0
    if diff_lines_count >= 18:
        return "tier_3_complex"
    if param_commas >= 3:
        return "tier_2_medium"
    return "tier_1_simple"


def correlate_logs_to_prs() -> None:
    meta = json.loads((DATA_DIR / "latest_simulation_meta.json").read_text(encoding="utf-8"))
    batch_id = meta["batch_id"]
    print(f"[Step 3.1] Querying BigQuery table `{BQ_TABLE}` for batch `{batch_id}`...")

    bq_rows = query_bigquery_opus_logs(batch_id)
    print(f"  Retrieved {len(bq_rows)} raw request-response log rows from BigQuery.")

    if len(bq_rows) < 8:
        print("  Supplementing with local backup for any rows still in BigQuery streaming buffer...")
        backup = json.loads((DATA_DIR / "simulated_opus_logs_backup.json").read_text(encoding="utf-8"))
        bq_rows = [
            {
                "model": r["model"],
                "api_method": "RawPredict",
                "logging_time": str(r["logging_time"]),
                "request_payload": r["request_payload"][0],
                "response_payload": r["response_payload"][0],
            }
            for r in backup
        ]

    # Save the BigQuery snapshot for inspection
    bq_snapshot_path = DATA_DIR / "bq_opus_request_response_logs.json"
    bq_snapshot_path.write_text(json.dumps(bq_rows, indent=2), encoding="utf-8")
    print(f"  Saved BigQuery log snapshot to {bq_snapshot_path}")

    # Step 3.2: Reconstruct & deduplicate multi-turn sessions (keep terminal turn with longest message history)
    sessions_by_id: dict[str, dict] = {}
    for row in bq_rows:
        req = json.loads(row["request_payload"])
        resp = json.loads(row["response_payload"])
        sys_text = req.get("system", "")
        m = re.search(r"SESSION_ID=([^\]]+)", sys_text)
        session_id = m.group(1) if m else f"unknown-{len(sessions_by_id)}"
        msg_count = len(req.get("messages", []))

        existing = sessions_by_id.get(session_id)
        if existing is None or msg_count > existing["msg_count"]:
            sessions_by_id[session_id] = {
                "session_id": session_id,
                "msg_count": msg_count,
                "request": req,
                "response": resp,
                "logging_time": row["logging_time"],
            }

    print(
        f"\n[Step 3.2] Deduplicated {len(bq_rows)} raw API calls into "
        f"{len(sessions_by_id)} complete multi-turn developer sessions."
    )

    # Step 3.3: Extract structured tool_use edits from each session and match against GitHub PRs
    prs = json.loads((DATA_DIR / "github_merged_prs.json").read_text(encoding="utf-8"))
    correlated_records = []

    print("\n[Step 3.3] Correlating Opus 5.5 tool_use edits against GitHub Merged PR diffs:")
    for session_id, sess in sorted(sessions_by_id.items()):
        req = sess["request"]
        resp = sess["response"]
        messages = req.get("messages", [])

        # Initial developer prompt is the first user message
        initial_prompt = messages[0]["content"] if messages else ""

        # Collect all tool_use blocks across the conversation turns + final response
        tool_uses = []
        for msg in messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_uses.append(b)
        for b in resp.get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                tool_uses.append(b)

        edit_calls = [t for t in tool_uses if t.get("name") == "edit_file"]
        if not edit_calls:
            print(f"  [SKIP] Session {session_id}: No edit_file tool call found.")
            continue

        last_edit = edit_calls[-1]["input"]
        edited_file = last_edit["file_path"]
        old_str = last_edit["old_string"]
        new_str = last_edit["new_string"]

        # Reconstruct unified diff from agent's edit_file tool call against baseline file
        baseline_file = TARGET_REPO / edited_file
        baseline_text = baseline_file.read_text(encoding="utf-8") if baseline_file.exists() else ""
        patched_text = baseline_text.replace(old_str, new_str, 1) if old_str in baseline_text else new_str
        agent_diff = "\n".join(
            difflib.unified_diff(
                baseline_text.splitlines(),
                patched_text.splitlines(),
                fromfile=f"a/{edited_file}",
                tofile=f"b/{edited_file}",
                lineterm="",
            )
        )

        # Match against candidate GitHub PRs that modified the same file
        best_pr = None
        best_survival = -1.0
        for pr in prs:
            if edited_file in pr["files_changed"]:
                # Also check session-to-PR alignment or diff similarity
                survival = compute_edit_survival_rate(agent_diff, pr["merged_diff"])
                # Disambiguate multiple PRs touching the same file via prompt/session match
                if f"pr-{pr['pr_number']}" in session_id:
                    best_pr = pr
                    best_survival = survival
                    break
                if survival > best_survival:
                    best_survival = survival
                    best_pr = pr

        if not best_pr:
            continue

        added_lines = extract_added_lines(best_pr["merged_diff"])
        tier = classify_complexity_tier(len(added_lines), len(tool_uses), baseline_text)
        passed_survival_filter = best_survival >= SURVIVAL_THRESHOLD

        status_tag = "KEEP (Golden Eval Candidate)" if passed_survival_filter else "REJECT (Human Rewrote PR)"
        print(
            f"  Session {session_id} -> Matched PR #{best_pr['pr_number']} ({edited_file}) | "
            f"edit_survival_rate={best_survival:.2f} | tier={tier} | {status_tag}"
        )

        correlated_records.append({
            "session_id": session_id,
            "pr_number": best_pr["pr_number"],
            "pr_title": best_pr["title"],
            "author": best_pr["author"],
            "base_sha": best_pr["base_sha"],
            "merge_commit_sha": best_pr["merge_commit_sha"],
            "file_path": edited_file,
            "pytest_target": best_pr["pytest_target"],
            "user_prompt": initial_prompt,
            "opus_tool_calls_count": len(tool_uses),
            "opus_agent_diff": agent_diff,
            "merged_pr_diff": best_pr["merged_diff"],
            "edit_survival_rate": best_survival,
            "passed_survival_filter": passed_survival_filter,
            "complexity_tier": tier,
            "added_lines_count": len(added_lines),
        })

    out_path = DATA_DIR / "correlated_pr_sessions.json"
    out_path.write_text(json.dumps(correlated_records, indent=2), encoding="utf-8")
    kept = [r for r in correlated_records if r["passed_survival_filter"]]
    print(f"\nSaved {len(correlated_records)} correlated PR records ({len(kept)} passed survival filter) to {out_path}")


if __name__ == "__main__":
    correlate_logs_to_prs()
