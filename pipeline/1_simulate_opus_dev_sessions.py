"""Step 1 & 2: Enable Vertex AI Request-Response Logging for Claude Opus 5.5 and simulate developer coding sessions.

What this script does:
1. Configures `PublisherModelConfig` on Vertex AI for `publishers/anthropic/models/claude-opus-5-5`
   so all rawPredict/streamRawPredict calls are logged to BigQuery:
   `bq://<GOOGLE_CLOUD_PROJECT>.opus_logging.request_response_logs`.
2. Runs 8 realistic multi-turn developer coding sessions against `claude-opus-5-5` on Vertex AI
   using Claude Code-style tools (`read_file`, `edit_file`):
   - Turn 1: Developer submits task prompt -> Opus 5.5 calls `read_file`.
   - Turn 2: Client returns baseline file content -> Opus 5.5 calls `edit_file(file_path, old_string, new_string)`.
3. Because Vertex AI Request-Response Logging is active on `claude-opus-5-5`, both Turn 1 and Turn 2
   for all 8 developer sessions are automatically captured in BigQuery (`opus_logging.request_response_logs`).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
TARGET_REPO = ROOT_DIR / "target_repo"
DATA_DIR = ROOT_DIR / "data"

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
BQ_DEST = f"bq://{PROJECT_ID}.opus_logging.request_response_logs"
OPUS_MODEL = "claude-opus-5-5"


CLAUDE_CODE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a source file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file in the repo, e.g. orderflow/pricing.py",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace old_string with new_string in a repository file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]


def get_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token


def ensure_opus_logging_enabled(location: str = "global") -> dict:
    """Enable Vertex AI Request-Response Logging on claude-opus-5-5 to BigQuery."""
    token = get_token()
    base = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
    fetch_url = f"{base}/v1beta1/projects/{PROJECT_ID}/locations/{location}/publishers/anthropic/models/{OPUS_MODEL}:fetchPublisherModelConfig"
    req_fetch = urllib.request.Request(
        fetch_url,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
    )
    try:
        with urllib.request.urlopen(req_fetch, timeout=15) as resp:
            config = json.loads(resp.read().decode())
            if config.get("loggingConfig", {}).get("enabled"):
                return config
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    set_url = f"{base}/v1beta1/projects/{PROJECT_ID}/locations/{location}/publishers/anthropic/models/{OPUS_MODEL}:setPublisherModelConfig"
    payload = {
        "publisherModelConfig": {
            "loggingConfig": {
                "enabled": True,
                "samplingRate": 1.0,
                "bigqueryDestination": {"outputUri": BQ_DEST},
                "enableOtelLogging": True,
            }
        }
    }
    req = urllib.request.Request(
        set_url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        _ = json.loads(resp.read().decode())

    with urllib.request.urlopen(req_fetch, timeout=15) as resp:
        return json.loads(resp.read().decode())


def call_opus_raw_predict(
    messages: list[dict],
    system_prompt: str,
    location: str = "global",
) -> dict:
    """Invoke claude-opus-5-5 via Vertex AI rawPredict."""
    token = get_token()
    base = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
    url = f"{base}/v1/projects/{PROJECT_ID}/locations/{location}/publishers/anthropic/models/{OPUS_MODEL}:rawPredict"
    body = {
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": messages,
        "tools": CLAUDE_CODE_TOOLS,
        "max_tokens": 1500,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e


def simulate_developer_sessions() -> None:
    print(f"[Step 1] Verifying Vertex AI Request-Response Logging for {OPUS_MODEL} -> {BQ_DEST}...")
    cfg = ensure_opus_logging_enabled("global")
    print("Active PublisherModelConfig:", json.dumps(cfg, indent=2))

    prs = json.loads((DATA_DIR / "github_merged_prs.json").read_text(encoding="utf-8"))
    batch_id = f"swe-eval-batch-{uuid.uuid4().hex[:8]}"
    print(f"\n[Step 2] Simulating {len(prs)} multi-turn developer sessions with {OPUS_MODEL} (batch_id={batch_id})...")

    local_log_records = []

    for pr in prs:
        pr_num = pr["pr_number"]
        session_id = f"{batch_id}-session-pr-{pr_num}"
        user_prompt = pr["user_prompt_ground_truth"]
        system_prompt = (
            "You are Claude Code, an autonomous software engineering assistant. "
            "First call `read_file` to inspect the target file, then call `edit_file` with exact "
            "`old_string` and `new_string` to implement the fix. "
            f"[TELEMETRY_BATCH={batch_id} SESSION_ID={session_id}]"
        )

        messages: list[dict] = [{"role": "user", "content": user_prompt}]

        # Turn 1: User prompt -> Opus calls read_file
        t1_req = {
            "anthropic_version": "vertex-2023-10-16",
            "system": system_prompt,
            "messages": json.loads(json.dumps(messages)),
            "tools": CLAUDE_CODE_TOOLS,
            "max_tokens": 1500,
        }
        t1_resp = call_opus_raw_predict(messages, system_prompt)
        local_log_records.append({
            "session_id": session_id,
            "batch_id": batch_id,
            "turn": 1,
            "model": f"publishers/anthropic/models/{OPUS_MODEL}",
            "logging_time": time.time(),
            "request_payload": [json.dumps(t1_req)],
            "response_payload": [json.dumps(t1_resp)],
        })

        # Process tool_use from Turn 1
        assistant_content = t1_resp.get("content", [])
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.get("type") == "tool_use":
                t_name = block.get("name")
                t_id = block.get("id")
                t_input = block.get("input", {})
                if t_name == "read_file":
                    fpath = t_input.get("file_path", pr["files_changed"][0])
                    full_p = TARGET_REPO / fpath
                    file_text = full_p.read_text(encoding="utf-8") if full_p.exists() else "File not found"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": t_id,
                        "content": file_text,
                    })
                elif t_name == "edit_file":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": t_id,
                        "content": "Edit applied.",
                    })

        # If Opus read the file on Turn 1, run Turn 2 so Opus calls edit_file
        has_edit = any(b.get("type") == "tool_use" and b.get("name") == "edit_file" for b in assistant_content)
        if tool_results and not has_edit:
            messages.append({"role": "user", "content": tool_results})
            t2_req = {
                "anthropic_version": "vertex-2023-10-16",
                "system": system_prompt,
                "messages": json.loads(json.dumps(messages)),
                "tools": CLAUDE_CODE_TOOLS,
                "max_tokens": 1500,
            }
            t2_resp = call_opus_raw_predict(messages, system_prompt)
            local_log_records.append({
                "session_id": session_id,
                "batch_id": batch_id,
                "turn": 2,
                "model": f"publishers/anthropic/models/{OPUS_MODEL}",
                "logging_time": time.time(),
                "request_payload": [json.dumps(t2_req)],
                "response_payload": [json.dumps(t2_resp)],
            })
            edits = [
                b.get("input", {}).get("file_path")
                for b in t2_resp.get("content", [])
                if b.get("type") == "tool_use" and b.get("name") == "edit_file"
            ]
            print(f"  Session {session_id}: Turn 1 (read_file) -> Turn 2 (edit_file on {edits})")
        else:
            print(f"  Session {session_id}: Completed in Turn 1 (has_edit={has_edit})")

    meta_path = DATA_DIR / "latest_simulation_meta.json"
    meta_path.write_text(
        json.dumps({"batch_id": batch_id, "num_raw_requests": len(local_log_records)}, indent=2),
        encoding="utf-8",
    )
    fallback_path = DATA_DIR / "simulated_opus_logs_backup.json"
    fallback_path.write_text(json.dumps(local_log_records, indent=2), encoding="utf-8")
    print(f"\nCompleted {len(local_log_records)} live Vertex AI calls to {OPUS_MODEL}.")
    print(f"Saved batch metadata to {meta_path}")


if __name__ == "__main__":
    simulate_developer_sessions()
