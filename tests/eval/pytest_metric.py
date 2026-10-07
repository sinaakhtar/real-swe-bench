"""Deterministic CodeExecutionMetric (`pytest_unit_test_pass`) that verifies the agent's patch in a sandbox."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT_DIR = (
    Path(__file__).resolve().parents[2]
    if "__file__" in globals()
    else Path.cwd().resolve()
)
TARGET_REPO = ROOT_DIR / "target_repo"


def _extract_prompt_text(instance: dict) -> str:
    prompt = instance.get("prompt")
    if isinstance(prompt, dict):
        parts = prompt.get("parts") or []
        return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if isinstance(prompt, str):
        return prompt
    turns = (instance.get("agent_data") or {}).get("turns") or []
    if turns:
        for ev in turns[0].get("events") or []:
            if ev.get("author") == "user":
                parts = (ev.get("content") or {}).get("parts") or []
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return ""


def _extract_patch_calls(instance: dict) -> list[dict]:
    calls = []
    turns = (instance.get("agent_data") or {}).get("turns") or []
    for turn in turns:
        for ev in turn.get("events") or []:
            parts = (ev.get("content") or {}).get("parts") or []
            for p in parts:
                if isinstance(p, dict) and "function_call" in p:
                    fc = p["function_call"]
                    if fc and fc.get("name") == "test_and_verify_patch":
                        calls.append(fc.get("args") or {})
    return calls


def evaluate(instance: dict) -> dict:
    prompt_text = _extract_prompt_text(instance)
    m_pytest = re.search(r"\[PYTEST_TARGET=([^\]]+)\]", prompt_text)
    pytest_target = m_pytest.group(1) if m_pytest else ""

    patch_calls = _extract_patch_calls(instance)
    if not patch_calls:
        return {
            "score": 0.0,
            "explanation": "Agent did not invoke test_and_verify_patch tool.",
        }

    last_patch = patch_calls[-1]
    file_path = (last_patch.get("file_path") or "").strip().lstrip("/")
    old_string = last_patch.get("old_string") or ""
    new_string = last_patch.get("new_string") or ""

    src_file = TARGET_REPO / file_path
    if not file_path or not src_file.is_file():
        return {
            "score": 0.0,
            "explanation": f"Invalid target file_path in patch call: {file_path!r}",
        }

    original_text = src_file.read_text(encoding="utf-8")
    if old_string not in original_text:
        return {
            "score": 0.0,
            "explanation": "old_string from test_and_verify_patch did not match baseline file.",
        }

    patched_text = original_text.replace(old_string, new_string, 1)

    with tempfile.TemporaryDirectory(prefix="swe_grade_sandbox_") as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        shutil.copytree(
            TARGET_REPO,
            tmp_repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )
        (tmp_repo / file_path).write_text(patched_text, encoding="utf-8")

        venv_bin = ROOT_DIR / ".venv" / "bin" / "python"
        venv_python = str(venv_bin) if venv_bin.exists() else sys.executable
        cmd = [venv_python, "-m", "pytest", "tests/test_orderflow.py", "-q"]
        if pytest_target:
            cmd.extend(["-k", pytest_target])

        proc = subprocess.run(
            cmd,
            cwd=tmp_repo,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if proc.returncode == 0:
            return {
                "score": 1.0,
                "explanation": f"Unit test `{pytest_target or 'all'}` PASSED ({output.splitlines()[-1] if output else 'OK'}).",
            }
        return {
            "score": 0.0,
            "explanation": f"Unit test `{pytest_target or 'all'}` FAILED: {output[:300]}",
        }
