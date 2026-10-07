# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADK SWE Coding Agent for evaluating candidate models (Gemini 3.8 Flash vs Claude Sonnet 5.5)."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.anthropic_llm import Claude
from google.adk.models.registry import LLMRegistry
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_REPO = ROOT_DIR / "target_repo"


class VertexClaude(Claude):
    """Subclasses ADK Claude to register all claude-* model versions including 5.x."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"claude-.*"]


LLMRegistry.register(VertexClaude)

MODEL_NAME = os.environ.get("SWE_AGENT_MODEL", "gemini-3.8-flash")


def _build_llm(model_name: str):
    if model_name.startswith("claude"):
        return VertexClaude(model=model_name, max_tokens=4096)
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def read_repo_file(file_path: str) -> dict:
    """Read the contents of a file in the target repository at the pre-PR baseline commit.

    Args:
        file_path: Relative path to the file inside the repository, e.g. 'orderflow/pricing.py'.

    Returns:
        Dictionary with 'file_path' and 'content' (or 'error' if not found).
    """
    clean_rel = file_path.strip().lstrip("/")
    target = (TARGET_REPO / clean_rel).resolve()
    if not str(target).startswith(str(TARGET_REPO.resolve())) or not target.is_file():
        return {"file_path": clean_rel, "error": f"File not found: {clean_rel}"}
    return {
        "file_path": clean_rel,
        "content": target.read_text(encoding="utf-8"),
    }


def test_and_verify_patch(
    file_path: str,
    old_string: str,
    new_string: str,
    pytest_target: str = "",
) -> dict:
    """Apply a proposed code edit in an isolated sandbox copy of the repo and run pytest.

    Args:
        file_path: Relative path to the file to modify, e.g. 'orderflow/pricing.py'.
        old_string: Exact existing substring in the file to replace.
        new_string: Replacement code string.
        pytest_target: Optional pytest -k test function filter, e.g. 'test_calculate_tax_round_half_up'.

    Returns:
        Dictionary with 'status' ('PASSED' or 'FAILED'), 'unified_diff', and 'pytest_output'.
    """
    clean_rel = file_path.strip().lstrip("/")
    src_file = (TARGET_REPO / clean_rel).resolve()
    if not str(src_file).startswith(str(TARGET_REPO.resolve())) or not src_file.is_file():
        return {"status": "ERROR", "error": f"File not found: {clean_rel}"}

    original_text = src_file.read_text(encoding="utf-8")
    if old_string not in original_text:
        return {
            "status": "ERROR",
            "error": "old_string not found in target file. Read the file first and match exact indentation.",
        }

    patched_text = original_text.replace(old_string, new_string, 1)
    unified_diff = "\n".join(
        difflib.unified_diff(
            original_text.splitlines(),
            patched_text.splitlines(),
            fromfile=f"a/{clean_rel}",
            tofile=f"b/{clean_rel}",
            lineterm="",
        )
    )

    with tempfile.TemporaryDirectory(prefix="swe_eval_sandbox_") as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        shutil.copytree(
            TARGET_REPO,
            tmp_repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )
        (tmp_repo / clean_rel).write_text(patched_text, encoding="utf-8")

        cmd = [sys.executable, "-m", "pytest", "tests/test_orderflow.py", "-q"]
        if pytest_target:
            cmd.extend(["-k", pytest_target])

        proc = subprocess.run(
            cmd,
            cwd=tmp_repo,
            capture_output=True,
            text=True,
            timeout=20,
        )
        passed = proc.returncode == 0
        return {
            "status": "PASSED" if passed else "FAILED",
            "exit_code": proc.returncode,
            "unified_diff": unified_diff,
            "pytest_output": (proc.stdout + "\n" + proc.stderr).strip()[:1500],
        }


root_agent = Agent(
    name="root_agent",
    model=_build_llm(MODEL_NAME),
    instruction=(
        "You are an expert software engineering coding agent. "
        "For every task:\n"
        "1. Call `read_repo_file` on the target file mentioned in the prompt.\n"
        "2. Call `test_and_verify_patch` with the exact `old_string`, `new_string`, and `pytest_target` "
        "to verify your fix passes the unit test suite.\n"
        "3. In your final response, include the exact unified diff inside a ```diff code block "
        "and confirm the test status."
    ),
    tools=[read_repo_file, test_and_verify_patch],
)

app = App(
    root_agent=root_agent,
    name="app",
)
