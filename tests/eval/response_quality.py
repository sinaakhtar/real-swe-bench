"""LLM-as-judge for semantic equivalence against the merged GitHub PR diff (`patch_semantic_equivalence`)."""

from __future__ import annotations

import os
from google import genai
from google.genai import types
from pydantic import BaseModel


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def evaluate(instance: dict) -> dict:
    reference = instance.get("reference")
    prompt = (
        "You are a Staff Software Engineer evaluating an AI coding agent's proposed code patch "
        "against the golden merged GitHub PR diff.\n"
        "Score the agent's patch on a 1 to 5 scale:\n"
        "- 5: Semantically equivalent to the merged PR diff, handles all specified edge cases, clean idiomatic code.\n"
        "- 4: Solves the core bug/feature and passes tests with minor stylistic differences.\n"
        "- 3: Partially addresses the issue but misses a minor edge case.\n"
        "- 2: Incorrect logic or introduces regressions.\n"
        "- 1: Failed to produce a valid patch.\n\n"
        f"Developer Task Prompt:\n{instance.get('prompt', '')}\n\n"
        f"Agent Final Response:\n{instance.get('response', '')}\n\n"
        f"Golden Merged GitHub PR Diff (Ground Truth Reference):\n{reference}\n\n"
        f"Full Agent Tool Trajectory:\n{instance.get('agent_data', '')}\n"
    )

    from dotenv import load_dotenv
    import google.auth

    load_dotenv()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
    client = genai.Client(vertexai=True, project=project, location="global")
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_Verdict,
        ),
    )
    verdict = response.parsed
    if verdict is None:
        return {"score": 0, "explanation": response.text or ""}
    return {"score": max(1, min(5, verdict.score)), "explanation": verdict.explanation}
