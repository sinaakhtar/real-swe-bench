# SWE model routing evaluation pipeline

This repository demonstrates an end-to-end pipeline for building an internal software engineering benchmark from real developer coding sessions and evaluating lower-cost candidate models on Vertex AI using `agents-cli eval`.

Instead of relying on static public benchmarks, this workflow captures real developer interactions with a frontier model (`claude-opus-5-5`), joins those trajectories against merged GitHub pull requests, filters out human-rewritten PRs, clusters tasks semantically with `gemini-embedding-2`, and evaluates candidate models (`gemini-3.8-flash` and `claude-sonnet-5-5`) across task complexity tiers.

## Architecture and five-step workflow

1. **Enable Vertex AI request-response logging (`pipeline/1_simulate_opus_dev_sessions.py`).**
   Configures `PublisherModelConfig` on Vertex AI for `publishers/anthropic/models/claude-opus-5-5` so all API requests and tool calls stream into BigQuery (`<GOOGLE_CLOUD_PROJECT>.opus_logging.request_response_logs`).
2. **Track merged GitHub PRs and baseline repo state (`pipeline/0_setup_target_repo.py`).**
   Initializes `target_repo/` (`orderflow`, a Python pricing, validation, discount, webhook, and multi-currency settlement package) with a pre-PR baseline commit (`base_sha`), unit tests (`tests/test_orderflow.py`), and 8 merged PR branches (`pr-101` through `pr-108`).
3. **Correlate BigQuery logs to merged PR diffs (`pipeline/2_correlate_logs_to_prs.py`).**
   Queries BigQuery for raw `claude-opus-5-5` logs, deduplicates multi-turn prefix requests to keep only the terminal turn per session, extracts `edit_file` tool calls, and computes `edit_survival_rate` against the merged GitHub PR diffs. Sessions where a human rewrote the code before merging (`edit_survival_rate < 0.70`, such as `PR #108`) are rejected.
4. **Deduplicate and stratify with `gemini-embedding-2` (`pipeline/3_curate_eval_dataset.py`).**
   Embeds all passing `(prompt, file_path, merged_diff)` records into 3072-dimensional vectors using `gemini-embedding-2` on Vertex AI, computes the cosine similarity matrix, drops near-duplicate bugfixes (`similarity >= 0.78`), and produces a stratified `EvaluationDataset` across `tier_1_simple`, `tier_2_medium`, and `tier_3_complex` in `tests/eval/datasets/swe_routing_dataset.json`.
5. **Evaluate and compare models with `agents-cli eval` (`pipeline/4_compare_and_route_report.py`).**
   Runs the ADK coding agent (`app/agent.py`) over the curated dataset using `gemini-3.8-flash` and `claude-sonnet-5-5`, grades both runs with deterministic `pytest` sandbox execution (`pytest_unit_test_pass`), LLM-as-judge diff equivalence (`patch_semantic_equivalence`), and Vertex AI Eval Service multi-turn metrics (`multi_turn_task_success`, `multi_turn_tool_use_quality`), then compares them with `agents-cli eval compare`.

## Repository layout

* `target_repo/`: Git repository containing the `orderflow` Python package, unit tests, baseline commit, and 8 merged PR branches.
* `pipeline/0_setup_target_repo.py`: Builds `target_repo/` and exports `data/github_merged_prs.json`.
* `pipeline/1_simulate_opus_dev_sessions.py`: Enables BigQuery request-response logging on `claude-opus-5-5` and runs 8 multi-turn tool-use sessions.
* `pipeline/2_correlate_logs_to_prs.py`: Queries BigQuery, reconstructs multi-turn trajectories, computes `edit_survival_rate`, and writes `data/correlated_pr_sessions.json`.
* `pipeline/3_curate_eval_dataset.py`: Embeds tasks with `gemini-embedding-2`, clusters near-duplicates, and writes `tests/eval/datasets/swe_routing_dataset.json`.
* `pipeline/4_compare_and_route_report.py`: Runs `agents-cli eval compare` and generates the stratified routing breakdown.
* `app/agent.py`: ADK SWE coding agent with `read_repo_file` and hermetic `test_and_verify_patch` tools. Supports `SWE_AGENT_MODEL=gemini-3.8-flash` and `SWE_AGENT_MODEL=claude-sonnet-5-5`.
* `tests/eval/eval_config.yaml`: Evaluation configuration combining deterministic `pytest` execution, reference diff grading, and Vertex AI multi-turn trajectory metrics.

## Running the full demo

```bash
# Configure Application Default Credentials (e.g. `gcloud auth application-default login`)
# or copy .env.example to .env and set your project ID:
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="true"

# Step 0: Initialize target Git repo and merged PR branches
uv run python pipeline/0_setup_target_repo.py

# Step 1 & 2: Enable BigQuery logging for claude-opus-5-5 and simulate developer sessions
uv run python pipeline/1_simulate_opus_dev_sessions.py

# Step 3: Query BigQuery logs, reconstruct sessions, and correlate with GitHub PR diffs
uv run python pipeline/2_correlate_logs_to_prs.py

# Step 4: Deduplicate with gemini-embedding-2 and build stratified dataset
uv run python pipeline/3_curate_eval_dataset.py

# Step 5a: Evaluate Gemini 3.8 Flash
SWE_AGENT_MODEL="gemini-3.8-flash" agents-cli eval generate \
  --dataset tests/eval/datasets/swe_routing_dataset.json \
  --output artifacts/traces/traces_gemini_3_8_flash.json
agents-cli eval grade \
  --traces artifacts/traces/traces_gemini_3_8_flash.json \
  --config tests/eval/eval_config.yaml \
  --output artifacts/grade_results/gemini_3_8_flash

# Step 5b: Evaluate Claude Sonnet 5.5
agents-cli run --stop-server || true
SWE_AGENT_MODEL="claude-sonnet-5-5" agents-cli eval generate \
  --dataset tests/eval/datasets/swe_routing_dataset.json \
  --output artifacts/traces/traces_claude_sonnet_5_5.json
agents-cli eval grade \
  --traces artifacts/traces/traces_claude_sonnet_5_5.json \
  --config tests/eval/eval_config.yaml \
  --output artifacts/grade_results/claude_sonnet_5_5

# Step 5c: Compare results and generate routing table
uv run python pipeline/4_compare_and_route_report.py
```
