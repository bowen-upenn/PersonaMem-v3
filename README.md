# PersonaMem-v3

## Setup

```bash
docker build -t personamem-v3 .
docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
```

For API mode only, copy `.env.example` to `.env` and fill in Azure OpenAI or OpenAI credentials.

## Input Data

Real-world interaction data from Meta, with all user-private and personally identifiable information removed. See `data/test_interactions.csv` for the expected format:

| Column | Description | Example |
|--------|-------------|---------|
| `interaction_type` | Type of user engagement | `explicit_positive`, `explicit_negative`, `implicit_positive`, `implicit_negative` |
| `user_id` | Anonymized user identifier | `2124791` |
| `object_id` | Anonymized content identifier | `122137823030860919` |
| `interaction_time` | Unix timestamp of the interaction | `1758690616` |
| `object_text` | Hashtags associated with the content | `#RelationshipGoals #exproblems ...` |
| `dataset` | Source dataset label | `synthetic_anonymized` |
| `ds` | Date partition | `2026-02-12` |

## Pipeline

Each user goes through 16 steps (see [skill.md](skill.md) for the full spec):

1. **Infer** atomic persona traits from hashtags with confidence scores
2. **Promote implicit negatives** — weighted net-sentiment + temporal spread
3. **Cross-reference** across interaction rows — dedupe, init filter (0.5), count corroboration, discover similar/contradictory relationships
4. **Temporal graph** — organize contradictions into a timeline
5. **Update histories** — track how preferences evolve (new, reinforced, faded, expanded)
6. **User profile** — synthetic demographics, Big Five, career, education, bio
7. **Hidden personas** — infer deeper motivational layers from cross-row hashtag clustering
8. **App sub-personas** — distinct personas for Instagram, Facebook, Threads, Chatbot
9. **Build sessions** — group rows into temporal browsing sessions
10. **Route preferences to apps** — LLM-based assignment driven by per-app sub-personas
11. **Assign rows to apps** — session majority vote + 8% noise
12. **Interaction formats** — sample actions from per-app catalogs (`PLATFORM_INTERACTION_FORMATS`)
13. **Chatbot conversations** — multi-turn task-oriented conversations with implicit preference embedding
14. **Stereotype marks** — demographics-only annotation (neutral/stereotypical/anti-stereotypical)
15. **Train/test split** — time-based, cross-app, LLM inferrability-gated, with distractor pairing
16. **Save** — per-user JSON files to `backend/{uid}/`

## End-to-end workflow

Three sequential stages. Each stage is a separate script with its own
artifacts; you can rebuild any stage without touching upstream output.

```
data/test_interactions.csv
        │   stage 1 — persona generation
        ▼
backend/{uid}/{profile,instagram,facebook,threads,chatbot,calendar}.json
backend/{uid}/persona.html
        │   stage 2 — eval data preparation
        ▼
benchmark/{uid}/queries.csv
backend/{uid}/test.json
backend/{uid}/test_audit_*.{md,json}
        │   stage 3 — evaluation
        ▼
benchmark/{uid}/runs/{ts}/results.csv  +  summary.{md,json}
        │   stage 4 — aggregation
        ▼
benchmark/_aggregate/accuracy_pct_macro.md
```

### Stage 1 — generate persona data from interactions

The persona pipeline (16 steps, see [skill.md](skill.md)) reads
`data/test_interactions.csv` and writes per-user JSON to
`backend/{uid}/`.

**Claude Code subagents (default).** Open Claude Code in the project
directory and prompt:

```
Process all users in data/test_interactions.csv through the persona
pipeline following skill.md, with one subagent responsible for one
persona in parallel.
```

One parallel subagent per user; output schemas match
[`data_preparation/persona_agent.py`](data_preparation/persona_agent.py).

**API mode (Azure OpenAI / OpenAI).** Identical output format,
sequential or worker-pool parallelism:

```bash
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv --user_id 2124791
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv --model gpt-5-chat --max_workers 1
```

> ⚠️  LLM calls are expensive and concurrent edits are common. Confirm
> with collaborators before re-running the pipeline. Per-user output
> goes to `/tmp/persona_regen/{uid}.{stdout,stderr}` so a long run is
> debuggable.

Per-user files produced:

| File | Description |
|------|-------------|
| `profile.json` | Demographics + Big Five + 4 AppPersonas + flat preference list |
| `instagram.json` / `facebook.json` / `threads.json` | Per-app interaction events with nested preferences |
| `chatbot.json` | Chatbot events with multi-turn conversations + `ask_to_forget` flag |
| `calendar.json` | Calendar add/update/remove modification stream |
| `persona.html` | Self-contained interactive visualization |

### Stage 2 — prepare evaluation queries

Build the frozen benchmark + standalone query dump for review:

```bash
# Single user
python scripts/prepare_eval_data.py --user_id 115

# Range or all users (cross-user ProcessPool parallelism)
python scripts/prepare_eval_data.py --user_range 100-200 --parallel 4
python scripts/prepare_eval_data.py --all --parallel 8

# Skip the E6 LLM discovery to keep iteration fast
python scripts/prepare_eval_data.py --user_id 115 --skip_e6
```

Each call produces:

| Output | Description |
|---|---|
| `benchmark/{uid}/queries.csv` | One row per test query, ~220 instances spanning 27 task types. **Source of truth for the eval harness.** |
| `backend/{uid}/test.json` | Same queries in JSON, with normalized `ground_truth_preference`, `reference_example`, `distractor_preferences`. **Human-readable; review this.** |
| `backend/{uid}/test_audit_{phase}.md` | Optional audit report (run `audit_test_queries.py` separately). |

Per-task counts are governed by quotas in
[`evaluation/task_distribution.py`](evaluation/task_distribution.py)
(spread max:min ≈ 14:5). Caps are enforced via stratified random
sampling at the orchestrator. Floor-gap warnings indicate task types
where the source data is too sparse to reach the floor.

### Stage 3 — run evaluation

Per-persona runner:

```bash
# Single persona, mcp_agent mode (Claude Code + structured MCP tools)
scripts/run_eval.sh 115 --mode mcp_agent --claude_model sonnet

# All personas in parallel (PARALLEL=N controls concurrency, default 8)
PARALLEL=4 scripts/run_eval_all.sh --mode mcp_agent --claude_model sonnet

# Quick smoke test — first 20 rows only, no LLM judge
scripts/run_eval.sh 115 --mode mcp_agent --limit 20 --no-enable_llm_judge

# Resume an interrupted run
scripts/run_eval.sh 115 --mode mcp_agent --resume
```

Each run writes to `benchmark/{uid}/runs/{YYYYMMDD_HHMMSS}/`:
- `results.csv` — per-query agent response + metric scores
- `writes.jsonl` — agentic tool-call overlay (only for write-capable modes)
- `summary.json` / `summary.md` — per-task accuracy roll-up

#### Modes

`--mode` selects the agent setup; comparing modes isolates which
component (framework, retrieval, structured API) carries the
performance.

| Mode | Runner | Backend access | What it isolates |
|---|---|---|---|
| `mcp_agent` | Claude Code subagent + 4 mock MCP servers | Structured per-app tools (`get_feed`, `create_post`, `react`, `send_dm`, …); writes go to `writes.jsonl` overlay | Structured-API agentic behavior — comparable to real app integrations |
| `agent_tools` | Real Claude Code subagent via `claude -p` | Read-only into time-masked filesystem snapshot at `/tmp/pm3_eval_snapshots/{uid}/T_{t_test}/` | Claude Code's filesystem-agent behavior |
| `agent_longctx` | Same Claude Code subagent, **no tools** | Full pre-`T_test` history pre-loaded in the prompt | Claude Code framework effect without retrieval |
| `llm_longctx` | Direct `QueryLLM.query_llm` call (Azure / OpenAI / Claude / Gemini) | Full history concatenated with per-app token annotations | Pure long-context baseline, no agent framework |

Cross-mode comparison answers: (a) does structured MCP beat
filesystem search? (b) does Claude Code's filesystem retrieval beat
stuffing history? (c) does the agent framework add value over a
plain LLM call? See [EVAL.md](EVAL.md) for rubric dimensions and
per-task metric details.

### Stage 4 — aggregate results

Roll up per-task accuracy across all personas + modes into the
headline `accuracy_pct_macro`:

```bash
python scripts/aggregate_eval.py
```

Output: `benchmark/_aggregate/` — per-task tables, mode comparison
plots, and the macro/micro headline number.

## Auditing the eval data

Phase 1 surfaces every test query in one JSON file plus a structural
audit report. No LLM calls — uses signals already on disk.

```bash
# Dump every test query into backend/{uid}/test.json
python scripts/dump_test_json.py --user_id 115

# Audit (realism, ground-truth presence, distractor sanity, label
# honesty via the build-time blind_check_score, distribution quotas)
python scripts/audit_test_queries.py --user_id 115 --phase snapshot

# Compare two prior phase snapshots
python scripts/audit_test_queries.py --user_id 115 --diff before after
```

`prepare_eval_data.py` auto-emits `test.json` after every CSV write,
so the dump is always fresh. Re-render `persona.html` after a rebuild
so the HTML proofreading view matches the new buckets:

```python
from data_preparation.visualize import generate_persona_html
generate_persona_html("115")
```

## Code Structure

| File | Role |
|------|------|
| `data_preparation/prompts.py` | All persona-pipeline LLM prompt templates |
| `data_preparation/persona_agent.py` | PersonaAgent class, dataclasses, platform/action catalogs |
| `data_preparation/chatbot_conversation.py` | Chatbot multi-turn conversation generation |
| `data_preparation/main.py` | CSV loading, grouping, orchestration |
| `data_preparation/visualize.py` | HTML visualization + `dump_test_samples_json` |
| `query_llm.py` | Multi-provider LLM client (Azure / OpenAI / Claude / Gemini) |
| `scripts/run_persona_pipeline.py` | Stage 1 — persona generation CLI |
| `scripts/prepare_eval_data.py` | Stage 2 — eval data prep CLI |
| `scripts/run_eval.sh` / `run_eval_all.sh` | Stage 3 — evaluation runner |
| `scripts/aggregate_eval.py` | Stage 4 — cross-run aggregation |
| `scripts/dump_test_json.py` / `audit_test_queries.py` | Phase 1 audit tooling |
| `evaluation/build_benchmark.py` | Benchmark builder (called by `prepare_eval_data.py`) |
| `evaluation/task_registry.py` | Per-task metadata + query_kind / expected_behavior |
| `evaluation/task_distribution.py` | Per-task quotas + cap enforcement |
| `evaluation/audit_rules.py` | Modular audit rules (realism, label honesty, distribution) |
| `evaluation/run_eval.py` | Per-persona eval harness |
| `skill.md` | Claude Code subagent specification |
| [EVAL.md](EVAL.md) | Full eval rubric, modes, metric definitions |
