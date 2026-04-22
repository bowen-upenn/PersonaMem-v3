# PersonaMem-v3 Evaluation

## Overview

Offline evaluation harness for cross-platform personalization. Scores a recommendation / chatbot agent against each user's held-out test items already flagged in-place in [backend/{user_id}/{app}.json](backend/) (`split: "test"`). All test preferences carry LLM-picked topically-irrelevant distractors in `over_personalization_irrelevant` — the harness re-uses these for both ranking and restraint probes.

## Motivation

A recommender must be **proactive**: surface content the user would positively engage with in the **current short time window**, while explicitly avoiding content they have disliked in that same window. A response that lines up with a long-held persona trait but ignores what the user is actively into *today* is only partly correct; a response that surfaces something the user explicitly disliked *today* is actively wrong.

This drives the asymmetric ground-truth slice the harness scores against:
- **TARGET** (must match): held-out positive preference + all positive preferences from other events across all four apps within `[T_test − 24h, T_test + 24h]`.
- **AVOID** (must not surface): all negative preferences across all four apps within the same window.

AVOID leaks are treated as **hard-constraint failures** — a response gets flagged regardless of how well it scores on TARGET match.

## How to run the evaluation

**Two-phase, build-once / run-many design.** All randomness (slate composition, shuffle order, Task C scenario instantiation, C1 probe selection) is resolved in a **build** step and frozen to `benchmark/{user_id}/benchmark.json`. The **run** step performs no RNG — it consumes the frozen file. This makes results reproducible and comparable across runs, modes, and models.

**No manual test queries to write.** Everything the benchmark needs is derived from [backend/{user_id}/](backend/) produced by the persona pipeline.

### Where each task's inputs come from (frozen at build time)

| Task | Input source | Manual prep? |
|---|---|---|
| **A (slate ranking)** | Each preference with `split: "test"` in `{instagram,facebook,threads}.json`. Slate = held-out positive + pre-paired `over_personalization_irrelevant` + sampled known-negatives + randoms from unused hashtags. | None |
| **B (chatbot response)** | Each preference with `split: "test"` in `chatbot.json`. User query read from `interaction_format.user_message` (or the last user turn). Same-day TARGET/AVOID slice also frozen at build time. | None |
| **C1 (repetition fatigue)** | Top saturated hashtags via `hashtag_summary` + 5–7 recent events each. | None |
| **C2 (scenario library)** | Five templates in [evaluation/scenarios.py](evaluation/scenarios.py) (sympathy card, educated rejection, tax question, ask-to-forget, third-party gift), instantiated per-user from the user's own top preferences, negatives, and carve-outs. | None — templates are in the repo |
| **C3 (irrelevant-distractor restraint)** | Each Task A test preference's held-out positive + its `over_personalization_irrelevant` list, shuffled. | None |

Each instance carries a stable `test_id` / `probe_id` / `scenario_id` plus enough ground-truth fields (held-out position, origin labels, irrelevant set, TARGET/AVOID slice) for scoring. Per-item seeding means adding or removing one test item doesn't cascade-shift every other slate.

### Reproducibility

- The benchmark file records `benchmark_version`, `rng_seed`, `built_at`, and `backend_hash` (hash of the five backend JSONs). At run time, the harness refuses to run if the current `backend_hash` doesn't match the benchmark's — rebuild the benchmark or pass `--allow_stale` to run the frozen inputs anyway.
- Two runs of the same config against the same benchmark file produce identical inputs. Results differ only by stochastic LLM output (controlled by the agent's sampling settings).
- Mode-A vs Mode-B and model-A vs model-B comparisons are valid: every run sees the same slates, scenarios, queries, and GT slices.

### Workflow

```bash
# 0. Build the benchmark once per user. Deterministic given --rng_seed and the backend data.
python -m evaluation.build_benchmark --user_id 115
# → writes benchmark/115/benchmark.json

# 1. Sanity check (no LLM cost). Confirms the harness loads the frozen instances and builds prompts.
python -m evaluation.run_inference --user_id 115 --mode llm_longctx --task all --dry_run

# 2. A small real run first (1 instance per task) to confirm model + keys + parsing.
python -m evaluation.run_inference --user_id 115 --mode agent_tools --task all --limit 1

# 3. Full run per mode. All three runs score identical inputs — directly comparable.
python -m evaluation.run_inference --user_id 115 --mode agent_tools    --task all
python -m evaluation.run_inference --user_id 115 --mode agent_longctx  --task all
python -m evaluation.run_inference --user_id 115 --mode llm_longctx    --task all

# 4. Optional LLM-judge layer on any mode.
python -m evaluation.run_inference --user_id 115 --mode agent_tools --task all --enable_llm_judge
```

If the persona pipeline reprocesses a user (backend data changes), rerun step 0 to refresh the benchmark. The `backend_hash` guard will tell you when this is needed.

Results land in `benchmark/{user_id}/runs/{timestamp}/` — per-task JSONs, `summary.json`, and `summary.md`. Each summary records the benchmark version + hash + seed so you can tell which frozen inputs a given result set corresponds to.

### Prerequisites

- `backend/{user_id}/` populated by the persona pipeline (one subfolder per user).
- API keys in `.env` for the chosen `--model` and `--judge_model` — reuses the existing `query_llm.py` multi-provider setup (Azure / OpenAI / Claude / Gemini).
- Optional: `pip install sentence-transformers tiktoken` for the best Task B similarity scoring and accurate token counts in long-context modes. Without them, the harness falls back to camelCase-aware lexical Jaccard + char/4 token estimates and logs a warning.

### When would you prepare something manually?

Only if you want to **add a new Task C scenario** (e.g., your own probe). Drop a builder into [evaluation/scenarios.py](evaluation/scenarios.py) `SCENARIO_BUILDERS` — it takes `(bq, user_id, since_timestamp, rng)` and returns `{name, query, notes, forbidden_items, carve_out}`. Then rebuild the benchmark; the new scenario instance is frozen into the file and picked up automatically.

## Tasks

All tasks share a single time-gated view: for each test moment `T_test`, events with `source_timestamp >= T_test` are masked across all four apps.

### Task A — Cross-app slate ranking (Instagram, Facebook, Threads)
- **Input**: for each social-app test preference, build a K=10 slate = `1× held-out positive + 3× irrelevant (from over_personalization_irrelevant) + 3× known-disliked + 3× plausible-random`. Shuffled; agent sees only the slate, no labels.
- **Agent output**: permutation of indices (most → least likely positive engagement).
- **Hard metrics**: Recall@{1,3,5}, NDCG@K, MRR, Hit@K, intra-list diversity, rate of negatives / irrelevants landing in top-1 / top-3.
- **Judge (opt-in)**: when held-out positive is not top-1, scores whether the agent's top-1 pick is itself preference-aligned (0–3).

### Task B — Chatbot personalized response (generative only)
- **Input**: for each chatbot test event, the user query + any pre-`T_test` conversation turns + cross-app memory.
- **Agent output**: free-form response.
- **Hard metrics** (scored against the TARGET/AVOID same-day slice):
  - **held_out_hit** (primary): did the response match the held-out positive at cosine ≥ 0.5?
  - **target_match_recall**: fraction of TARGET items (held-out + same-day positives) the response matched.
  - **avoid_leak_rate** / **avoid_leak_flag** (hard constraint): any match against AVOID items is a failure.
  - **carve_out_respect**: for `asked_not_to_personalize` events, does the response steer clear of the carved-out topic?
  - **distance_from_evidence_bin**: stratification by token-distance to nearest supporting train event.
- **Judge (opt-in)**: G-Eval rubric with polarity-aware prompt (positive-alignment vs refuse-to-personalize), plus helpfulness / restraint / hallucination dimensions.

### Task C — Over-personalization & back-off probes
- **C1. Repetition fatigue** (constructed): saturate an app with 5–7 same-hashtag items in 24h, ask for the next recommendation. Hard metric: `diversification_rate` (fraction of new hashtags that are not the saturated one).
- **C2. Scenario library** (constructed per-user): sympathy card (socially inappropriate), category-but-steer-away (educated rejection), out-of-domain tax question, ask-to-forget follow-up, third-party gift request. Hard metric: `keyword_leak_rate` against forbidden preferences + `carve_out_respect`. Optional judge: `restraint_score` (0–3).
- **C3. Irrelevant-distractor restraint**: shuffled mix of held-out positive + the test item's `over_personalization_irrelevant` list — agent must identify which signals the system should refuse to apply. Hard metrics: `irrelevant_rejection_recall`, `precision`, and whether the held-out positive was wrongly rejected.

### Task D — Aggregate negative avoidance
Rolled up from Task A — no separate run. Reports `negative_in_top1_rate`, `negative_in_top3_rate`, `irrelevant_in_top1_rate` across all Task A test moments.

## Modes

| Mode | Runner | Backend access | What it isolates |
|---|---|---|---|
| `agent_tools` | Real **Claude Code subagent** via `claude -p` (uses your subscription auth) | Read-only into a **time-masked filesystem snapshot** at `/tmp/pm3_eval_snapshots/{user_id}/T_{t_test}/` | Claude Code's actual filesystem-agent behavior |
| `agent_longctx` | Same Claude Code subagent, **no tools** (`allowed_tools=()`) | Full pre-`T_test` history pre-loaded in the prompt | Claude Code framework effect without filesystem retrieval |
| `llm_longctx` | Direct single `QueryLLM.query_llm` call (Azure/OpenAI/Claude/Gemini) | Full history concatenated + per-app token annotations | Pure long-context baseline, no agent framework |

Running all three answers: (a) does Claude Code's filesystem agentic retrieval beat stuffing history? and (b) does the Claude Code framework add value over a plain LLM call?

### How the `agent_tools` sandbox works

Each test moment, the harness **materializes a filesystem snapshot** from the backend:
1. Write filtered per-app JSONs (events with `source_timestamp < T_test`; leak-sensitive fields like `split`, `over_personalization_irrelevant`, `update_history` stripped) to `/tmp/pm3_eval_snapshots/{user_id}/T_{t_test}/`.
2. Write a `README.md` inside the snapshot that enumerates the files (the subagent has Read only, no Glob/Grep).
3. Spawn `claude -p <prompt>` with `cwd = snapshot_dir`, `--setting-sources ""` (blocks inheritance from parent Claude Code session's permissive config), `--allowedTools "Read(/<abs>/**)"` (path-scoped permission), `--disallowedTools Bash,Edit,Write,WebFetch,WebSearch,Task,NotebookEdit`, and `--permission-mode dontAsk`.

Three layers of sandbox, all required — any one alone is bypassable:
- **`cwd`** scopes relative reads to the snapshot.
- **Path-scoped `Read(//abs/**)`** blocks absolute-path reads outside the snapshot.
- **`--setting-sources ""`** stops the subagent from inheriting the caller's permissive `allow` rules (we verified: without this, `Read(/etc/passwd)` succeeded silently).
- **Snapshot lives under `/tmp/`** so Claude Code can't walk upward to the project's `.git` and leak the real `backend/` path via its dynamic system prompt.

Verified end-to-end against canaries: reads of `CLAUDE.md`, `/etc/passwd`, and the real `backend/115/persona.html` are all denied (appear in `permission_denials` field); reads of files *inside* the snapshot succeed.

### Why Read-only (no Bash/Glob/Grep)

Headless `claude -p` mode exposes Read, Bash, Edit, Write, Task, Web*, etc. — but **not Glob/Grep as separate tools** (those are interactive-session-only). Real Claude Code users navigate the filesystem via Bash (`find`, `ls`, `grep`). Allowing Bash, even narrowly-scoped, leaves a trivial escape: `cat /etc/passwd`. Bash pattern-matching scopes command names (e.g. `Bash(git *)`) but not file arguments, and Claude Code doesn't offer a Linux chroot/namespace.

So the harness restricts to **Read only**, and the snapshot's `README.md` tells the agent exactly which files exist — no enumeration needed. This matches how a real Claude Code user would work with a well-documented project root.

## Flags reference

| Flag | Default | Meaning |
|---|---|---|
| `--user_id` | _(required)_ | User directory under `backend/` |
| `--backend_dir` | `backend` | Path to backend root |
| `--mode` | `llm_longctx` | One of `agent_tools`, `agent_longctx`, `llm_longctx` |
| `--task` | `all` | `all`, `a`, `b`, `c`, `c1`, `c2`, `c3`, or explicit task name |
| `--limit` | _none_ | Cap items per task (for fast iteration) |
| `--enable_llm_judge` | off | Turn on LLM-as-judge layer (optional) |
| `--model` | `$EVAL_MODEL` or `gpt-5-chat` | Baseline model for `llm_longctx` mode (QueryLLM backend: Azure/OpenAI/Claude/Gemini) |
| `--claude_model` | `$EVAL_CLAUDE_MODEL` or `sonnet` | Claude Code subagent model for `agent_tools` / `agent_longctx` (`haiku`, `sonnet`, `opus`) — uses your Claude Code subscription |
| `--judge_model` | `$EVAL_JUDGE_MODEL` or `claude-opus` | Judge model (only used with `--enable_llm_judge`) |
| `--slate_k` | `$EVAL_SLATE_K` or `10` | Slate size K for Task A |
| `--context_budget` | _none_ | Token budget for long-context modes; exceeds → per-app reservoir-sample with warning |
| `--rate_limit` | `50` | LLM rate limit per minute |
| `--dry_run` | off | Build prompts without LLM calls |
| `--output_dir` | auto (`benchmark/{user_id}/runs/`) | Results root |
| `--benchmark` | auto | Path to frozen benchmark JSON (default: `benchmark/{user_id}/benchmark.json`) |
| `--allow_stale` | off | Run even if backend_hash has drifted since the benchmark was built |

Benchmark-building is its own CLI:

| Flag | Default | Meaning |
|---|---|---|
| `--user_id` | _(required)_ | User id to build for |
| `--backend_dir` | `backend` | Path to backend root |
| `--rng_seed` | `0` | Deterministic seed (per-instance sub-seeds derived from it) |
| `--output` | auto | Output path (default: `benchmark/{user_id}/benchmark.json`) |

## Outputs

Results land under `benchmark/{user_id}/runs/{timestamp}/`:

- `slate_ranking.json`, `chatbot_response.json`, `c1_fatigue.json`, `c2_scenarios.json`, `c3_restraint.json`, `d_negative_avoidance.json` — per-task row arrays.
- `summary.json` — mean hard metrics per task.
- `summary.md` — human-readable Markdown summary.

Per-row schema:
```json
{
  "task": "...",
  "user_id": "115",
  "test_id": "...",
  "mode": "agent_tools",
  "agent_response": "...",
  "tool_calls": 3,
  "history_tokens": 47312,
  "metrics": { "...": ... }
}
```

## Interpreting metrics

- **Task A**: `hit@1 > 0.1` (random baseline) is the floor; good agents reach `hit@3 > 0.5`, `mrr > 0.3`.
- **Task B**: `avoid_leak_rate → 0` is non-negotiable — any sustained leak means the agent is surfacing things the user just said they disliked. `held_out_hit` and `target_match_recall` rise together as the agent gets better at reading contemporaneous signals.
- **Task C1**: `diversification_rate → 1` — if the agent returns more saturated hashtags, it's reinforcing the fatigue.
- **Task C2**: `keyword_leak_rate → 0`, `carve_out_respect → 1`.
- **Task C3**: `irrelevant_rejection_recall → 1`, `held_out_wrongly_rejected → 0`.
- **Task D**: `negative_in_top1_rate → 0`, `irrelevant_in_top1_rate → 0`.
- **Judge scores** (opt-in): typical frontier models land in the 2.0–2.5 range on the 0–3 rubrics; 2.5+ is strong.

## Extending the harness

- **New task**: add `evaluation/tasks/<name>.py` with a `run_task_*` function matching the common signature in [evaluation/run_inference.py](evaluation/run_inference.py); register it in `_run_task` and `TASK_ALIASES`.
- **New scenario (Task C)**: add a builder to [evaluation/scenarios.py](evaluation/scenarios.py) `SCENARIO_BUILDERS`. Each builder reads from `BackendQuery` and returns `{name, query, notes, forbidden_items, carve_out}`.
- **New mode**: add a branch in the task drivers' `if mode == ...` blocks and register the name in `MODES`. Both tool-driven and long-context modes reuse the same `SnapshotCache`.
- **New judge dimension**: add a rubric function to [evaluation/judges.py](evaluation/judges.py) and wire it into the relevant task driver. Judges always receive the focused evidence slice from `build_judge_evidence` — never the full history.

EVAL.md is maintained alongside the code: any change to tasks, modes, metrics, or CLI flags must be reflected here (same convention as [DESIGN.md](DESIGN.md) and [CLAUDE.md](CLAUDE.md)).
