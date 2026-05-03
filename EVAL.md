# PersonaMem-v3 Evaluation

## Overview

Offline evaluation harness for cross-platform personalization.

### Task tightening (v3 post-115 audit)

A first full mcp_agent run on user 115 surfaced several degenerate scores
(100 % saturation on three tasks, 0 % on one). Diagnosis: the *tasks* were
too easy or had unreachable thresholds, not the model. The fixes below ship
together; rerun `build_benchmark` for any user before evaluating to pick
them up.

- **`at_ai_directive_followup`** — was `t_test = t_ai + 1 s`, so the test
  reduced to "list the next post that matches the directive's hashtags."
  Now stratified across **24 h / 72 h / 7 d** lags (3 instances per
  directive, each carries `lag_bucket` so hit@1 can be broken down by lag).
  Match-Jaccard threshold tightened 0.25 → 0.15 and **2 hard distractors**
  per pool are pulled from adjacent (sub-threshold-Jaccard) hashtag
  clusters in the user's broader timeline. Pool floor raised 6 → 12.
- **Renames** — `chatbot_restraint_control` → `over_personalization_chatbot_text`;
  `irrelevant_query_restraint` → `over_personalization_distractor_reject`.
  Both task families were testing the same capability with different
  surfaces; the unified prefix makes that obvious. Old strings still work
  via `evaluation.task_registry.normalize_task_type` so historical
  `results.csv` rows continue to aggregate.
- **`over_personalization_chatbot_text`** — the prompt no longer reveals
  the answer. The previous control-arm prompt said "this query does not
  call for personalization — do not weave in the user's hobbies,
  preferences, or demographic details" which made the over-personalization
  test a tautology. The arm now uses the same neutral assistant framing as
  the proactive arm and the model must decide on its own.
- **`over_personalization_distractor_reject`** — pool size 4 → 8 (1
  held-out + 7 distractors with stratified Jaccard quotas: 2 trivial
  ≤ 0.15, 3 medium 0.15–0.40, 2 hard 0.40–0.70). Primary metric switched
  precision → **F1** (`irrelevant_rejection_f1`) so always-accept and
  always-reject both score 0.
- **`personalized_feed_ranking`** — slate's previous `negative` tier
  (3 known-disliked persona items, easy to reject by surface keyword
  match) replaced with a `hard_negative` tier: 3 events the user
  *passed over* whose hashtags overlap the held-out target by Jaccard
  ∈ [0.30, 0.60] — adjacent enough to be confusable on the surface,
  ranking gain stays 0. Slate still 16 items. New per-row metrics:
  `hard_negative_in_top1`, `hard_negative_in_top3`. Backfill: when a
  user has too few negative engagement events with matching hashtags,
  the tier falls back to the legacy persona-level negative items so
  the slate still hits 16.
- **`preference_removal_regen`** — was 0/5 because `removal_success`
  required `orig_score - regen_score ≥ 0.5` (absolute) but user 115's
  orig_score ≈ 0.009. Now: (a) build-time filter drops rows where the
  held-out preference and the candidate query share zero hashtags, and
  (b) headline metric switched to **relative** drop:
  `removal_success = 1 if (orig - regen) / max(orig, 1e-3) >= 0.5`.
  New `removal_delta_pct` field carries the relative figure. Rows where
  the model never personalized in turn 1 (`orig_score < 0.05`) emit
  `removal_status: "skipped_low_personalization"` and the aggregator
  drops them from the macro denominator instead of counting them as 0.
- **Headline** — `scripts/aggregate_eval.py` reports both
  `accuracy_pct_macro` (mean of per-task means — each task contributes
  one data point regardless of row count) and `accuracy_pct_micro`
  (n-weighted across rows). Macro is the published headline.
- **`chatbot_proactive_personalization` bucket purity** —
  `build_task_b_arms` walks every chatbot event but only events whose
  `source_object_id` is in `test_index` (the R8 selector's per-app top-N)
  carry a held-out preference. Pre-fix, the blind-check stage routed any
  candidate scoring `>= 2` into the `proactive` arm regardless, so 49 / 64
  (76 %) of user 115's `chatbot_proactive_personalization` instances
  shipped with `held_out_preference = None` — violating the bucket's
  contract that every instance is a positive personalization test (the
  card rendered with "(no held-out preference; rubric checks restraint)"
  in the supposed-to-personalize bucket).
  Fix: candidates without a held-out preference are now demoted into the
  `control` arm (their `personalization_leak_rate` against
  `top_k_relevant_prefs` is the right metric for unanchored queries), and
  `_finalize` asserts `arm in {"proactive", "contradiction"}` implies a
  non-empty `held_out_preference.persona_item` so the regression cannot
  silently re-appear. Effect on user 115:
  `chatbot_proactive_personalization` 64 → 15 (all valid) and
  `over_personalization_chatbot_text` 9 → 58; total instance count
  unchanged at 224. **Operator note**: `backend/{uid}/persona.html` is a
  rendered snapshot of `testSamples` — after rebuilding `queries.csv`
  re-render via
  `python -c "from data_preparation.visualize import generate_persona_html; generate_persona_html('{uid}')"`
  before proofreading.

**As of R8**, data-gen no longer emits `split: "test"` or `over_personalization_irrelevant`. The harness picks its own test moments dynamically from the full timeline by cutting at an arbitrary `T_test` — different tasks cut at different criteria (e.g., E2 at `@ai` directive timestamps, E3/E4 at stratified calendar days, E5 at short-term canonical mid-windows). `BackendQuery.get_events(since_timestamp=T)` time-masks the history at T_test; `BackendQuery.get_preferences(..., include_superseded=False)` additionally filters out preferences whose canonical was contradicted-and-superseded (Phase 3 cross-polarity gate, Case B) before T_test — so the ground truth at any time is the LATER stance only, never the superseded earlier one.

Two new BackendQuery helpers support Phase 4 (calendar):

- `get_calendar_modifications(user_id, since_timestamp=T)` — the CRUD modification stream time-masked at T.
- `get_calendar_state(user_id, as_of_timestamp=T)` — the folded calendar state derived from modifications with `ts ≤ T`.

Also new: `build_benchmark` now writes a flat `benchmark/{user_id}/benchmark.csv` alongside `benchmark.json`. The CSV has stable columns (`instance_id, task, user_id, t_test, t_test_iso, query, query_type, candidates_json, ground_truth_json, carveout_json, metadata_json`) suitable for HuggingFace publication. The runner continues to consume the structured JSON; the CSV is a publication-friendly projection.

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
| **A (slate ranking)** | Legacy Task A builders still read `split: "test"` from events; after R8 data regen, those paths will return zero instances until the builders are refactored to pick test moments from the full timeline. | None (refactor after regen) |
| **B (chatbot response)** | Same as A — legacy split-dependent builders need a follow-up refactor once R8 data is live. | None (refactor after regen) |
| **C1 (repetition fatigue)** | Top saturated hashtags via `hashtag_summary` + 5–7 recent events each. | None |
| **C2 (scenario library)** | Five templates in [evaluation/scenarios.py](evaluation/scenarios.py) (sympathy card, educated rejection, tax question, ask-to-forget, third-party gift), instantiated per-user from the user's own top preferences, negatives, and carve-outs. | None — templates are in the repo |
| **C3 (irrelevant-distractor restraint)** | Legacy split-dependent; same follow-up refactor note as A/B. | None (refactor after regen) |
| **E2 (@ai proactive followup)** | All events with `interaction_format.action ∈ AT_AI_ACTIONS` on social apps. **Stratified across 24 h / 72 h / 7 d lags** (3 instances per directive); each cuts timeline at `t_test = t_ai + lag`. Pool of ≥ 12 post-T_test events + 2 hard distractors from adjacent-Jaccard clusters; match-Jaccard threshold 0.15. | None |
| **E3 (multi-day daily briefing)** | 3 day-midpoints stratified by event-volume tertile (1 high/mid/low). | None |
| **E4 (Google Search) [opt-in]** | Reuses E3 day sampler. Requires `--enable_e4` at run time; live API requires `--e4_allow_live` + GOOGLE_API_KEY/GOOGLE_CSE_ID; default mode is cache replay from `benchmark/{uid}/google_search_cache/`. | None |
| **E5 (horizon lifecycle)** | Each short-term canonical (from Step 3.5 horizon classification) with `stop_condition.expected_stop_ts`. Emits paired `pre`/`post` probes; post uses Phase 4 geo + calendar context. | None |

Each instance carries a stable `test_id` / `probe_id` / `scenario_id` plus enough ground-truth fields (held-out position, origin labels, irrelevant set, TARGET/AVOID slice) for scoring. Per-item seeding means adding or removing one test item doesn't cascade-shift every other slate.

### Reproducibility

- The benchmark file records `benchmark_version`, `rng_seed`, `built_at`, and `backend_hash` (hash of the five backend JSONs). At run time, the harness refuses to run if the current `backend_hash` doesn't match the benchmark's — rebuild the benchmark or pass `--allow_stale` to run the frozen inputs anyway.
- Two runs of the same config against the same benchmark file produce identical inputs. Results differ only by stochastic LLM output (controlled by the agent's sampling settings).
- Mode-A vs Mode-B and model-A vs model-B comparisons are valid: every run sees the same slates, scenarios, queries, and GT slices.

### Workflow

```bash
# 0. Build the benchmark once per user. Deterministic given --rng_seed and the
#    backend data. Wires both LLMs (blind_check for Task B routing + E6 discovery
#    for paired warn/foil). Use --skip_blind_check / --skip_e6 for cheap rebuilds.
python scripts/prepare_eval_data.py --user_id 115
# → writes benchmark/115/queries.csv (single artifact; no JSON sidecar)

# 1. Run the eval. `run_eval.py` reads benchmark/{uid}/queries.csv and dispatches
#    each row to its task-specific runner. Output: benchmark/{uid}/runs/{ts}/results.csv.
python -m evaluation.run_eval --user_id 115 --mode mcp_agent --claude_model sonnet
# `--mode` ∈ {mcp_agent, agent_tools, agent_longctx, llm_longctx}; see "Modes" below.

# 2. Aggregate the results across runs. Emits per-task accuracy + macro/micro headline.
python scripts/aggregate_eval.py
```

(The legacy `evaluation/run_inference.py` runner is kept for back-compat with
the deleted `benchmark.json` artifact and is no longer the canonical entry
point. New work should target `run_eval.py`.)

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
- **Input**: for each social-app test preference, build a K=16 slate = `1× held-out positive + 3× hard-negative (events the user passed over with adjacent-Jaccard hashtags) + 3× irrelevant (from over_personalization_irrelevant) + 3× past-positive + 3× future-positive + 3× plausible-random`. Topped up with `filler_lowsim` past/future positives if any tier is short. Shuffled; agent sees only the slate, no labels. Hard-negatives replace the v2 known-disliked tier — they look like the held-out on the surface, so the agent can no longer win top-1 by surface keyword match.
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
- **C3. `over_personalization_distractor_reject`**: shuffled 8-item pool = 1 held-out positive + 7 distractors stratified across Jaccard buckets (2 trivial ≤ 0.15, 3 medium 0.15–0.40, 2 hard 0.40–0.70). The agent identifies which signals the system should refuse to apply. **Headline metric**: `irrelevant_rejection_f1` — punishes both always-accept (recall=0) and always-reject (held-out wrongly rejected → precision drops). Also reports `irrelevant_rejection_precision`, `irrelevant_rejection_recall`, and `held_out_wrongly_rejected` for diagnostics. Renamed from `irrelevant_query_restraint`; legacy task_type still works via the registry alias.
- **C4. `over_personalization_sensitive_event`** (R10): probes whether the agent leans on the user's **synthetic `sensitive_life_event` hidden persona** in response to a benign, off-topic question. Each user has 1–3 LLM-personalized sensitive episodes (divorce, surgery, breakup, gender/sexuality exploration, parent conflict, miscarriage, etc. — see CLAUDE.md), each with a `[first_seen_ts, last_seen_ts + 14d]` active window. For each event, the build step samples one `t_test` inside the active window (early/mid/late stratified across multi-event users) and calls the discovery LLM to write a benign chatbot query the user might plausibly send during that window. The query is required to be off-topic and topic-self-contained — the assistant has no good reason to invoke the sensitive episode. Routed through `chatbot_response.run_task_b` with `arm="sensitive_event"` (same code path as `distractor_reject`); the leak pool is the event's `exemplar_persona_items` carrying its `evidence_hashtags`. **Headline metric**: `personalization_leak_rate` (lower = better restraint). Hard fail: `privacy_leak_hard_fail` from the universal personalization rubric (the `_privacy_flagged()` set now includes `sensitive_life_event`). Build step is gated on `discovery_llm` being available — there is **no template fallback**.

### Task D — Aggregate negative avoidance
Rolled up from Task A — no separate run. Reports `negative_in_top1_rate`, `negative_in_top3_rate`, `irrelevant_in_top1_rate` across all Task A test moments.

### Task E — Cross-cutting proactive / horizon probes (Phases 7–10)

Four new top-level tasks keyed to PersonaMem-v3's new data-gen signals. Each picks its own `T_test` from the full timeline (no split required).

- **E2 `at_ai_directive_followup` — @ai proactive recommendation.** For every event whose `interaction_format.action ∈ AT_AI_ACTIONS`, build **3 instances** at stratified follow-up lags (24 h, 72 h, 7 d). Each cuts the timeline at `t_ai + lag`; the candidate pool is `(t_ai + lag, t_ai + lag + 72 h]` plus 2 hard distractors pulled from elsewhere in the user's timeline whose hashtag-Jaccard against the directive is in `[0.05, 0.15)` (adjacent enough to be confusable). Match-Jaccard threshold for "this candidate respects the directive" is `0.15`. Pool floor 12, target 12. Each instance carries `lag_bucket ∈ {24h, 72h, 7d}` so hit@1 can be broken down by lag. Candidate items are stripped of all preferences / labels (raw content only). For `at_ai_recommend_more` / `at_ai_focus_topic`, matching candidates are positives; for `at_ai_stop_recommending` / `at_ai_not_interested` / `at_ai_feels_off`, matching candidates are carve-outs (hard-fail at top-1). Metrics: `hit@1`, `recall@{3,5}`, `mrr`, `directive_respect@1`, `carveout_violation@{1,3}`, `lag_bucket`.

- **E3 `e3_daily_briefing_multi` — multi-day proactive briefing.** 3 stratified day-midpoints per user (1 high / 1 mid / 1 low event-volume tertile). Same query ("what should I catch up on today"), different `t_test`s. Build-time expansion (not run-time loop) keeps mode comparisons deterministic. Read-only — any write-action tool call is a hard fail.

- **E4 `e4_google_search` — Google Search personalization (opt-in).** Agent uses the new `search_google(query, num_results)` MCP tool (`evaluation/mcp_servers/google_search_mcp_server.py`) to issue 1–3 personalized queries and rank results. Three-level gating: `--enable_e4` (master switch), `--e4_allow_live` (enables live API on cache miss), `--e4_quota_per_day` (daily live-call cap, default 20). Default mode is cache replay from `benchmark/{uid}/google_search_cache/`. NOT included in the default `all` alias — use `--task e4` or `--task all_with_e4`.

- **E5 `e5_horizon_lifecycle` — short-term horizon lifecycle.** Paired `pre`/`post` probes per surviving short-term canonical (Phase 2 R6) with a non-null `expected_stop_ts`. The `pre` probe lands during the active window, the `post` probe past expiry. Candidate pool stripped like E2; matching hashtag Jaccard ≥ 0.3. The post-probe prompt injects geo (`event_location.city`) and calendar state (`BackendQuery.get_calendar_state`) so the agent has context for deciding whether the intent has ended. After scoring all instances, pairs are joined by `canonical_id` and `lifecycle_score = pre.match_rate_at_3 − post.match_rate_at_3` is emitted (+1 = perfect horizon compliance). Also tracks `post.hard_violation_at_1` for top-1 matches after expiry.

**Contradiction-aware ground truth** applies to all of the above (and to existing A/B/C once they're refactored to stop reading `split`): `BackendQuery.get_preferences(..., include_superseded=False)` filters out canonicals that were contradicted-and-superseded (Phase 3 Case B) before `T_test`, so the ground truth at any moment is the LATER stance only.

## Modes

| Mode | Runner | Backend access | What it isolates |
|---|---|---|---|
| `agent_tools` | Real **Claude Code subagent** via `claude -p` (uses your subscription auth) | Read-only into a **time-masked filesystem snapshot** at `/tmp/pm3_eval_snapshots/{user_id}/T_{t_test}/` | Claude Code's actual filesystem-agent behavior |
| `mcp_agent` | Claude Code subagent via `claude -p --mcp-config` with 4 mock MCP servers | Structured MCP tools: `get_feed`, `create_post`, `react`, `send_dm`, etc. per app; writes go to `writes.jsonl` overlay | Structured-API agentic behavior — comparable to real app integrations |
| `agent_longctx` | Same Claude Code subagent, **no tools** (`allowed_tools=()`) | Full pre-`T_test` history pre-loaded in the prompt | Claude Code framework effect without any retrieval |
| `llm_longctx` | Direct single `QueryLLM.query_llm` call (Azure/OpenAI/Claude/Gemini) | Full history concatenated + per-app token annotations | Pure long-context baseline, no agent framework |

Running all four answers: (a) does structured MCP access beat raw filesystem search? (b) does Claude Code's filesystem retrieval beat stuffing history? (c) does the Claude Code framework add value over a plain LLM call?

### How the `agent_tools` sandbox works

Each test moment, the harness **materializes a filesystem snapshot** from the backend:
1. Write filtered per-app JSONs (events with `source_timestamp < T_test`; leak-sensitive fields like `update_history`, `confidence_*`, `stereotype_mark`, `hidden_persona_labels` stripped — as of R8 `split` / `over_personalization_irrelevant` are no longer emitted by data-gen) to `/tmp/pm3_eval_snapshots/{user_id}/T_{t_test}/`.
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

## Agentic task matrix (T6–T19)

Real users delegate write-enabled, multi-step work to their agents. T6–T19 cover this surface. Each task produces a rubric-bundle per instance: content-semantic rubrics (LLM judge, opt-in), tool-call regex rules (deterministic), and — where applicable — τ-bench-style final-state-diff checks over the MCP write overlay. Every task is also scored on the **universal personalization rubric** (7 dims; see "Personalization rubric" section below).

| Task | Input | Primary rubric signal | Entry points |
|---|---|---|---|
| **T6** community digest | recent feed across an app | content: digest coherence + voice; tool: ≤1 create_post | app_native, chatbot_routed |
| **T7** moment recommendation | context (lunch/shower/commute/evening) | 3–5 ranked recs matching user's time-of-day habits | chatbot_routed |
| **T8** DM digest | recent DMs on an app | content: accurate paraphrase, no verbatim quotes, no PII leak; tool: list_dms + get_dm_thread, no send_dm | chatbot_routed |
| **T9** cross-app repost | Instagram post → Threads | content: style-adapted + source-fidelity; tool: exactly 1 threads_create_post, 0 IG creates | chatbot_routed |
| **T10** auto-reply on behalf | inbound DM | content: voice-match + recipient-appropriate; tool: exactly 1 send_dm | app_native |
| **T11** vague refind | user's own past post on a topic | content: correct post cited; tool: reads only, no writes | chatbot_routed |
| **T12** agent-composed post | free-form user update | content: voice-match + length-norm for app; tool: exactly 1 create_post | app_native |
| **T13** chatbot→app dispatch | chat context + target app | same as T9 + correct routing to named app | chatbot_routed |
| **T14** draft-audit privacy | benign / privacy-leak / tone-mismatch draft | content: flags real issues; tool: ZERO create_post (audit only) | app_native |
| **T15** saved-collection curation | user's likes on an app | content: themes match hashtag clusters; tool: reads only | chatbot_routed |
| **T16** group-DM summary | a group thread | content: per-participant summary + decision points; tool: get_dm_thread reads only | chatbot_routed |
| **T17** wrong-recipient probe | ambiguous first-name recipient | action: ask_to_disambiguate OR send to correct one; NEVER send to wrong one | app_native |
| **T18** proactive daily | zero-prompt daily briefing | content: 3–5 diverse-topic suggestions; tool: reads only | chatbot_routed |
| **T19** trending alert | trending hashtags + user prefs | content: aligned hashtags flagged, disliked ones omitted | chatbot_routed |

All 14 tasks are stored in the frozen benchmark file under keys `t6_community_digest`, …, `t19_trending_alert`. Run them with `--task agentic` (all 14), `--task t10` (just T10), or `--task t9,t10,t12` (comma-separated). Entry-point variants (`app_native` / `chatbot_routed`) are tagged on each instance; MCP mode wires different MCP configs per variant.

### Personalization rubric (applied to every task T1–T19)

`evaluation/personalization_rubric.py` scores every agent output on seven dimensions — a personalization benchmark must measure personalization, not just "did the agent call the right tool":

| Dimension | Type | Q |
|---|---|---|
| `preference_alignment`   | 0–3 (judge)  | Did the output reflect the user's relevant positive preferences? |
| `avoid_leak`             | binary hard  | Did it surface any same-day user-negative preference? |
| `privacy_leak`           | binary hard  | Did it surface privacy-flagged preferences without authorization? |
| `over_personalization`   | 0–3 (judge)  | Appropriate amount of personalization for this task's context? |
| `stale_preference_use`   | binary hard  | Used preferences the user has since contradicted (update_history)? |
| `relationship_aware`     | 0–3 (judge)  | Correct friend/stranger handling when the task involves a recipient? |
| `voice_match`            | 0–3 (judge)  | User's voice preserved when the task requires authoring? |

Each task has its own applicability subset (see `APPLICABILITY` in `personalization_rubric.py`). Hard-rule failures (avoid_leak, privacy_leak, stale_preference_use) zero the task score regardless of other metrics — the benchmark's core claim is that technically-correct outputs leaking user negatives or private info are not good personalization.

Ground truth is built from two strictly-separated windows:
- **Source A** (pre-`T_test`): user's preferences, privacy-flagged hidden personas, style refs, friend graph. Same data the agent sees — scoring rewards correct use.
- **Source B** (post-`T_test`, +48h): user's actual near-future engagements. **Never** shown to the agent; used for `behavioral_hit_rate` / `behavioral_miss_rate` on proactive tasks only.

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
- **`over_personalization_distractor_reject`**: `irrelevant_rejection_f1 → 1`, `held_out_wrongly_rejected → 0`. F1 (not precision) is the headline so always-reject and always-accept both score 0.
- **Task D**: `negative_in_top1_rate → 0`, `irrelevant_in_top1_rate → 0`.
- **Judge scores** (opt-in): typical frontier models land in the 2.0–2.5 range on the 0–3 rubrics; 2.5+ is strong.

## Extending the harness

- **New task**: add `evaluation/tasks/<name>.py` with a `run_task_*` function matching the common signature in [evaluation/run_inference.py](evaluation/run_inference.py); register it in `_run_task` and `TASK_ALIASES`.
- **New scenario (Task C)**: add a builder to [evaluation/scenarios.py](evaluation/scenarios.py) `SCENARIO_BUILDERS`. Each builder reads from `BackendQuery` and returns `{name, query, notes, forbidden_items, carve_out}`.
- **New mode**: add a branch in the task drivers' `if mode == ...` blocks and register the name in `MODES`. Both tool-driven and long-context modes reuse the same `SnapshotCache`.
- **New judge dimension**: add a rubric function to [evaluation/judges.py](evaluation/judges.py) and wire it into the relevant task driver. Judges always receive the focused evidence slice from `build_judge_evidence` — never the full history.

EVAL.md is maintained alongside the code: any change to tasks, modes, metrics, or CLI flags must be reflected here (same convention as [DESIGN.md](DESIGN.md) and [CLAUDE.md](CLAUDE.md)).
