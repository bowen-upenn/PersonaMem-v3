# Eval data quality audit guide

Procedural guide for auditing `backend/{user_id}/test.json` after a regen. The audit catches problems the unit-level self-checks and verifiers cannot — silent task-type loss, schema drift, unfair test pairs, jargon / template leaks, GT/query mismatches, voice drift across rows, and distribution gaps.

`test.json` is a JSON list of test-instance dicts. Each item carries `query_id`, `task_type`, `ts`, `user_query`, `example_response`, `inferior_response`, `groundtruth_preference`, `rubric_tags`, and an `instance_full` block with the runner-side payload. (The legacy `benchmark/{uid}/queries.csv` is no longer produced.)

This document is methodology only. It contains no findings from any specific audit. Run the procedure below after every regen and write the findings into a separate report.

## When to run

Run a **full audit** after any of:

1. A regen of `queries.csv` via `scripts/prepare_eval_data.py`.
2. A change to a task builder, an LLM prompt that produces `example_response` / `inferior_response`, or the task registry meta (`evaluation/task_registry.py`).
3. A change to `_project_row` in `scripts/prepare_eval_data.py` (schema-level).
4. A change to any verifier or generator under `evaluation/tasks/`, `evaluation/llm_postprocess.py`, or `evaluation/audit_query_quality.py`.
5. Before shipping a new benchmark version externally.

A **quick partial audit** (only Slice C, see §3) is enough for unrelated code changes that touched a single task type.

## Setup

1. Confirm all per-user runs completed cleanly. Inspect `/tmp/eval_regen/{uid}.stderr` for `Exit status: 0` and `/tmp/eval_regen/{uid}.stdout` for the `=== prepare_eval_data summary ===` block at the bottom. Read every funnel line (`[task_distribution] capped …`, `dropped … queries with empty GT`, `dropped … queries before the 20% engagement-history mark`, `sensitive-event coverage: dropped …`, `format-verify dropped …`, `personalization-routing verify: dropped …`) and write them into a table. **Large unexpected drop counts are the first signal of a regression.**
2. Confirm each user's `queries.csv` mtime is newer than its `backend/{uid}/profile.json`. If not, the regen didn't actually rewrite the file.
3. Note the user-ID set you're auditing. The standard set is `{105, 115, 229, 282, 760}`; substitute as needed.

## Audit shape — three parallel Explore agents

The audit splits into three orthogonal slices so that (a) parallelism halves wall time, (b) each agent has a focused remit, and (c) findings from one slice don't bias the others. **Spawn all three in a single message** with multiple Agent tool calls.

### Slice A — Personalization / restraint fairness

Focuses on tasks where the right answer depends on what the model is allowed (or forbidden) to say about the user. The trap: a "restraint" test that actually punishes a correct personalized answer, or a "personalization" test where personalization wouldn't help.

Task types to sample:

- `chatbot_personalized_response` — must weave in held-out pref
- `over_personalization_chatbot_text` (drift / control / stale / context_shift arms) — must NOT lean on pref
- `over_personalization_sensitive_event` — must NOT lean on planted episode
- `over_personalization_repetition_chatbot` / `over_personalization_repetition_recsys` — diversity restraint
- `restraint_sensitive_event_silence` — proactive silence
- `preference_shift_followthrough` — track NEW pref, not OLD

Quality dimensions per row:

1. Query reads like a real user message (or `[system prompt] …` fallback is sensible).
2. GT actually answers the query / is non-empty.
3. Personalization arms: example genuinely benefits from knowing user. Restraint arms: inferior actually leans on a real pref in its BODY (not just preamble).
4. Same-domain trap check on `over_personalization_context_shift`: does the LLM-declared `chosen_pref_domain` actually differ from `query_domain`? Are domain labels too generic ("lifestyle", "general", "content")?
5. Recency check on over_pers foils: `flaw_evidence.recency_delta_seconds` should be ≤ 30 days from t_test (the hard ceiling in `evaluation/llm_postprocess.py:_OVER_PERS_HARD_MAX_DAYS`).
6. Preamble-stripped inferior body should differ materially from example body (≥30% token diff) for restraint arms.
7. **Foil pick must be in the Forbidden list**: for `over_personalization_chatbot_text` and `over_personalization_context_shift`, the persona_item the Inferior leans on (`inferior_response.flaw_evidence.persona_item`) MUST appear in the row's `groundtruth_preference` "Forbidden items" / "must NOT be surfaced" list. If the foil leans on a top-recency category that's outside the pre-baked list (top-K relevant prefs / scenario-curated forbidden_items), the judge has no anchor to penalize that specific leak. `data_preparation/visualize.py::_gt_chatbot_restraint` and `_gt_context_shift_scenarios` hoist the surfaced item into the list when missing.

### Slice B — Recommendation / agentic content quality

Focuses on tasks that produce content (compose tasks) or ranked slates. The trap: short / off-voice / off-format outputs, or ranking errors.

Task types to sample:

- `personalized_recommendation`, `hidden_persona_recommendation` (ranking, 16-item slate)
- `hidden_persona_implicit_qa` — chatbot QA grounded on hidden persona
- `local_recommendation_geo_shift` — adapts to inferred city
- `at_ai_directive_followup` — **`expected_response_kind = "ranking"`** (do NOT flag the ranked-indexes format as a bug)
- `active_mistake_prevention` — proactive warning
- `agentic_*` family: `community_post`, `send_post`, `cross_app_repost`, `auto_reply`, `dm_digest`, `group_dm_summary`, `vague_refind`, `trending_alert`, `proactive_daily_catchup`, `wrong_recipient_check`

Quality dimensions per row:

1. Ranking tasks: `example_response` = exactly `"Ranked indexes: [0..15]"` with 16 distinct indices; held_out_idx is in position 0; hard_negative_idxs are at the bottom.
2. Compose tasks: word count ≥ `MIN_COMPOSE_WORDS` (currently 100). Distribution sanity (min, p25, median, p75, max).
3. Voice match: response feels like the user's idiolect (emoji density, lowercase / capitalization, register, signature phrases). Check `backend/{uid}/profile.json::user_voice` for the canonical voice.
4. Tool call shape matches `tool_call_rules`. Each agentic instance should have a non-empty `tool_call`.
5. Geo-shift: target city differs from user's home city. (Same city across rows IS by design — diversity is on the category axis, not the city axis. The builder emits per-(transition × category).)
6. Hidden persona QA: rubric / GT references an actual `hidden_personas[*]` entry from the user's profile.json.
7. `agentic_cross_app_repost`: first sentence references the source app or carries a crossposting marker ("crossposting", "saw this on X", "originally a X post", etc.).

### Slice C — Cross-cutting mechanical scan

Programmatic checks across ALL rows of ALL users. Cheap, exhaustive, catches schema drift and silent omissions.

Mandatory checks (write a single Python script using `csv.DictReader` + `csv.field_size_limit(sys.maxsize)`):

1. **Schema integrity**: every row has all 17 columns; `instance_json` parses 100%; `query_id` globally unique; `ts` monotonic per user; `ts_iso ↔ ts` consistent; `task_family ↔ task_type` 1:1.
2. **Field presence per task type**: for each `task_type`, fraction of rows with non-empty `example_response`, `inferior_response.text`, `groundtruth_preference`. Flag types where >10% are missing one.
3. **Shape consistency per task type**: within a `task_type`, `inferior_response` is always dict-with-text OR always string (not mixed). Same for `groundtruth_preference` (string vs list vs dict).
4. **Coverage gaps**: count rows per `task_type` per user. Highlight types with 0 or 1 row for any user — that's a coverage gap likely from over-aggressive filtering.
5. **Cross-user task absence**: a task type present in 4/5 users but missing in the 5th — investigate the missing user's funnel.
6. **Substring blocklists** (per `instance_json`, count > 0 = flag):
   - `(none identified)` — should always be 0.
   - `slate` — flag for review (may be legitimate product / show name).
   - `head-zone`, `tail-zone`, `token Jaccard`, `n_allowed_repetitions`, `persona-aligned hashtags`, `the agent should`, `the agent must` — internal jargon leaks.
   - `\nQ1: `, `\nQ2: `, `\nTurn 1: `, `\nResponse 1: ` — outline-shape responses.
   - `share the thread`, `paste it here`, `I can't see your`, `I don't have access` — refusals.
   - `{privacy_rubric_line}`, `{surfaced_suffix}`, `{T}`, `{warmup_window}`, `{monitored_start}`, `{head_window}`, `{tail_start}`, `{target_pref}`, `{gold_idx}` — un-substituted templates.
7. **Empty fields**: report rows where any required column is empty: `query_id`, `task_family`, `task_type`, `instance_id`, `ts`, `expected_response_kind`, `rubric_tags`, `display_rubric`. (Empty `query_text` is OK for tasks with `[system prompt]` fallback.)
8. **Empty user_query on USER_MESSAGE_TASKS**: cross-check that no row of `chatbot_personalized_response`, `over_personalization_chatbot_text`, `over_personalization_context_shift`, `over_personalization_sensitive_event`, `active_mistake_prevention`, `local_recommendation_geo_shift` has empty `instance_json.user_query` (or `query` / `user_message`).
9. **Compose-task length distribution**: for each compose task type, compute `min/p25/median/p75/max` word counts and `count_under_floor`.
10. **Phrase variety on sensitive_event queries**: count rows where `user_query` starts with stock fillers (e.g. "low-key way to", "without making it"). Flag if >10% of any user's sensitive_event rows share an opener.

## Existing automated checks — what the pipeline already enforces

A human audit is COMPLEMENTARY, not redundant, to the automated checks the pipeline already runs. Use this list to decide what to skip (already covered) and what to focus on (not covered). If a finding implies an automated check SHOULD have caught it, **that is itself a finding** — the check has regressed or is misconfigured.

### Build-time gates in `scripts/prepare_eval_data.py`

These gates execute during build, BUT they are also **post-generation audit checkpoints**. For each gate, the auditor must verify two things after a regen:

1. **The gate fired correctly**: nothing that SHOULD have been dropped slipped through into `queries.csv`. Re-run the gate's logic over the shipped rows and confirm zero violations remain (e.g. `grep -c '(none identified)'` returns 0, no `USER_MESSAGE_TASKS` row has empty `user_query`, etc.).
2. **The gate didn't over-drop**: the drop count makes sense in context. A 0% drop rate is suspicious (gate may be silently no-op'ing); an unexpectedly high drop rate (e.g. >50% of one task type) means the gate is misconfigured or its precision regressed. Compare drop counts against the prior regen as a baseline — a sudden jump is a finding.

The drop counts are emitted to `/tmp/eval_regen/{uid}.stdout`. **Read those lines as part of every audit.**

| # | Check | Location | Drops what | Post-gen audit method |
|---|---|---|---|---|
| 1 | Empty-GT filter | `:575` | rows where `groundtruth_preference ∈ {"", "(none identified)"}` | `grep -c '(none identified)' backend/*/test.json` → 0 |
| 2 | History-floor filter | `:596–604` | rows whose `t_test` falls before the 20% mark of user's engagement history | python: for every row, confirm `ts ≥ engagement_ts[len(events)//5]` |
| 3 | Sensitive-event coverage check | `:606–672` | `over_personalization_sensitive_event` rows whose planted episode topic / hashtags / label_fragment never appear in chatbot / ai_studio events before `t_test` | sample 3 surviving sensitive-event rows; confirm the episode topic appears in some pre-T_test chatbot / ai_studio event |
| 4 | Format-verify gate (a) | `:682–697` | rows where `inferior_response` is empty (string `""` OR dict with empty `.text`) | python: for every row, `inferior_response` is non-empty string OR dict with non-empty `.text` |
| 5 | Format-verify gate (b) | `:698–706` | `USER_MESSAGE_TASKS` rows with no `query` / `user_message` / `user_query` field populated | python: for every USER_MESSAGE_TASKS row, at least one of those three keys is non-empty |
| 6 | Personalization-routing verify | `:293–367` | `chatbot_personalized_response` NEUTRAL + `over_personalization_chatbot_text` HELPS via mini-LLM judge | check post-routing counts ≥ floor (e.g. ≥10 per user for chatbot_personalized) — a collapse is a signal the verifier is too strict |
| 7 | Tool-call gate | `:422–465` | agentic / E3 / E6 instances with invalid `tool_call` payloads. Drop reasons now print to stdout inline (no on-disk log file) | every agentic row in test.json has non-empty `tool_call` matching its `tool_call_rules`; the stdout drop reasons should align with what's missing |
| 8 | Per-instance self-check | `llm_postprocess.py:1607` | `_run_self_check(task_type, query, response)` — task-specific LLM-judge that catches off-task example responses; failed responses get regenerated once before being dropped | log lines `self_check_failed=N` — a high N relative to total self_checks is a prompt regression |
| 9 | Voice-evidence distinguishability | `llm_postprocess.py:600` | agentic compose rows where example_response and inferior_response voice-evidence sets are too similar to support a fair voice_match grade | log lines `voice_check_failed=N` + `voice_check_regen=M`; sample 3 surviving rows, confirm example/inferior carry visibly different voice anchors |
| 10 | Triplet self-check | `llm_postprocess.py:778` | chatbot personalized response triplets (proactive / control / adversarial) where the triplet doesn't satisfy the held-out alignment criteria | log lines `chatbot_triplet_built=N chatbot_triplet_failed=M`; failure count > 0 is a signal |
| 11 | Compose-length validator | `llm_postprocess.py:_validate_compose_length` | example_response is below 100 words on any of the 4 compose tasks; triggers a regen pass during generation | python: median word count per compose task ≥ 100 per user; flag if `under_100` > 20% of compose rows |
| 12 | Sensitive-event preamble guard | `llm_postprocess.py:_preamble_stripped_too_similar` | sensitive-event inferior whose body (with leading "as a [ROLE], …" preamble stripped) shares ≥0.7 token Jaccard with example — regenerates the inferior | sample 5 sensitive-event rows; strip the leading "as a [ROLE], " preamble from each inferior; confirm Jaccard against example < 0.7 |

### Structural and contamination checks in `evaluation/audit_helpers.py`

| # | Check | Location | Detects |
|---|---|---|---|
| 13 | Slate canonical-keys uniformity | `:55–64` | extra keys present on some slate items but not others — leaks format-by-presence |
| 14 | Length normalization | `:67–73` | max/min caption length ratio > 4× — held-out item visibly different from distractors |
| 15 | Content-type uniqueness check | `:78–86` | held_out's content_type is unique among slate items (e.g. only `reel`, all others `text`) — text-blind format leak |
| 16 | Question-token overlap | `:89–112` | any candidate exclusively shares > 3 non-stopword tokens with the user query — text-only discrimination signal |
| 17 | Blind-baseline LLM probe | `:118–177` | for ranking instances, a context-free LLM probe that sees only the slate. If it picks the held-out target, the slate is contaminated. Drives audit-and-regenerate loop in `audit_instance(:180)` |

### LLM-graded axis audit — `evaluation/audit_query_quality.py:1742–1791`

Twelve dimensions, each a callable; results aggregated per query. Self-skips when the task type is out of scope for a dimension:

| Dimension | Catches | Applies to |
|---|---|---|
| `completeness` | Empty `example_response` / `inferior_response` / `groundtruth_preference` | all |
| `schema_sanity` | Structural validity of `instance_json` (deep check) | all |
| `sensitive_probe_placement` | Sensitive-event probe lands at a sensible temporal position relative to the planted episode | `over_personalization_sensitive_event` |
| `response_quality` (×3 sub-results) | `telegraph_avoidance`, `no_refusal`, `no_rubric_leak` (extended to flag outline-shaped responses) | all |
| `naturalness` | Query reads like a real user message | tasks with `user_query` |
| `context_required` | Query genuinely benefits from personalization | personalization tasks |
| `context_restraint` | Query genuinely should NOT benefit from personalization | restraint tasks |
| `inferior_targets_task_axis` | Inferior response targets the task's intended failure mode | per-task foil validation |
| `gt_alignment` | Example response weaves in / matches the `groundtruth_preference` | `GT_ALIGNMENT_APPLICABLE` set |
| `privacy_leak` | No surface mention of privacy-flagged hidden personas | tasks anchored on hidden personas |
| `tool_call_validity` | Agentic + E3/E6 tool-call payloads dry-run successfully via MCP at `t_test` | `TOOL_CALL_VALIDITY_TASKS` set |
| `frame_consistency` | User-voiced response carries the dominant motivational frame | `FRAME_CONSISTENCY_TASKS` set |

This audit is invoked as the post-write step from `prepare_eval_data.py`.

### Agentic per-task verifiers — `evaluation/tasks/agentic_verifiers.py`

Each agentic task has a verifier that emits a checklist of `(check_name, pass / fail)` pairs scored downstream:

| Task | Verifier | Key checks |
|---|---|---|
| compose family (community_post / send_post / cross_app_repost / auto_reply) | `_verify_*_post` | length ≥ `MIN_COMPOSE_WORDS` (100), voice match (downstream judge), tool_call shape, target_app match |
| `cross_app_repost` (extra) | `_verify_cross_app_repost` | first ~240 chars must name the source app OR carry a crosspost marker ("crossposting", "saw this on", "originally on", etc.) |
| `agentic_dm_digest` | `_verify_dm_digest` | digest covers required threads, doesn't fabricate |
| `agentic_group_dm_summary` | `_verify_group_dm_summary` | summary respects group privacy |
| `agentic_vague_refind` | `_verify_vague_refind` | refind targets the right item |
| `agentic_wrong_recipient_check` | `_verify_wrong_recipient_check` | warns on cross-thread leak |
| `agentic_proactive_daily_catchup` | `_verify_proactive_daily_catchup` | catchup hits required threads |
| `agentic_trending_alert` | `_verify_trending_alert` | alert lands on a real trending topic |
| `t14_draft_audit` | `t14_draft_audit` in `prompts_agentic.py:411` | mini-tier LLM audits the draft for voice + safety before commit |

### Pre-build eligibility filters

| Filter | Location | Filters what |
|---|---|---|
| Proactive candidate filter | `build_benchmark.py:573` | `_proactive_filter_ok` — drops candidates failing basic proactive shape (timing, fresh-start, action body length) |
| Hidden persona QA eligibility | `tasks/hidden_persona_implicit_qa.py:525` | personas with insufficient evidence rows / temporal span |
| Hidden persona rec eligibility | `tasks/hidden_persona_recommendation.py:394` | distinct candidate pool requirements |
| Preference-shift candidate harvester | `tasks/preference_shift_followthrough.py:_harvest_shift_candidates` | accepts `contradicted` (with `stance_shift_with_precedent` / `suppressed_insufficient_precedent`) and `shifted` update_types; also short_term_expirations whose t_test fits the window OR falls back to t_now - 1h for future stops |

### Drop-log persistence

Three durable logs per regen tell you what was lost:

- `/tmp/eval_regen/{uid}.stdout` — the canonical per-run log. Carries:
  - tool-call gate drops (printed inline, was `build_benchmark.dropped.jsonl`)
  - per-user skip reasons (printed to stderr, was `_prepare_eval_data.skipped.txt`)
  - `dropped … queries with empty GT`
  - `dropped … queries before the 20% engagement-history mark`
  - `sensitive-event coverage: dropped …`
  - `format-verify dropped …`
  - `personalization-routing verify: dropped …`

**Audit step zero**: read all three before sampling individual rows. A regression in any automated check should manifest as a delta in these counts vs. the prior regen.

### What the automated checks DON'T catch (where the human audit earns its keep)

1. **Silent schema drift**: a task type that builds successfully but never lands in queries.csv (caps logged, format-verify dropped them all without a louder alarm).
2. **Same-domain trap in restraint queries**: the LLM declares two domains are different, the lexical post-check passes, but a human reading the query+pref pair sees they're conceptually the same domain.
3. **Preamble-only inferiors**: the response_quality LLM grader sees a leaned-on preamble and marks the inferior as "over-personalized" — but the body is identical to the example.
4. **GT shape mismatch within a task type**: a task type ships some rows with string GT and others with list GT — both pass `completeness` but downstream graders break.
5. **Distribution gaps**: a task type with only 1 surviving row for one user. Each row passes its checks; the audit failure is the COUNT.
6. **Cross-row diversity collapse**: sensitive-event queries all start with the same opener; compose-tasks all hit exactly 100 words to clear the floor. Each row in isolation looks fine.
7. **Tool-call schema drift**: the gate accepts `{"tool":..., "args":...}` shape, but the runner expects `{"name":..., "input":...}` — both pass build time; eval breaks silently.
8. **Voice drift across rows**: each row passes voice_match in isolation; reading 5 rows in a row, the user's idiolect varies more than it should.

## Severity rubric

- **P0** — Blocks shipping. Schema break, silent task-type loss, un-substituted template visible to reviewer, internal jargon leak in human-visible text.
- **P1** — Unfair or misleading test. Restraint test with same-domain trap, GT absent on an active arm, recency violation on over_pers foil, compose-task length floor breached at 100%.
- **P2** — Quality polish. Coverage gap that's not critical, residue strings that may be legitimate product names, etc.

## Reporting format

Each finding includes:

- Severity (P0 / P1 / P2)
- Task type + user_id + query_id (or row offset)
- One-sentence problem statement
- Concrete excerpt (query / GT / example / inferior text — whichever shows the issue)
- (For pattern findings) row count per user + total

Group findings by severity. Aim for ≤ 2000 words per slice. **Do not propose fixes inside the audit report** — fixes belong in a separate plan step gated on the user's review.

## Known false positives — do not flag

- `at_ai_directive_followup` `example_response = "Ranked indexes: [...]"` — INTENTIONAL. `expected_response_kind: "ranking"` in `evaluation/task_registry.py:773`.
- `[system prompt] …` query_text for ranking / proactive / agentic tasks — INTENTIONAL fallback for tasks with no live user message.
- `slate` substring in candidate item titles (product / show names) — likely legitimate; flag only `[system prompt]` rows.
- Sensitive-event queries phrased as confessions or imperatives — INTENTIONAL phrase-variety in the generation prompt.
- `local_recommendation_geo_shift` repeating the same city across rows — INTENTIONAL: the builder emits per-(transition × category) instances; diversity is on the category axis, not the city axis.
- `preference_shift_followthrough` with `new_preference: null` — INTENTIONAL for `short_term_expiration` shift_kind (the preference simply expired with no replacement).
- `groundtruth_preference` as a dict (not a string) — INTENTIONAL for tasks with structured GT (`preference_shift_followthrough`, ranking tasks, etc.). The visualize.py renderers flatten to string for display.

## Verification commands

After a regen, run these spot-checks before declaring the audit closed:

```bash
# All 5 users have non-zero context_shift rows
for u in 105 115 229 282 760; do
  echo -n "$u: "
  grep -c context_shift backend/$u/test.json
done

# No un-substituted placeholders or known leaks
grep -E '\{privacy_rubric_line\}|\{surfaced_suffix\}|\{warmup_window\}|\{monitored_start\}|\{head_window\}|\{tail_start\}|\{target_pref\}|\{gold_idx\}|n_allowed_repetitions|token Jaccard|\(none identified\)' \
  backend/*/test.json | wc -l   # should be 0

# Compose-task word floor
python3 -c "
import json
COMPOSE = {'agentic_send_post','agentic_community_post','agentic_cross_app_repost','agentic_auto_reply'}
for u in ['105','115','229','282','760']:
    counts = []
    with open(f'backend/{u}/test.json') as f:
        for r in json.load(f):
            if r.get('task_type') not in COMPOSE: continue
            ex = r.get('example_response') or (r.get('instance_full') or {}).get('example_response')
            if isinstance(ex, dict): ex = ex.get('text','')
            counts.append(len((ex or '').split()))
    if counts:
        counts.sort()
        print(f'{u}: n={len(counts)} median={counts[len(counts)//2]} under_100={sum(1 for w in counts if w<100)}')
"

# Empty display_rubric on active_mistake_prevention
# (display_rubric is no longer a top-level column; check the instance's
# rubric_tags array carries non-empty entries instead)
python3 -c "
import json
for u in ['105','115','229','282','760']:
    empty = 0
    with open(f'backend/{u}/test.json') as f:
        for r in json.load(f):
            if r.get('task_type') != 'active_mistake_prevention': continue
            if not (r.get('rubric_tags') or []): empty += 1
    print(f'{u}: amp_empty_rubric={empty}')
"
```

Expected results after a clean regen:

- Every user has ≥ 5 `context_shift` rows.
- Substring blocklist grep returns 0.
- Median compose-task word count ≥ 100 per user.
- All `active_mistake_prevention` rows carry non-empty `display_rubric`.

## Anti-patterns

### Delegating understanding

When you spawn the three Explore agents, do NOT delegate "find quality problems and fix them" to a single agent — that conflates judgment with action. The audit and the fix planning are separate steps; each requires the user's eyes on the findings before the fix runs.

### Counting instead of reading

A mechanical check ("0 rows of `(none identified)`") is satisfying but doesn't catch semantic issues. For each task type, the auditor MUST read at least 3 actual sample rows end-to-end, comparing query against example_response against inferior_response against rubric. Counting alone misses things like the preamble-only inferior pattern.

### Trusting the LLM's own self-judgment

Some prompts ask the LLM to declare structured fields ("the two domains are different"). The LLM's self-judgment is one signal, not the only signal. Always add a lexical post-check (e.g. `_domains_overlap`) so an LLM mis-judgment doesn't ship.

### Auditing only the happy path

A response_quality grader that only checks for telegraph phrases will silently pass an outline-shaped response. When extending an audit dimension, also write the *failure* example you expect the grader to catch — if the grader passes that example, the dimension is broken.
