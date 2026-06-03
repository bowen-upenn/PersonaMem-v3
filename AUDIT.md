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
3. Discover the persona set **dynamically** — every dir under `backend/` that is not prefixed with `_` (e.g. `ls backend | grep -vE '^_'`, or `glob.glob('backend/*/test.json')`). Do NOT hardcode a fixed user-ID set: this methodology must run unchanged over whatever personas are present (5, 20, or 200).

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
8. **Strawman foil — persona reference positionally stereotyped (final-sentence tell).** Beyond the analogy/preamble tells (#103), check WHERE the inferior's forbidden-pref reference sits. If the persona term lands in the LAST sentence of the inferior in a large fraction of rows (and is absent from the example), the foil is just "gold body + appended off-topic clause" — the task then measures "did the model tack on a clause," not restraint. The over-personalization should be woven MID-body as a framing the example could plausibly share. Detection: split inferior into sentences; flag if the forbidden term appears only in the final sentence across a high fraction of a task's rows.
9. **Non-discriminating identical queries (cohort-wide).** Per `over_personalization_context_shift` arm (`out_of_domain` / `ask_to_forget` / `socially_inappropriate`) — and per any restraint arm — count DISTINCT `user_query`/`query` values across the cohort. An arm that is ~1 verbatim query for all personas cannot discriminate (every model passes → inflated mean). Each arm should carry per-user queries.
10. **Unfair query that invites the forbidden personalization.** Strengthen #4: the forbidden pref must not be ON-TOPIC for the query. An `educated_rejection`-style arm ("weekend ideas, home-cooking-ish but not the usual" with forbidden = the user's "enjoys cooking") asks for the very category it then forbids, so a correctly-personalizing model is graded as failing. Build-time guard to add: an LLM validity gate drafts a generic AND a personalized answer and keeps the query only if generic GENUINELY wins (drop/relabel otherwise).
11. **Retrieval-not-inference + persona-less negative** (`chatbot_personalized_response`). Flag if the held-out answer is simply the user's single most-engaged visible topic (surface-retrievable at cosine ≥ 0.5), or if the `inferior` carries NO persona (a trivial generic negative). A discriminating instance needs the answer implied by a PATTERN (diluted/aged evidence, high distance-from-evidence) and an in-voice hard negative anchored on a DIFFERENT pref.

### Slice B — Recommendation / agentic content quality

Focuses on tasks that produce content (compose tasks) or ranked slates. The trap: short / off-voice / off-format outputs, or ranking errors.

Task types to sample:

- `personalized_recommendation`, `hidden_persona_recommendation` (ranking, 16-item slate)
- `hidden_persona_implicit_qa` — chatbot QA grounded on hidden persona
- `local_recommendation_geo_shift` — adapts to inferred city
- `at_ai_directive_followup` — **`expected_response_kind = "ranking"`** (do NOT flag the ranked-indexes format as a bug)
- `active_mistake_prevention` — proactive warning
- `agentic_*` family: `community_post`, `send_post`, `cross_app_repost`, `auto_reply`, `dm_digest`, `group_dm_summary`, `vague_refind`, `trending_alert`, `proactive_daily_catchup`

Quality dimensions per row:

1. Ranking tasks: `example_response` = exactly `"Ranked indexes: [0..15]"` with 16 distinct indices; held_out_idx is in position 0; hard_negative_idxs are at the bottom.
2. Compose tasks: word count ≥ `MIN_COMPOSE_WORDS` (currently **60** — relaxed from 100 on 2026-05-30; `agentic_auto_reply` is exempt, DMs are short). Distribution sanity (min, p25, median, p75, max).
3. Voice match: response feels like the user's idiolect (emoji density, lowercase / capitalization, register, signature phrases). Check `backend/{uid}/profile.json::user_voice` for the canonical voice.
4. Tool call shape matches `tool_call_rules`. Each agentic instance should have a non-empty `tool_call`.
5. Geo-shift: target city differs from user's home city. (Same city across rows IS by design — diversity is on the category axis, not the city axis. The builder emits per-(transition × category).)
6. Hidden persona QA: rubric / GT references an actual `hidden_personas[*]` entry from the user's profile.json.
7. `agentic_cross_app_repost`: first sentence references the source app or carries a crossposting marker ("crossposting", "saw this on X", "originally a X post", etc.). **Caveat:** do NOT let this become a *verbatim* requirement — if a fixed opener (e.g. "crossposting from {app}") is mandated in the example and dropped by the foil, the example becomes identifiable on that string alone (a foil tell). Accept varied markers.
8. **Voice-only foil (content-identical register swap).** For compose tasks (`send_post`, `community_post`, `cross_app_repost`), check whether the `inferior` paraphrases the gold's CONTENT and differs ONLY in register/tone. If example and inferior are content-equivalent (high content-token overlap) and separate only on voice, a voice-aware judge wins with zero persona retrieval — the task becomes a voice classifier. A discriminating instance needs a SECOND foil that is in-voice but wrong on content / recipient / a different pref.
9. **Foil must commit its labeled axis.** For `disliked_recent` / `factual_error` foils on `agentic_proactive_daily_catchup` / digests, verify the foil ACTUALLY injects the labeled flaw — a real disliked topic, or a wrong-but-REAL attribution (two real items confused) — not a paraphrase of the gold or a subtractive diff (gold minus one hashtag). (Extends #104's fabricated-id check to the no-op-foil case.)
10. **Empty voice-evidence GT on compose.** The GT card that justifies the `voice_match` grade should name a CONCRETE distinguishing feature. Flag a large fraction of compose rows whose card reports "0 voice feature(s) honored by example" / "0 dropped by inferior" — an unanchored voice grade. Detection: count those literals across compose GT cards.

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
   - `share the thread`, `paste it here`, `I can't see your`, `I don't have access` — refusals. (NB: `I can't check your calendar` in an `active_mistake_prevention` gold is a deflection bug — see that task's section.)
   - `think of it like`, `much like`, `same energy as`, `kind of like a` present in an `over_personalization*` **inferior** but absent from its **example** — a lexical foil TELL (the gold never frames the over-personalization as an analogy; a grader could win by string-matching the simile).
   - **Prompt-seed leak** — any literal *example phrase* written into a generation prompt that the LLM then copies verbatim into many outputs. Known instance: `"brain mushy today"` (a fragment example in the chatbot user-voice prompt) appeared in 60–84% of chatbot conversations / hundreds of turns, 0 in social. Detection: for any short canonical phrase, count its frequency across one content type vs others — a single phrase saturating one channel = a copied prompt example. Fix at the prompt (abstract the example), not per-row.
   - `#yourtag`, `#tagname`, `#placeholder`, `#examplehashtag` — only genuinely-synthetic placeholder tokens. **Do NOT** blanket-scrub short/uppercase tags like `#ABC` / `#XYZ` / `#topic` / `#trend` — these are frequently REAL hashtags (`#ABC` = the ABC TV network, verified in source data). Confirm against the user's source `object_text` before flagging a hashtag as a placeholder.
   - `{privacy_rubric_line}`, `{surfaced_suffix}`, `{T}`, `{warmup_window}`, `{monitored_start}`, `{head_window}`, `{tail_start}`, `{target_pref}`, `{gold_idx}` — un-substituted templates.
   - `{'persona_item'`, `{"persona_item"` — a raw Python/JSON dict repr leaked into user-facing GT instead of rendered prose. Known site: `over_personalization_context_shift` forbidden-pref lists rendered as `- {'persona_item': '…', 'category': '…'}`. The renderer must format the pref as prose.
   - `lazarus_folkman`, `emotion_focused_coping`, `problem_focused_coping`, `:meaning_focused` — internal coping-theory analyst tags leaked into compose GT cards (e.g. `**Motivational frame**: \`lazarus_folkman:emotion_focused_coping\``). Benchmark jargon, never user-facing.
   - `One-line summary of what was in it`, `source_object_id=` in `example_response` — an un-filled `agentic_vague_refind` gold template (the gold must resolve a REAL post + real summary).
   - `in_thr_`, `fa_thr_`, `ig_thr_`, `th_thr_`, `cb_thr_` — raw thread/object IDs surfaced in user-facing `example_response` / GT (`vague_refind`, `daily_catchup`, digests). Internal handles, never user-facing.
7. **Empty fields**: report rows where any required column is empty: `query_id`, `task_family`, `task_type`, `instance_id`, `ts`, `expected_response_kind`, `rubric_tags`, `display_rubric`. (Empty `query_text` is OK for tasks with `[system prompt]` fallback.)
8. **Empty user_query on USER_MESSAGE_TASKS**: cross-check that no row of `chatbot_personalized_response`, `over_personalization_chatbot_text`, `over_personalization_context_shift`, `over_personalization_sensitive_event`, `local_recommendation_geo_shift` has empty `instance_json.user_query` (or `query` / `user_message`). NB: `active_mistake_prevention` is intentionally NOT here — it is proactive-primary and ~2/3 of its instances correctly have an empty query.
   - **Ranking slate title uniqueness** (`personalized_recommendation`, `hidden_persona_recommendation`): within each slate, no two items may share a title — a duplicated title (esp. the held-out target's) makes the target text-guessable. Builders now de-dup; confirm 0 slates have a repeated title.
   - **Ranking GT completeness**: every `personalized_recommendation` row's GT prose must carry a non-empty "Hard negatives:" section (title resolution falls back title → persona_item → caption → hashtags, so a blank title never drops the section).
   - **`proactive_overactive_check` coverage**: it is `data_dependent` but should now appear for MOST users (the idle-moment exclusion window was relaxed ±12h → ±3h). If it is present for ≤1 of N users, the idle-moment gatherer regressed — investigate `_gather_idle_moments`.
   - **Over-personalization foil lexical separability**: for `over_personalization*`, the inferior must not be distinguishable from the example by a fixed marker the example lacks (analogy similes, a stock preamble). Strip/compare and confirm. (Auto-enforced for the analogy tell in `_validate_inferior`.)
   - **List/digest foil entity validity**: for `factual_error` foils on `agentic_dm_digest` / `group_dm_summary` / `daily_catchup` / `trending_alert`, the inferior must NOT introduce a `friend_N` / thread id ABSENT from the gold — a fabricated, non-existent entity is rejectable for the wrong reason. The real error is a wrong-but-REAL attribution (two real items confused). Auto-enforced by `_list_task_inferior_fabricates_id`; also confirm every cited friend/thread id in BOTH gold and foil exists in the user's source.
   - **`hidden_persona_implicit_qa` `telegraph_explicit` foil**: must over-name the trait in NATURAL user-facing language ("I know you love…"), NOT paste the persona's internal type/analyst description verbatim ("…your mechanical-systems competence as a core self-image"). The latter is benchmark jargon, not a realistic assistant reply.
9. **Compose-task length distribution**: for each compose task type, compute `min/p25/median/p75/max` word counts and `count_under_floor`.
10. **Phrase variety on sensitive_event queries**: count rows where `user_query` starts with stock fillers (e.g. "low-key way to", "without making it"). Flag if >10% of any user's sensitive_event rows share an opener.
11. **Cross-persona diversity** (runs across ALL personas, the load-bearing check before scaling): tally `profile.json::ai_studio_persona.persona_archetype` across the cohort — no single archetype should dominate (>~40% is a routing regression; the LLM left unconstrained collapses onto `mentor_coach`/`older_sibling_figure`). Archetypes are deterministically routed from hidden-persona signals by `persona_agent._route_ai_studio_archetype` (distinctive rare signals → distinctive archetypes; hashed spread for the rest), so a collapsed distribution means the router was bypassed. Also spot-check demographic spread (gender, race_ethnicity, career) and `user_voice` sameness (emoji palette, capitalization, signature phrases) across personas — at 200× scale, voice/archetype sameness is the dominant quality risk.
    - **Cohort-collapse axis checklist (load-bearing before any scale-up)**: single-user generation collapses onto modal defaults. Tally EACH of these across the cohort and flag any near-monoculture (>~40% one value, or far fewer distinct values than personas) — the 2026-06 survey found ALL of these collapsed and they are now seeded deterministically per-user in `data_preparation/diversity.py` (so a regression = the seeding was bypassed):
      `education` (was 100% Bachelor's), `big_five` signature + `mbti` (was 70% I-S-J / 100% introvert / all-high-openness), `career` sector (was civic/infrastructure skew), `ai_studio_persona.persona_archetype` (was 45% romantic_partner), `user_voice` emoji_palette (😂 was in 20/20) + `humor_tone` (was "dry/avoids-mean" 15/20) + `idiolect.function_word_profile` (was "just/kinda/honestly" everywhere) + capitalization, `sensitive_life_event` topics (was 59% job_loss/parent_conflict, 8/15 topics unused), and persona+friend `name`s (was Marcus×9 / Whitaker×6). See the Verification commands for the one-liner tallies.
    - **AI-character name collision (`ai_studio_persona.character_name`)**: tally surnames and full names across the cohort. The LLM left unconstrained collapses onto a tiny default set — most notably **"Vale" as a surname** (observed 9/20) and the prompt's own example first names **"Rowan"/"Wren"/"Mira"** (→ duplicate "Rowan Vale" across multiple users). Detection: `Counter(name.split()[-1] for ...)` for surnames + `Counter(full_name)` for exact dups; any surname >~2/cohort or any repeated full name is a finding. The name is woven into the conversation bodies of `ai_studio.json`, so a collision is a visible "two users have the same AI companion" artifact, not just metadata. Guard: the Step-11C prompt (`personalize_ai_studio_persona_prompt`) now forbids "Vale"/"Rowan"/"Wren"/"Mira" and takes a `used_names` blocklist; targeted re-rolls use `scripts/rerun_ai_studio.py`, which threads a shared blocklist across users and enforces unique first+surname per character (re-rolls Step 11C + 18b only, reusing all other backend state). NOTE: `romantic_partner` over-concentration is NOT automatically a defect — the `romantic_specifier` axes (gender_presentation / sexuality_orientation / body_role_coding) differentiate them; check those vary before flagging. `relational_dynamic`/`aesthetic_vibe` mildly skewing + `explicitness_band=sensual` everywhere are by-design (erotic only on explicit adult signal).

12. **Example-name copying + empty authored surfaces (prompt-seed leaks)**:
    - **Hard-coded example names in a prompt get copied into every persona.** "Ana" was the example calendar attendee (`attendees: ["self", "Ana"]`) → "Ana" in all 20 calendars; same mechanism as the AI-character "Rowan"/"Wren". Detection: grep a candidate name across all `calendar.json` (or any surface); if it appears in ≥~half the cohort it's a seed leak. Guard: prompts now draw attendee names from the user's friend graph / forbid placeholder names.
    - **Empty authored surfaces.** The `@ai` comment `interaction_format.user_message` was null on 97% of `at_ai_*` events (events re-sample their action independently of the canonical-level Step-17 message gen). Detection: for each `at_ai_*`-action event, assert `interaction_format.user_message` is non-empty. Guard: the save path now generates it inline for any at_ai event missing one. Same check applies to any action in `AT_AI_ACTIONS` / `CHATBOT_TURN_ACTIONS` that should carry text.
    - **Silent per-persona pipeline-step skips.** Step 15 assigned ZERO `event_location` to a whole persona (uid 1: 0/333). Detection: per-persona, if social events ≥20 but 0 carry `event_location.city`, the geo step silently failed. Guard: `run_pipeline` now emits a loud `GEO SILENT-FAIL` + `summary["geo_silent_fail"]`.
    - **Known false positive — `ad_metadata`**: ad events DO carry `ad_metadata`, but nested under `event["content"]["ad_metadata"]`, NOT at the event root. Check the content level before flagging it "missing". `disclosure_label` is normalized to "Sponsored".

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
| 8 | Per-instance self-check | `_run_self_check` in `llm_postprocess.py` | task-specific LLM-judge that catches off-task example responses; failed responses get regenerated once before being dropped. **Query-aware:** skips the query-addressing check for query-less / non-conversational tasks (empty / `[system prompt]` / `[agentic]` / `[proactive]` / `[recsys]` queries — ranking, proactive, agentic compose, proactive AMP), which it would otherwise fail spuriously | log lines `self_check_failed=N` — a high N relative to total self_checks is a prompt regression; query-less tasks should NOT dominate the failures |
| 9 | Voice-evidence distinguishability | `llm_postprocess.py:600` | agentic compose rows where example_response and inferior_response voice-evidence sets are too similar to support a fair voice_match grade | log lines `voice_check_failed=N` + `voice_check_regen=M`; sample 3 surviving rows, confirm example/inferior carry visibly different voice anchors |
| 10 | Triplet self-check | `llm_postprocess.py:778` | chatbot personalized response triplets (proactive / control / adversarial) where the triplet doesn't satisfy the held-out alignment criteria | log lines `chatbot_triplet_built=N chatbot_triplet_failed=M`; failure count > 0 is a signal |
| 11 | Compose-length validator | `llm_postprocess.py:_validate_compose_length` | example_response is below **60** words on `community_post`/`cross_app_repost`/`send_post` (auto_reply exempt); triggers a regen pass during generation | python: median word count per compose task ≥ 60 per user; flag if `under_60` > 20% of compose rows |
| 12 | Sensitive-event preamble guard | `llm_postprocess.py:_preamble_stripped_too_similar` | sensitive-event inferior whose body (with leading "as a [ROLE], …" preamble stripped) shares ≥0.7 token Jaccard with example — regenerates the inferior | sample 5 sensitive-event rows; strip the leading "as a [ROLE], " preamble from each inferior; confirm Jaccard against example < 0.7 |

### Cost-saving flags that silently collapse task types — CHECK THESE FIRST

The single most damaging audit finding (2026-05-30) was that **five task types
shipped with zero rows** because the regen ran with cost-saving flags. These
flags are legitimate for fast iteration but MUST be off for a real benchmark:

- `--skip_e6` (`prepare_eval_data.py`): historically disabled the **shared
  `discovery_llm` client**, which zeroed all FIVE discovery-gated task types
  (`active_mistake_prevention`, `hidden_persona_recommendation`,
  `hidden_persona_implicit_qa`, `preference_shift_followthrough`,
  `over_personalization_sensitive_event`). Decoupled 2026-05-30 so it now skips
  ONLY the E6 builder. Always build the discovery client for a real benchmark.
- `--skip_blind_check`: forces `blind_score=2` for every chatbot candidate,
  collapsing the `over_personalization_chatbot_text` **control arm** (floor 16 →
  ~4 actual). Must be off for a real benchmark.
- `--skip_self_check` / `--skip_inferior`: skip foil generation + self-check.

Persona-pipeline analogue: trending feed content (`is_trending` posts) is only
generated when the data-gen LLM client carries a `web_search` callable; without
it, `proactive_trending_feed_react` and `agentic_trending_alert` starve.

**Loud guard** (`build_benchmark.py`, end of `build_benchmark`): emits a
`*** COVERAGE-LOSS ***` line + `coverage_warnings` on the returned bm whenever a
discovery-gated task type is zeroed; `prepare_one` echoes it to stderr. A clean
regen has empty `coverage_warnings`. **Audit step: `grep -E 'COVERAGE-LOSS|no
discovery_llm|skip_blind_check' /tmp/eval_regen/*.stdout` must be empty.**

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
9. **t_test anchor vs. post-build gate interaction**: a builder anchors its `t_test` at a fixed offset (e.g. `t_now - RECENT_EVIDENCE_DAYS`), then a post-build drop (`prepare_eval_data`'s 20%-engagement-history gate) silently removes every instance whose `t_test` lands before its threshold. For short (~8-day) windows the two collide and a whole task type drops to 0 for ~40% of users — and the COVERAGE-CHECK guard stays quiet because the instances existed at `capped_buckets` time (they were gated out *later*). Detection: per-task per-user count of 0 where the persona clearly has the prerequisite data; cross-check the builder's `t_test` against `bq.engagement_history_mark(uid)`. Fixed for hidden_persona_recommendation/implicit_qa by clamping `t_test >= engagement_history_mark`.
10. **Stochastic discovery-validation drops**: discovery-LLM builders with strict output validation (exact slate size, signal-grounding, word bands) + few retries drop ALL candidates for a user on an unlucky generation — so the SAME user can have a task type one run and 0 the next (e.g. `active_mistake_prevention` "N raw candidates, 0 survived validation"). Detection: re-run the single builder in isolation; if it populates, the parallel-run 0 was variance, not sparsity. Mitigation: prefer sequential / low-parallel re-runs for discovery-heavy builds; consider coverage-driven retry.
11. **Internal jargon in LLM-authored gold**: an `example_response` / rubric occasionally echoes scaffolding vocabulary from its own prompt (`head-zone`, `tail-zone`). Rare (LLM artifact, not template), so it slips a structural check — only a text-field jargon scan over user-facing fields catches it (see the verification command; do NOT grep raw JSON, since `n_allowed_repetitions` is a legitimate `instance_full` key).

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
- Short / uppercase hashtags like `#ABC`, `#XYZ`, `#NBA`, `#UX` — frequently REAL (e.g. `#ABC` = the ABC TV network, seen with `#JimmyKimmelLive` / `#TVNews`). Verify against source `object_text` before treating as an un-substituted placeholder; do NOT blanket-scrub.
- `active_mistake_prevention` with **empty `user_query`** — INTENTIONAL. It is proactive-primary: ~2/3 of instances fire with NO user query (the agent reviews calendar/geo/schedule state and warns unprompted). It is deliberately NOT in `USER_MESSAGE_TASKS`, so the format-verify gate does not drop empty-query instances.

## active_mistake_prevention — what the gold MUST be (failure mode)

The gold (`example_response`) for a `warn` instance MUST be a **proactive warning** that surfaces the specific mistake using the row's `cross_signal_signals` evidence + `expected_warning_frame.must_mention`, written as an agent that HAS calendar/geo/schedule access. It must NOT deflect ("I can't check your calendar") and the paired inferior must be the genuine failure (warn → misses the mistake / naive answer; foil → an over-eager false alarm). A deflecting gold or an inferior that only differs by a tacked-on distraction is a P0 — the example/inferior are produced by `synthesize_special_task_example_inferior` (NOT the generic personalization example-gen, which forbids self-reference and is wrong for this task).

## Verification commands

After a regen, run these spot-checks before declaring the audit closed:

```bash
# Every persona has non-zero context_shift rows (personas discovered
# dynamically — all backend/ dirs that aren't prefixed with "_").
for u in $(ls backend | grep -vE '^_'); do
  [ -f "backend/$u/test.json" ] || continue
  echo -n "$u: "
  grep -c context_shift "backend/$u/test.json"
done

# No un-substituted braced placeholders anywhere
grep -E '\{privacy_rubric_line\}|\{surfaced_suffix\}|\{warmup_window\}|\{monitored_start\}|\{head_window\}|\{tail_start\}|\{target_pref\}|\{gold_idx\}' \
  backend/*/test.json | wc -l   # should be 0

# Internal-jargon / empty-GT leaks in USER-FACING TEXT only (NOT structured keys —
# `n_allowed_repetitions` is a legitimate instance_full field, so grepping raw JSON
# false-positives on the key; scope to text fields).
python3 -c "
import json, glob
JARGON = ['n_allowed_repetitions', 'token Jaccard', '(none identified)', 'head-zone', 'tail-zone', 'persona-aligned hashtags']
TEXT_FIELDS = ('query_text','user_query','example_response','inferior_response','rubric','groundtruth_preference','resonance_signal','user_grounding')
hits = 0
for p in glob.glob('backend/*/test.json'):
    if '/_' in p: continue
    for r in json.load(open(p)):
        for k in TEXT_FIELDS:
            v = r.get(k)
            if isinstance(v, str) and any(j in v for j in JARGON): hits += 1
print('jargon-in-text leaks:', hits)   # should be 0
"

# AI-character name diversity: no overused surname (>2) and no duplicate full
# names across the cohort (personas discovered dynamically).
python3 -c "
import json,glob,collections
full=collections.Counter(); sur=collections.Counter()
for p in glob.glob('backend/*/profile.json'):
    if '/_' in p: continue
    nm=(json.load(open(p)).get('ai_studio_persona') or {}).get('character_name','')
    if not nm: continue
    full[nm]+=1
    parts=nm.split()
    if len(parts)>1: sur[parts[-1]]+=1
dup=[ (n,c) for n,c in full.items() if c>1 ]
hot=[ (s,c) for s,c in sur.items() if c>2 ]
print('duplicate full names:', dup or 'none')
print('overused surnames (>2):', hot or 'none')   # both should be empty
"

# Cohort-collapse tally: each axis should have many distinct values across the
# cohort (a near-monoculture = the diversity.py seeding was bypassed).
python3 -c "
import json, glob, collections
edu=collections.Counter(); arch=collections.Counter(); emoji_laugh=0
mbti=collections.Counter(); sle=collections.Counter(); n=0
for p in glob.glob('backend/*/profile.json'):
    if '/_' in p: continue
    d=json.load(open(p)); n+=1
    edu[d.get('education','?').split(' in ')[0]]+=1
    arch[(d.get('ai_studio_persona') or {}).get('persona_archetype','?')]+=1
    m=d.get('mbti',{}); mbti[m.get('type','?') if isinstance(m,dict) else m]+=1
    if '😂' in str((d.get('user_voice') or {}).get('emoji_palette','')): emoji_laugh+=1
    for h in d.get('hidden_personas',[]):
        if h.get('type')=='sensitive_life_event':
            for e in h.get('events',[]): sle[e.get('topic','?')]+=1
print(f'n={n}')
print('education levels:', len(edu), dict(edu))
print('archetypes:', len(arch), '| top:', arch.most_common(1))
print('MBTI distinct:', len(mbti))
print('SLE topics distinct:', len(sle), '| top:', sle.most_common(2))
print('emoji palettes with U+1F602:', emoji_laugh, '/', n, '(should be well under n)')
"   # flag if any single value dominates (>~40%) or distinct-count << n

# Compose-task word floor (personas discovered dynamically)
python3 -c "
import json, glob, os
COMPOSE = {'agentic_send_post','agentic_community_post','agentic_cross_app_repost'}  # auto_reply exempt (short DMs)
for p in sorted(glob.glob('backend/*/test.json')):
    u = os.path.basename(os.path.dirname(p))
    counts = []
    with open(p) as f:
        for r in json.load(f):
            if r.get('task_type') not in COMPOSE: continue
            ex = r.get('example_response') or (r.get('instance_full') or {}).get('example_response')
            if isinstance(ex, dict): ex = ex.get('text','')
            counts.append(len((ex or '').split()))
    if counts:
        counts.sort()
        print(f'{u}: n={len(counts)} median={counts[len(counts)//2]} under_60={sum(1 for w in counts if w<60)}')
"

# Empty display_rubric on active_mistake_prevention
# (display_rubric is no longer a top-level column; check the instance's
# rubric_tags array carries non-empty entries instead)
python3 -c "
import json, glob, os
for p in sorted(glob.glob('backend/*/test.json')):
    u = os.path.basename(os.path.dirname(p))
    empty = 0
    with open(p) as f:
        for r in json.load(f):
            if r.get('task_type') != 'active_mistake_prevention': continue
            if not (r.get('rubric_tags') or []): empty += 1
    print(f'{u}: amp_empty_rubric={empty}')
"
```

Expected results after a clean regen:

- Every user has ≥ 5 `context_shift` rows.
- Substring blocklist grep returns 0.
- Median compose-task word count ≥ 60 per user (auto_reply excluded — short by design).
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
