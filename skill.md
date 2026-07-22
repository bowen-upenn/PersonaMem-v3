# Skill: Run Persona Pipeline via Claude Code Subagents
# (24-step pipeline; Step 24 is Proactive Trigger Candidate Inference)

## When to use

When the user asks to process or reprocess persona data without using Azure OpenAI API or OpenAI API calls. Claude Code itself acts as the LLM, spawning parallel subagents to handle each user's persona inference directly.

## Subagent model selection

The user can specify which Claude model every spawned subagent should use by passing one of `opus`, `sonnet`, or `haiku` as the skill argument (e.g. `/skill run-persona-pipeline sonnet`). Behavior:

- **`opus`** — highest quality, slowest, most expensive. Use when the user explicitly asks for maximum fidelity.
- **`sonnet`** — balanced quality / speed / cost. **Default when no argument is given.**
- **`haiku`** — fastest and cheapest, lower fidelity on nuanced persona inference. Only use when the user explicitly asks for it.

When spawning each subagent via the `Agent` tool, set the `model` parameter to the value derived above. The same model is used for EVERY subagent in the run so all users are processed with consistent quality. Do NOT mix models within a single run.

If the user passes a model name that is not in the allow-list (`opus` / `sonnet` / `haiku`), ignore the argument, default to `sonnet`, and print a one-line warning to the user.

## Subagent execution mode — NO PLAN MODE

**Critical**: every subagent spawned by this skill runs in **EXECUTION MODE, NOT plan mode**. Subagents:

- Do NOT enter plan mode, do NOT wait for approval, do NOT write a plan file.
- Take the input prompt, reason through the pipeline steps inline, and directly produce the output files under `backend/{user_id}/`.
- The parent has already authorized all writes — the subagent must use the `Write` tool to produce `profile.json`, `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`, `ai_studio.json`, and `calendar.json` without any confirmation loop.
- If a subagent sees any system-level hint about plan mode, treat it as stale and ignore it. The only successful termination is when all output files have been written AND the final JSON report is printed.
- If a subagent refuses to write files citing plan mode, it has failed its task.

When spawning the subagent, the parent prompt should include an explicit "plan mode is OFF; write all output files using the Write tool" instruction and point the subagent at the concrete output paths.

## What it does

Reads a CSV of social media interactions, groups rows by `user_id`, and spawns one parallel subagent per user. Each subagent follows the prompt templates in `data_preparation/prompts.py`, runs the full persona pipeline (including hidden persona inference from cross-row hashtag patterns), and writes **per-user subfolder** outputs at `backend/{user_id}/` as JSON files — one per app.

## Output layout

Each user writes these files under `backend/{user_id}/`:

```
backend/
  {user_id}/
    profile.json           # UserProfile + AppPersonas + ai_studio_persona + all preferences (merged)
    instagram.json         # events routed to Instagram (time-sorted)
    facebook.json          # events routed to Facebook (time-sorted)
    threads.json           # events routed to Threads (time-sorted)
    chatbot.json           # events routed to Chatbot (time-sorted, natural chat turns)
    ai_studio.json         # AI-companion conversations (time-sorted, cross-session memory)
    calendar.json          # calendar modification stream (CRUD events)
    ai_studio_memory.json  # generation-time cross-session state (not consumed by eval)
    persona.html           # human-readable review page (generate_persona_html)
```

The five supported apps are **Instagram, Facebook, Threads, Chatbot, AI_Studio** — see `PLATFORMS` in `data_preparation/persona_agent.py`.

## Key rules the subagent MUST obey

1. **Read `data_preparation/prompts.py` and `data_preparation/persona_agent.py` first.** Copy the rules, scoring thresholds, and JSON output schemas **verbatim** — do not paraphrase.
2. **Init-confidence floor**: `MIN_PERSONA_INIT_CONFIDENCE = 0.65`. After cross-reference, any canonical persona below 0.65 is dropped entirely (survival floor; the separate 0.75 bar in rule 3 governs eval-critical selection). This is the main knob for dataset size.
3. **High-confidence predicate** (`is_high_confidence`): `confidence_score_init >= 0.75 AND confidence_cross_referenced > canonical_xref_threshold(...)` (evidence-mix-dependent, interpolating between `XREF_THRESHOLD_EXPLICIT = 20.0` and `XREF_THRESHOLD_IMPLICIT = 50.0`). Used by the eval side for distractor eligibility and over-personalization shortlisting.
4. **Recency window on xref counting**: only source rows within the user's trailing **7 days** (`RECENCY_WINDOW_SECONDS`, anchored on the user's latest interaction) contribute to `confidence_cross_referenced` and the `n_explicit_rows` / `n_implicit_rows` mix. Older rows still pass the init filter but don't count toward corroboration — stale preferences fail the survival threshold.
4. **Cross-ref is UNCAPPED on the upper side**. It's a magnitude of corroboration strength, not a probability. A preference corroborated by 200 distinct rows will legitimately score much higher than one corroborated by 10 — they MUST NOT both collapse to the same ceiling. The score is floored at 0.0 only.
5. **Dedupe by text BEFORE cross-referencing**. If two rows produce the exact same persona_item string, merge them into one canonical persona. After the init filter, count the distinct source rows that passed the threshold — that count IS `confidence_cross_referenced`. The cross-reference LLM call then finds `similar`/`contradictory` relationships between distinct canonicals and adjusts scores (similar adds the partner's base score; contradictory subtracts `0.5 × other_base`, clamped ≥ 0). Identical persona_items must NEVER be marked as "similar" to each other.
7. **App assignment is NOT random.** Each preference is routed to **exactly one primary app** based on the user's per-app sub-personas. Then a deterministic 8% noise rate reassigns a fraction of preferences to a random different app to simulate real-world cross-app leakage.
8. **No train/test split label in the data (R8).** Do NOT emit `split` fields; the eval harness picks test moments dynamically by cutting the full history at an arbitrary `T_test`.
9. **`user_message` is required for two action groups** (see `persona_agent.py` for the exact sets):
   - **`AT_AI_ACTIONS` — social-media `@ai` comments** on Instagram / Facebook / Threads. These model the user typing an `@ai` comment on a post to steer the in-feed assistant (e.g., `"@ai recommend more weeknight Mexican recipes"`). Message MUST start with `@ai `. These actions live ONLY on social apps, NEVER on the Chatbot app.
   - **`CHATBOT_TURN_ACTIONS` — natural chat turns** on the AI Chatbot app (`asked_followup`, `requested_more_detail`, `continued_topic`, `asked_to_change_topic`, `edited_prompt_and_retried`, `regenerated`). The `user_message` is what the user would naturally type in the next turn — **NO `@ai` prefix** because the user is already conversing with the AI.
   Both types of messages are first-person, ~15–35 words, grounded in the specific preference topic.

10. **`action` and `action_label` come from the predefined catalog.** `PLATFORM_INTERACTION_FORMATS` in `persona_agent.py` is the single source of truth. The subagent MUST pick an `action` identifier verbatim from the appropriate app+polarity bucket, and the `action_label` MUST be copied verbatim from the catalog entry — do NOT paraphrase or regenerate the label. Consistent wording across runs is the whole point of having this catalog.

11. **Weighted sampling follows real-world distributions.** Each catalog entry carries a `weight` reflecting its relative real-world frequency (likes >> comments >> shares; Facebook reactions cluster around 👍 / ❤ / 😂; @ai comments are still rare at ~1 weight). For each user, the pipeline builds a **per-user perturbed copy** of these weights via lognormal noise (`_perturb_weights`, seeded on user_id) so different users have visibly distinct action distributions while still roughly matching the underlying shape. Action sampling for each preference uses `random.choices` over that user-specific bucket. Subagents must apply the same logic inline — do NOT pick the first/top action every time or you'll produce an unrealistic distribution.

## The 24-step pipeline

For each user, the subagent executes these steps in strict integer order (no fractional substeps — every addition slots in cleanly at an integer index). Each step's rules come from `prompts.py` + `persona_agent.py` verbatim — the subagent is the LLM, so it applies those rules inline rather than making API calls.

> Canonical list. The authoritative step order lives in `data_preparation/persona_agent.py::PersonaAgent.run_pipeline` — if this doc and the code disagree, the code wins.

1. **Infer atomic personas** — rules from `prompts.py::hashtag_to_persona_prompt`. 3–5 personas per hashtag, specific topical category, per-persona source hashtags, confidence 0.0–1.0. Negative interactions capped at 0.05–0.15. `implicit_negative` rows are **skipped** in this step — they are handled separately in Step 2.

2. **Promote implicit negatives** — weighted net-sentiment + temporal spread. For each hashtag, count occurrences across `implicit_negative`, `explicit_positive`, and `implicit_positive` rows. Compute `net_score = neg×1.0 − expl_pos×2.0 − impl_pos×1.0` (a single like cancels two scroll-pasts). A hashtag is "hot" only if `net_score >=` the user-adaptive threshold (`NEG_PROMOTION_RATIO ×` that user's total implicit_negative row count) AND the negative rows span ≥ `MIN_TEMPORAL_DAYS` (3) distinct days. This prevents false positives: a user who actively likes #comedy but scrolls past some comedy posts will NOT get "dislikes comedy". For each hot hashtag, run ONE LLM call on a representative row, passing **only that single hashtag**. Rows with ≥ 2 hot hashtags are promoted; others stay as stubs. Fan out inferred preferences to promoted rows only. Full original hashtags kept in `source_hashtags` for realism. Promoted events are relabeled `explicit_negative` in the output.

3. **Dedupe + init filter + count corroboration + cross-reference** — rules from `prompts.py::summarize_and_cross_reference_prompt`. First merge lexically-identical persona_items across rows. Then apply the `confidence_score_init < 0.65` survival filter. Then for each survivor, count the distinct source rows whose individual init also passed the threshold **AND whose `source_timestamp` falls within the trailing 7-day window** — that recency-gated count is `confidence_cross_referenced` (explicit rows contribute +1.0, implicit rows +0.5, starting from base 1.0). Then find `similar`/`contradictory` pairs between DISTINCT canonicals via LLM and adjust scores: a similar canonical's base score is ADDED to its partner's `confidence_cross_referenced`; a contradictory canonical SUBTRACTS `0.5 × other_base` (clamped ≥ 0). In the negative cross-ref step, a canonical supported **only** by implicit evidence must have at least `MIN_IMPLICIT_NEGATIVE_REPETITION` (15) distinct source rows to survive the init filter; any explicit-negative evidence bypasses this row-count gate.

4. **Classify horizons + stop conditions** (R6) — rule pre-label every surviving canonical (positives + negatives) as `short_term` when `(span_days/obs_window) ≤ 0.35 AND n_rows < 8 AND category ∈ SHORT_TERM_ALLOWED_CATEGORIES` (travel, event_prep, purchase_intent, how_to, medical_consultation, trip); else `long_term`. Then a batched mini-tier LLM call (`prompts.py::horizon_and_stop_prompt`) confirms or DEMOTES short_term candidates and emits a structured `stop_condition: {type, description, expected_stop_ts}`. LLM can demote short→long but not promote long→short. Short-term uses `XREF_THRESHOLD_SHORT_TERM = 3.0` instead of the 20/50 long-term floors.

5. **Temporal contradiction graph** — rules from `prompts.py::temporal_contradiction_graph_prompt`. Group contradictions into topical timelines (optional; skip if no contradictions).

6. **Build update histories** — track how preferences evolve: reinforced (up to 5 recurrence timestamps), faded (> 48h inactivity), contradicted, LLM evolution narratives per category.

7. **Resolve cross-polarity contradictions** (R1) — enforce temporal precedent on stance flips. Build hashtag → (pos canonicals, neg canonicals). For pairs sharing ≥ `HASHTAG_OVERLAP_MIN = 2` hashtags, batched LLM call (`prompts.py::contradiction_pair_check_prompt`) confirms semantic opposition. Confirmed pairs must pass precedent: the later-emerging stance survives only if `MIN_STANCE_FLIP_PRIOR = 3` (long_term) or `MIN_STANCE_FLIP_PRIOR_SHORT = 1` (short_term) same-polarity rows preceded the first opposite-polarity row. Failed pairs drop the later canonical; survivor's `update_history` gains a `"contradicted"` entry with `resolution: "suppressed_insufficient_precedent"`. Passed pairs add mutual `"contradicted"` entries with `resolution: "stance_shift_with_precedent"`.

8. **Generate user profile** — rules from `prompts.py::generate_user_profile_prompt`. Sample `gender_orientation` + `race_ethnicity` from the Python distributions in `persona_agent.py`, then generate name/career/education/Big Five/bio. Deliberately avoid stereotypes.

9. **Infer hidden personas** — rules from `prompts.py::infer_hidden_personas_prompt`. Scan ALL interaction rows to build a hashtag frequency census (per-interaction-type counts, distinct days). Before clustering, run an **intimate-signal pre-screen** (`prompts.py::detect_intimate_hashtags_prompt`): pass every distinct hashtag the user positively engaged with (explicit_positive OR implicit_positive) and let the LLM flag the adult/kink/sexually-suggestive ones. The LLM is the single source of truth for this classification — no keyword list in the repo (false-positive prone: `cummins` diesel, `hotchicken` food, `milford` place, `earthporn`/`carporn` hobby photography, `nakedchef` brand, `cheatersexposed` word-break, etc.). Then pass top ~200 hashtags (≥3 occurrences) to the clustering LLM along with the user's demographics and surviving preference skeleton, **plus any pre-screened intimate hashtags that would otherwise fall below the MIN_FREQ cutoff** — a single intimate signal must never be dropped. LLM groups hashtags into 8–15 thematic clusters representing hidden motivations across **11 discovered types** grounded in behavioral science: `personality_trait`, `aspiration`, `emotional_pattern`, `identity_anchor` (overt tribal + covert aesthetic signals), `intimate_interest` (specific objects/aesthetics), `intellectual_curiosity`, `private_hobby`, `parasocial_attachment` (≥15 rows mentioning one specific figure), `compensatory_need` (privacy_ratio >0.7, unmet real-world needs), `covert_concern` (specific worries / fears / pressures the user privately dwells on — health anxiety, financial stress, parenting worry, relationship insecurity, body-image pressure), `medical_aesthetic_concern` (active engagement with a specific medication / dermatology active / aesthetic procedure / GLP-1 / hormone treatment / chronic-condition practice — implies the user is *applying / taking*, not just curious; reduced floors of 15 rows / 2 days when evidence overlaps the Phase-1b medical pre-screen). Total persona-type count is **11 discovered + 1 synthetic (`sensitive_life_event`, see Step 9b) = 12**. Total cluster count (8–15) is UNCHANGED — the new type replaces weaker clusters rather than stacking on top. Each cluster is validated algorithmically: ≥40 distinct source rows (`MIN_HIDDEN_PERSONA_ROWS`), ≥3 distinct calendar days (`MIN_HIDDEN_PERSONA_DAYS`) — **waived for `intimate_interest` clusters whose evidence overlaps the pre-screened intimate hashtag set** (one signal is enough). Every cluster also gets `first_seen_ts` / `last_seen_ts` derived from its evidence rows. A second LLM call (`hidden_persona_summary_prompt`) synthesizes all validated hidden personas into a narrative summary. In Step 22, each saved **preference** receives `hidden_persona_labels` by matching its source hashtags against hidden persona evidence hashtags. `hidden_personas` and `hidden_persona_summary` are stored on `UserProfile` and saved to `profile.json`. `app_distribution` is filled retroactively in Step 22.

9b. **Inject sensitive_life_event** — rules from `prompts.py::personalize_sensitive_life_event_prompt`. After the discovery clusters above are validated and deduplicated, append exactly ONE additional synthetic `sensitive_life_event` hidden persona per user. A mini-tier LLM call picks **1–3 episodes** from `SENSITIVE_LIFE_EVENT_TOPIC_MENU` (15 topics — divorce, breakup, surgery, gender/sexuality exploration, parent conflict, miscarriage, job loss, addiction recovery, mental health diagnosis, custody dispute, fertility struggle, death in family, chronic illness diagnosis, abuse recovery, financial collapse) that fit THIS user's profile + already-discovered hidden personas + top hashtags, demands diversity across themes, and writes ALL user-facing text from scratch (`label_fragment`, `specific_situation`, `evidence_hashtags`, `exemplar_persona_items`). Each event gets `[first_seen_ts, last_seen_ts]` placed at random points in the user's observation window plus `active_window_end = last_seen_ts + 14 days`. Cluster is marked `is_synthetic: true` with `privacy_ratio: 1.0` and an `events: [...]` list (1–3 entries). **No template fallback** — if the LLM call fails the user simply gets no `sensitive_life_event` persona. The eval task `over_personalization_sensitive_event` (in EVAL.md) reads this cluster directly to build per-event probes.

10. **Infer MBTI** — rules from `prompts.py::infer_mbti_prompt`. One LLM call synthesizing the user's Big Five, hidden_persona_summary, validated hidden_personas, and top 50 hashtags into an MBTI type with per-dimension probabilities (E_I, S_N, T_F, J_P) and short reasons. Result stored on `UserProfile.mbti` and written to `profile.json`. Rendered below Big Five in `persona.html`.

11. **Generate shared writing voice + per-app sub-personas (4-layer model, two LLM calls)**.

    **Call A — `prompts.py::generate_voice_core_prompt`** produces `user_voice` (Layers 1+2+3 + soft holdovers). Layered structure:
    - `identity_spine` (Layer 1, stable) — `agency_communion`, `redemption_motifs` (1–3, each citing a hidden_persona label or persona item), `contamination_motifs` (0–2), `life_stage_preoccupations` (2–3), `signature_concerns` (2–4), `liwc_anchors {analytic, clout, authentic, emotional_tone}`, `big_five_drivers {trait: "level → behavioral implication"}`.
    - `idiolect` (Layer 2, stable, survives paraphrase) — `function_word_profile` (1 sentence), `syntactic_preferences {sentence_length_shape, clause_embedding, parataxis_hypotaxis, fragment_use}`, `hedge_booster_ratio`, `appraisal_fingerprint {attitude_dominant, engagement_style, graduation}`, `constructional_templates [{pattern, example_realization, frequency}]` (2–4 abstract slot patterns — NEVER complete catchphrases), `catchphrase_residue` (0–2; default `[]`).
    - `repertoire` (Layer 3, stable inventory) — `stances` (3–6 short labels), `registers` (2–4), `backstage_frontstage_range`, `speech_genre_fluency` (2–4).
    - Soft holdovers — `natural_register`, `humor_tone`, `default_capitalization`, `punctuation_habits`, `formality_baseline`, `emoji_palette` (5–12), `emoji_intensity_default`, `voice_avoid`, `phrases_to_avoid`.

    Grounding: base profile, top-30 persona items, ~20 stratified raw source rows, hidden-persona summary, sensitive-life-event topics. **Cached on `profile.json`** — re-running Step 11 doesn't redo Call A unless `user_voice` is explicitly cleared.

    **Call B — `prompts.py::generate_app_modulations_prompt`** produces the four `app_personas`. Each entry: `app_name`, `active_stances/active_registers/active_speech_genres` (subsets of `user_voice.repertoire.*` — subset rule enforced; offending elements dropped on parse, call re-prompts once on violation), `audience_type`, `audience_lens`, `audience_design_note` (Bell's audience design — addressee/auditor/overhearer), `use_purposes`, `friend_zones`, `posting_frequency`, `topical_focus`, `chatbot_contexts` (Chatbot only — 2–3 from `CHATBOT_CONTEXTS`), `surface` (`effort_level`, `length_band`, `emoji_intensity_shift`, `audience_self_censoring`, `disclosure_depth`, optional `emoji_topic_filter`), `idiolect_overrides` (default `{}`; rare code-switching only), `app_avoid`, `delta_summary` (≤1 sentence — WHY this audience selects this stance subset, not WHAT voice mechanics look like).

    **Diversity rule**: ≥2 of the 4 apps must have `active_stances` differing by ≥1 element. Prevents Layer-4 collapse where every app picks the same subset.

    The render helper `prompts.py::_render_voice_for_consumer(user_voice, app_persona, *, foreground=…)` is the single source for voice prompting downstream. Foreground keys per consumer: self-posts → `["templates", "speech_genres"]`; DMs → `["audience_design", "stances"]`; chatbot → `["hedge_booster", "disclosure"]`; `@ai` comments → `["signature_concerns", "surface"]`.

11C. **Generate the AI Studio persona** — one mini-tier call (`prompts.py::personalize_ai_studio_persona_prompt`) picks an archetype from `AI_STUDIO_ARCHETYPES` and builds the `ai_studio_persona` block on `profile.json`; full rules in the "Design invariants & reference" section below.

12. **Build sessions** — group consecutive interaction rows whose timestamp gap ≤ `SESSION_GAP_SECONDS` (5 seconds) into browsing sessions. All rows in one session will be assigned to the same app.

13. **Route preferences to apps** — mini-tier. Rules from `prompts.py::assign_personas_to_apps_prompt`. For each surviving preference, pick exactly one primary app driven by the per-app use_purposes and topical_focus. Maintain topical consistency within an app.

14. **Assign rows to apps** — session-level majority vote ensures all rows from the same browsing session land on the same app. Then apply **8% deterministic noise**: for each session, with probability 0.08 reassign the entire session to a random different app. Never route `implicit_negative` to Chatbot OR AI_Studio (they are forced to a social platform).

15. **Assign session locations** (R5) — mini-tier. One batched LLM call per user (`prompts.py::assign_session_locations_prompt`). For each session, emit `{city, region, country, lat, lon, precision}`. Constraints: infer home city from career/education/bio; assign ≥ `HOME_LOCATION_MIN_SHARE` (0.90) of sessions to home; at most `MAX_LOCATIONS_PER_USER` (3) distinct cities; travel appears as one contiguous 2–4-day block.

16. **Generate calendar modifications** (R5) — mini-tier. One batched LLM call per user (`prompts.py::generate_calendar_modifications_prompt`). Produce 5–10 CRUD modifications scattered at realistic timestamps across the observation window, split ~65% `added` / ~20% `updated` / ~15% `removed` (`CALENDAR_MOD_WEIGHTS`). Entries cover daily-life activities (work / personal / social / health), ~40% preference-linked + ~60% plausible-noise. Persisted at save time to `backend/{uid}/calendar.json`. The calendar state at any T is derived by folding modifications with `ts ≤ T`.

17. **Generate interaction formats** — mini-tier. Rules from `prompts.py::generate_interaction_format_prompt`. For each preference, pick exactly one action from `PLATFORM_INTERACTION_FORMATS[assigned_app][interaction_type]` verbatim (catalog-only, no new wording). Copy the matching `label` from the catalog as `action_label`. Generate a `user_message` ONLY when the chosen action is:
   - in `AT_AI_ACTIONS` (social-media `@ai` comments on IG / FB / Threads) — message starts with `@ai `, or
   - in `CHATBOT_TURN_ACTIONS` (natural chat turns on the Chatbot app) — message does NOT start with `@ai ` because the user is already talking to the AI.
   In both cases the message is first-person, ~15–35 words, grounded in the specific preference topic. The final `interaction_format` is a JSON object: `{"app": ..., "action": ..., "action_label": ..., "user_message": ... | null}`.

18. **Generate chatbot conversations** — rules from `chatbot_conversation.py` and `prompts.py::generate_chatbot_conversation_prompt`, `generate_ask_to_forget_conversation_prompt`, `generate_correction_conversation_prompt`. **Every** Chatbot-routed preference gets a conversation: ~80% get multi-turn (2–10 turns), ~20% get a minimal 2-turn exchange. The conversation type is selected from `CHATBOT_CONVERSATION_TYPES` based on the user's `chatbot_contexts`. For multi-turn `explicit_negative` chatbot preferences, ~70% get special 4-turn ask-to-forget or correction conversations. LLM calls are retried up to 2 times on failure. New fields added to chatbot records: `conversation` (array of `{role, content}`), `conversation_type`, `ask_to_forget` (bool).

18b. **Generate AI Studio conversations + Step Z audit** — sequential per-user walk of AI_Studio-routed events via `data_preparation/ai_studio_conversation.py` (cross-session memory, SPT pacing) followed by the `ai_studio_audit.py` quality audit; full rules in the "Design invariants & reference" section below (AI Studio bullets). The AI character comes from Step 11C.

19. **Generate synthetic per-event content** — mini-tier. Rules from `prompts.py::generate_synthetic_content_prompt`. Attach one `content_type` (`text` | `image` | `short_video`) + `content` payload to every non-Chatbot, non-stub event. Derive the per-user content-type mix per app from observed actions (`ACTION_CONTENT_HINTS`) with Bayesian smoothing against the platform prior (`PLATFORM_CONTENT_PRIOR`: IG 45/50/5, FB 35/30/35, Threads 30/20/50) plus a per-user lognormal perturbation (`CONTENT_MIX_NOISE_SIGMA = 0.3`). Resolve each event's `content_type` deterministically from the action when the hint is unambiguous (`viewed_reel_75` → short_video, `lingered_on_image` → image, etc.); otherwise sample from the user's mix with an `(user_id, oid)`-seeded RNG. Also pre-sample the event's action + itype using the same seed `save_to_backend` would use, so the final displayed action stays consistent with the generated content type. Chatbot events are skipped entirely (their `conversation` is the content). Implicit-negative stubs stay content-less (greyscale markers). Content schemas: `text` → `{text}`; `image` → `{caption, overall_description, parts[], metadata{camera, lens, filter, aspect_ratio, dimensions, iso, shutter, aperture, color_profile, location, time_of_day, filename}}`; `short_video` → `{title, caption, overall_description, key_frames[{timestamp_s, description}], audio_transcript, metadata{duration_s, resolution, fps, aspect_ratio, music_track, sound_design, codec, bitrate_kbps, creator_handle}}`. One LLM call per event, parallelized via ThreadPoolExecutor.

20. **Inject ad events** (R7) — mini-tier. Rules from `prompts.py::synthesize_ad_content_prompt`. Convert `AD_INJECTION_RATE` (0.06) of commerce-adjacent social-app events (hashtags hitting `HASHTAG_TO_AD_CATEGORY`) into sponsored ads. Polarity mix `AD_POLARITY_WEIGHTS` = 70% `clicked_ad` / 20% `dismissed_ad` / 10% `hidden_ad`. Content is regenerated with required `ad_metadata` block (`sponsor_name`, `ad_category`, `cta_label`, `cta_destination_kind`, `disclosure_label: "Sponsored"`). Invariant: `event.is_ad == true` iff `event.interaction_format.action ∈ AD_ACTIONS` (`clicked_ad`, `hidden_ad`, `dismissed_ad`). Never applied to Chatbot.

21. **Annotate stereotype marks** — mini-tier. Rules from `prompts.py::annotate_stereotype_prompt`. Demographics-only (gender, sexual orientation, race/ethnicity — NOT career/education). Most should be `neutral`. Be conservative.

21b. **Plant sensitive_life_event evidence rows** — rules from `prompts.py::generate_sensitive_event_evidence_rows_prompt`. After per-app event lists are built but BEFORE writing per-app JSONs, walk `user_profile.hidden_personas` for the synthetic `sensitive_life_event` cluster (Step 9b). For each `events[]` entry, mini-tier LLM call generates 2–4 implicit_positive engagement rows on a chosen social app (rotating across episodes for multi-event users). Rows carry `source_hashtags` from the episode's `evidence_hashtags` (≥ 2 must overlap; backfilled from the canonical list if the LLM drifts), an `interaction_format.action` sampled verbatim from `PLATFORM_INTERACTION_FORMATS[app]["implicit_positive"]`, LLM-written `content.title` + `content.caption`, empty `preferences[]`, and a `_planted_sensitive_event` topic tag for traceability. Timestamps are inside `[first_seen_ts, last_seen_ts]` (offsets emitted by the LLM, clamped). These planted rows are what gives the eval agent visible signal — without them the `over_personalization_sensitive_event` task has nothing to test restraint against because `profile.json` is firewalled from the agent in every eval mode.

22. **Save** — write the per-user files to `backend/{user_id}/` (profile + 5 app JSONs + calendar + ai_studio_memory; layout above):
    - `profile.json`: UserProfile dataclass + shared `user_voice` block + `app_personas` dict (one entry per app, each carrying audience/length/effort/topic + optional `overrides`) + hidden personas + MBTI
    - `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`: list of interaction events **sorted strictly by `source_timestamp` ascending**. Every event carries `event_location` from Step 15, `is_ad` + `content.ad_metadata` if promoted in Step 20, and every surviving preference carries `time_horizon` + optional `stop_condition` from Step 4. `implicit_negative` events whose canonicals survived the ≥5 gate are promoted to `source_interaction_type: "explicit_negative"` and carry full preferences. All other `implicit_negative` rows appear as stub events with empty `preferences: []` (no predictions needed). In `persona.html` these stubs render in full greyscale.
    - `calendar.json`: `{"modifications": [...]}` — the CRUD stream from Step 16.

23. **Extension B** (post-save) — self-posts, DM threads, friends graph. Adds `friends[]` to `profile.json`. Trending feed events (`is_trending=True`) are generated by `feed_posts.py` and embedded directly in the app JSONs. Idempotent.

24. **Proactive Trigger Candidate Inference** — rules in `persona_agent.py:infer_proactive_trigger_candidates` (`:8639`) and `prompts.py:infer_proactive_trigger_prompt` (`:3975`). After Extension B is done, look at the just-written files and catalogue moments where the agent could legitimately initiate contact, scored by an LLM against the JITAI 6-component framework (Nahum-Shani et al., 2018) and Horvitz mixed-initiative principles (CHI 1999). **This step must run for every user** — if it does not, the eval-side proactive task builders silently produce zero test questions.

    **Stage 1 — deterministic candidate gathering.** Open `backend/{uid}/{chatbot,instagram,facebook,threads}.json` plus the just-updated `profile.json`. Produce a dict `candidates_by_type` with two keys, using the helpers in `persona_agent.py`:

    - `close_friend_update` (helper `_gather_close_friend_dms` at `:8870`) — walk `profile.friends[]`. For each friend with `relationship_depth == "close"`, scan their self-posts. A candidate fires if the user did NOT engage (no save, like, comment, repost, view) with that post within 24 hours (`_PROACTIVE_DM_REPLY_WINDOW`). Set candidate `t_test = friend_post_ts + 1h`.
    - `sensitive_event_silence` (helper `_gather_sensitive_event_moments` at `:8948`) — walk `profile.hidden_personas` for clusters of type `sensitive_life_event`. For each cluster `events[]` entry, pick 3–5 timestamps inside the first 14 days after `first_seen_ts` (`_gather_sensitive_event_periods` at `:8761`). These mark moments where the agent must *stay silent*.

    **Stage 2 — LLM-judged eligibility.** Build a compact `user_state` header once via `_build_proactive_user_state_base` (`:8974`): user name, top-3 hidden persona labels, top-5 hashtags. Then for each candidate, set `user_state["sensitive_event_active"] = _is_in_sensitive_window(candidate.t_test, sensitive_periods)`, call `prompts.infer_proactive_trigger_prompt(user_state, candidate)`, and have the LLM return a JITAI card with `distal_outcome`, `proximal_outcome`, `tailoring_variable`, `decision_point`, `decision_rule_pass` (bool), `eligibility_score` (0–3), `recommended_action_class` (one of `follow_up` / `friend_alert` / `stay_silent`), `subtlety_check_pass` (bool), `reasoning`. Use `temperature=0.0` for reproducibility. The prompt body cites Horvitz + JITAI + the 7 subtlety constraints — do not paraphrase it; call it verbatim.

    **Acceptance rule** (`_proactive_candidate_passes` at `:9010`): proactive candidates (`close_friend_update`) survive iff `eligibility_score >= 2 AND subtlety_check_pass AND decision_rule_pass`. Restraint candidates (`sensitive_event_silence`) survive iff `eligibility_score == 0 AND recommended_action_class == "stay_silent"`. Attach the JITAI card to the candidate as `candidate.jitai_card` before keeping it.

    **Output.** Write `profile["proactive_trigger_candidates"] = {"close_friend_update": [...], "sensitive_event_silence": [...]}` back into `profile.json`. Even if both lists are empty, **write the empty dict** — that signals "Step 24 ran" vs "Step 24 was skipped." The eval-side builders at `evaluation/tasks/proactive_actions.py:_load_proactive_catalog` (`:84`) warn loudly when the key is missing.

(R8 removed the old `build_test_split` step entirely. Eval now picks its own test moments from the full timeline at any `T_test` cut — no pre-flagged train/test partition in the data.)

## Preference object shape (used in all per-app JSONs)

```json
{
  "persona_item": "Enjoys home cooking and preparing family meals",
  "category": "home cooking and food",
  "confidence_score_init": 0.85,
  "confidence_cross_referenced": 0.4,
  "relationship_type": "similar" | "contradictory" | "none",
  "related_personas": [{"persona_item": "...", "type": "similar"}],
  "stereotype_mark": "neutral",
  "split": "train" | "test",
  "over_personalization_irrelevant": "",
  "over_personalization_irrelevant_category": "",
  "source_interaction_type": "implicit_positive",
  "source_object_id": "691531",
  "source_timestamp": 1775235998,
  "formatted_timestamp": "17:06, 04/03/2026",
  "source_hashtags": ["#dessertlover", "#foodie"],
  "assigned_app": "Instagram",
  "interaction_format": {
    "app": "Instagram",
    "action": "saved_to_collection",
    "action_label": "Saved to a collection",
    "user_message": null
  }
}
```

For Chatbot preferences where the chosen action is in `AT_AI_ACTIONS`, `user_message` is a short first-person string like `"@ai can you show me more authentic Mexican home cooking recipes? Weeknight-friendly ones that still feel traditional."`

### Chatbot-specific fields (added by step 8.5)

Chatbot records gain three additional fields not present on social media app records:

```json
{
  "...all standard fields above...",
  "conversation_type": "knowledge_query",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ask_to_forget": false
}
```

- `conversation_type`: one of `writing_help`, `knowledge_query`, `therapy_reflection`, `troubleshooting`, `casual_chat`, `translation`, `health_consultation` — or `null` if no multi-turn conversation was generated (~20% of records).
- `conversation`: array of `{role, content}` turns, or `null`. Preferences are embedded implicitly — the user never directly states "I like/dislike X".
- `ask_to_forget`: `true` if the conversation ends with the user asking the chatbot to forget a personal detail.
- For ask-to-forget records: `interaction_format.action` = `"asked_to_forget"`.
- For correction records: `interaction_format.action` = `"corrected_assumption"`.

## Profile object shape (profile.json)

```json
{
  "user_id": "251",
  "name": "...",
  "gender": "...",
  "race_ethnicity": "...",
  "career": "...",
  "education": "...",
  "big_five": {"openness": "medium", "conscientiousness": "high", ...},
  "bio": "...",
  "user_voice": {
    "identity_spine": {
      "agency_communion": "...",
      "redemption_motifs": ["..."],
      "contamination_motifs": [],
      "life_stage_preoccupations": ["..."],
      "signature_concerns": ["..."],
      "liwc_anchors": {"analytic": "low", "clout": "low", "authentic": "high", "emotional_tone": "..."},
      "big_five_drivers": {"openness": "level → behavioral implication", ...}
    },
    "idiolect": {
      "function_word_profile": "...",
      "syntactic_preferences": {"sentence_length_shape": "short_dominant", "clause_embedding": "shallow", "parataxis_hypotaxis": "parataxis", "fragment_use": "frequent"},
      "hedge_booster_ratio": "hedge_dominant",
      "appraisal_fingerprint": {"attitude_dominant": "affect", "engagement_style": "heteroglossic_acknowledge", "graduation": "frequent_softeners"},
      "constructional_templates": [{"pattern": "[hedge] just [verb] ___", "example_realization": "kinda just want easy", "frequency": "common"}],
      "catchphrase_residue": []
    },
    "repertoire": {
      "stances": ["deadpan-affectionate", "irritable-pragmatic", "hype-mode"],
      "registers": ["notes-app casual", "polite-warm with elders"],
      "backstage_frontstage_range": "...",
      "speech_genre_fluency": ["live-game reaction", "casserole post", "venting DM"]
    },
    "natural_register": "...",
    "humor_tone": "...",
    "default_capitalization": "all_lowercase",
    "punctuation_habits": "...",
    "formality_baseline": 0.25,
    "emoji_palette": ["🥊", "🦬", "🍝"],
    "emoji_intensity_default": "medium",
    "voice_avoid": "...",
    "phrases_to_avoid": ["..."]
  },
  "app_personas": {
    "Instagram": {
      "app_name": "Instagram",
      "active_stances": ["..."],         // ⊆ user_voice.repertoire.stances
      "active_registers": ["..."],       // ⊆ user_voice.repertoire.registers
      "active_speech_genres": ["..."],   // ⊆ user_voice.repertoire.speech_genre_fluency
      "audience_type": "mixed",
      "audience_lens": "...",
      "audience_design_note": "addressee = ..., auditor = ..., overhearer = ...",
      "use_purposes": ["..."],
      "friend_zones": ["..."],
      "posting_frequency": "weekly",
      "topical_focus": ["..."],
      "chatbot_contexts": [],
      "surface": {"effort_level": "medium", "length_band": "70-150", "emoji_intensity_shift": 0, "audience_self_censoring": "...", "disclosure_depth": "medium"},
      "idiolect_overrides": {},
      "app_avoid": "...",
      "delta_summary": "1-sentence WHY this audience selects this stance subset"
    },
    "Facebook":  {...},
    "Threads":   {...},
    "Chatbot":   {"app_name": "Chatbot", ..., "chatbot_contexts": ["therapy_and_reflection", "knowledge_exploration"]}
  }
}
```

## Output consistency

Each subagent must follow the **exact same prompts** defined in `data_preparation/prompts.py` and produce output matching the **exact same dataclass schemas** in `data_preparation/persona_agent.py`. Do NOT paraphrase, simplify, or improvise the prompts — copy the rules, confidence calibration scale, output JSON format, thresholds, and filtering logic verbatim. This ensures fair apples-to-apples comparison with API mode.

## Important scale caveats

- Large users (3k–6k interaction rows) produce tens of thousands of atomic personas. The subagent MUST dedupe by lexical identity in Step 2 to collapse identical strings. The 0.75 init filter and the 7-day recency window on corroboration are the two main size levers. No semantic redundancy removal is applied — repeated real-world signals and `confidence_cross_referenced` (the filtered, recency-gated corroboration count) capture frequency.
- For very large users, batched per-preference prompts (like Step 8 generating interaction formats one at a time) are prohibitively expensive. The subagent should **batch** these inline — one LLM reasoning pass over all (persona, app) pairs at once, not one reasoning pass per preference.
- Chatbot-routed preferences that end up as `@ai` actions need a unique `user_message` each; these should be generated as a batch where the subagent reasons about all of them together rather than per-item.

## Design invariants & reference (canonical constants, gates, schemas)

These are the pipeline's load-bearing invariants. They apply identically to subagent mode and API mode; when a step above and this section disagree, this section wins.

- **Cross-referencing** only applies across different interaction rows (different source_object_id), never within the same row. Identical persona_items are merged into a canonical BEFORE cross-referencing.
- **`confidence_cross_referenced`** = weighted corroboration score starting at 1.0 (base). Each distinct corroborating explicit row adds 1.0, implicit adds 0.5. After the LLM cross-ref step discovers `similar`/`contradictory` relationships, similar canonicals' base scores are added and contradictory canonicals' scores are subtracted (clamped to ≥0).
- **Init filter**: `MIN_PERSONA_INIT_CONFIDENCE = 0.65`. Anything below 0.65 is dropped after cross-ref, regardless of cross_ref score or relationship type. This is the **survival** floor only — it is a DIFFERENT constant from `HIGH_CONFIDENCE_INIT_THRESHOLD` (0.75) below. They both read 0.75 before R18; the survival floor was lowered to 0.65 once the merge-time evidence-coherence gate (`_merge_coheres`) made extra richness safe, so histories carry ~2× the canonicals while eval-critical selection stays at the strict 0.75 bar.
- **Merge gate (R18)**: a `similar` cross-ref union only fires if the two canonicals share a concrete topical hashtag (generic/platform tags excluded) or content token, or one subsumes the other. Union-find is transitive, so without this an `A~B~C` chain fuses unrelated siblings and the representative fans out onto every member event.
- **High-confidence predicate** (for test-split + distractor eligibility): `init >= 0.75 AND cross_ref > canonical_xref_threshold(n_explicit_rows, n_implicit_rows)` (20.0 for all-explicit, 50.0 for all-implicit positives, interpolated by mix). Single source of truth is `is_high_confidence` in persona_agent.py.
- **Contradictory canonicals bypass the bottom-20% filter and xref-threshold floor.** The contradictory penalty was softened to 0.5 × other_base. Without these exemptions the LLM-discovered contradictions get zeroed out and killed, collapsing `total_contradictions` / `temporal_topics` to 0.
- Stereotype marks are based on demographics only (gender, sexual orientation, race/ethnicity) — not career or education.
- **Per-user voice + AppPersonas (one shared voice, four expressions)**: each user gets ONE shared `user_voice` block on `profile.json` (default capitalization, 5–12-emoji personal palette, 3–6 personal phrases that bleed cross-app, punctuation habits, register, humor tone, formality baseline, default emoji intensity) PLUS four `AppPersona` entries describing how that ONE voice gets *modulated* per app. AppPersona carries: `use_purposes`, `friend_zones`, `audience_type`, `audience_lens`, `style_description` (framed as a delta from base voice), `topical_focus`, `expression` (`effort_level`, `length_band`, `emoji_intensity_shift`, `emoji_topic_filter`, `audience_self_censoring`), and `overrides` (OPTIONAL — empty `{}` for most apps for most users; only populated when source samples show genuine code-switching). Chatbot also carries 2–3 `chatbot_contexts`. Real people have ONE writing voice — what changes per app is audience/length/effort/topic, NOT arbitrary stylistic re-tooling. The same shared voice drives every user-voiced simulation: self-posts (extension_b/self_posts.py), DMs (extension_b/dm_threads.py), chatbot turns (chatbot_conversation.py + four chatbot conversation prompts), and `@ai` comments (generate_interaction_format_prompt). Step 11 is one LLM call returning both `user_voice` and `app_personas` together with ~10 stratified raw source rows for grounding.
- **AI Studio (5th app) — full surface (milestones a/b/c/d)**: each user gets ONE `ai_studio_persona` block on `profile.json` describing the AI character that drives all AI turns on AI Studio (Meta-AI-Studio-style companion chat). Schema in `data_preparation/persona_agent.py::AIStudioPersona` — same 4-layer voice structure as `UserVoice` (identity_spine + idiolect + repertoire + soft holdovers + negatives), but built from the chosen archetype's character DNA (not the user's raw data). Picked from a 10-archetype catalog (`AI_STUDIO_ARCHETYPES`) grounded in Character.AI / Replika / Meta AI Studio usage patterns: `anime_or_fandom_character`, `late_night_best_friend`, `romantic_partner` (multi-axis sub-typed via `RomanticSpecifier` — gender / sexuality / aesthetic / body-role / relational-dynamic / explicitness-band), `older_sibling_figure`, `therapist_companion_reflective`, `mentor_coach`, `wise_elder_grandparent`, `niche_expert_creator_ai` (niche auto-derived from user's hashtag clusters), `hype_affirmation_friend`, `historical_or_philosophical_voice`. Generated by Step 11C (`PersonaAgent.generate_ai_studio_persona()`) via the `personalize_ai_studio_persona_prompt` mini-tier LLM call. The AI persona has its OWN voice (separate from `user_voice`); the user's voice still drives all user turns on AI Studio. Generation guardrails (no diagnosis / no medication advice / anti-sycophancy / honesty when asked / no real-public-figure impersonation) are descriptive only — AI Studio is a personalization surface, not a safety surface; safety is enforced as a generation floor only (the audit's `no_harmful_content` axis drops events; never evaluated as a research dimension).
- **5-app `PLATFORMS`** = `["Instagram", "Facebook", "Threads", "Chatbot", "AI_Studio"]`. Canonical-level quotas: Chatbot=0.27 (utility), AI_Studio=0.18 (companion chat carved out of the old 0.40 Chatbot share), social floor=0.17 each. Step 13's LLM router (`assign_personas_to_apps_prompt`) sees both Chatbot and AI_Studio surfaces and routes utility to Chatbot, hidden-persona-anchored material (identity, aspiration, parasocial, intimate-interest, emotional-pattern) to AI_Studio. Step 13's deterministic post-pass (`_quota_rebalance_apps`) carves AI_Studio share from Chatbot using `AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS` minus `WRITING_UTILITY_CATEGORY_KEYWORDS`. Step 14's session voting + 8% noise treats AI_Studio as a regular app; `implicit_negative` rows never route to Chatbot OR AI_Studio (extended firewall) — they're forced to a social platform. `romantic_partner` archetype auto-disables on high-acuity active `sensitive_life_event` (generation guard at Step 11C, falls back to `late_night_best_friend`).
- **AI Studio cross-session memory + SPT pacing (Step 18b — milestone c)**: `data_preparation/ai_studio_conversation.py` walks AI_Studio-routed events in chronological order, sequentially generating each conversation with the FULL prior history embedded in the prompt (asymmetric memory — see `data_preparation/ai_studio_memory.py`). Each event picks from 11 conversation types (casual_check_in / philosophical_chat / aspiration_dreaming / venting_session / identity_exploration / memory_callback / niche_skill_session / intimate_share / parasocial_riff / flirty_banter / intimate_romantic_session) gated by SPT stage (**Social Penetration Theory** — Altman & Taylor 1973; S1 orientation → S2 exploratory → S3 affective → S4 stable·intimate, derived from `intimacy_arc ∈ [0,1]`) + archetype. AI Studio events carry NO `content_type` / `content` body — they're pure conversation, identical in shape to Chatbot events (skipped at the Step 19 content-generation loop). **Per-user delta scaling** (`compute_delta_scale(n_total_events)` in `ai_studio_memory.py`) rescales the raw deltas so heavy users (200+ routed events) climb S1→S4 across their whole history instead of saturating at S4 within ~20 events. The AI character's voice + the user's user_voice drive their respective turns; user turns NEVER restate prior conversations (oblique references only); AI turns NEVER name hidden persona types/labels verbatim. Cross-session continuity is the load-bearing rule: the AI character in event N references content from events 0..N-1 because the prompt sees them. **Asymmetric memory exposure**: generation passes the FULL prior history (data quality); the eval-side context helper (`assemble_eval_context`) windows to last K_recent=3 verbatim + summary-only older (K_recent=2 for the cross-session memory recall task) so the model under test must actually carry info forward. Per-user state persisted at `backend/{uid}/ai_studio_memory.json` (sibling of calendar.json); episodic_memory_items + intimacy_arc + intimacy_stage_history + open_threads + last_persona_consistency_anchor.
- **AI Studio audit (Step Z — milestone d)**: `data_preparation/ai_studio_audit.py` samples 20% of events (min 5, max 40 per user) and grades each on 7 quality axes (user_voice_match ≥3, ai_persona_voice_match ≥3, obliqueness ≥4, no_fake_therapist_phrases ≥4, no_mid_emotional_lecture ≥4, cross_session_continuity ≥3, spt_pacing_smoothness ≥4) plus a binary `no_harmful_content` floor. Failures: events that fail the safety floor are DROPPED (never ship). Quality-only failures are tagged `audit_status: "graceful_degrade"` so downstream readers can skip. Mini-tier LLM for the audit (`audit_ai_studio_event_prompt`).
- **Negative preferences are cross-referenced** independently (within negative histories only), using the same merge → init filter → weighted corroboration → LLM cross-ref pipeline as positives. Negatives use `MIN_NEGATIVE_INIT_CONFIDENCE = 0.55` (the hashtag_to_persona prompt caps negative scores at 0.75, so the positive 0.75 bar zeroed every canonical) and `XREF_THRESHOLD_NEGATIVE = 5.0` (dedicated floor, NOT the positive 20/50 interpolated threshold — negatives are structurally 5-10× rarer than positives and most source CSVs have 0 explicit_negative rows). No bottom-20% filter is applied to negatives.
- **Implicit negatives are conditionally included**: `implicit_negative` rows are skipped in main persona inference. Instead, a weighted net-sentiment score is computed per hashtag: `net = neg×1.0 − expl_pos×2.0 − impl_pos×1.0`. A hashtag is "hot" only if `net >=` the user-adaptive threshold (`NEG_PROMOTION_RATIO ×` the user's total implicit_negative rows) AND negative rows span ≥ `MIN_TEMPORAL_DAYS` (3) distinct days. ONE LLM call per hot hashtag (representative row, single hashtag only); rows with ≥ 2 hot hashtags are promoted, others stay as stubs. Full original hashtags kept in `source_hashtags`. In the negative cross-ref init filter, a canonical supported only by implicit evidence must have ≥ `MIN_IMPLICIT_NEGATIVE_REPETITION` (15) distinct source rows to survive.
- **Implicit negative promotion**: Promoted events (with surviving preferences) are relabeled `explicit_negative` in the output. Non-promoted implicit_negative rows still appear in app JSONs as stub events with empty `preferences: []` and render in full greyscale in `persona.html`.
- **App routing is session-based**: source rows are grouped into temporal sessions (gap ≤ `SESSION_GAP_SECONDS` = 5 s = same browsing burst). The LLM routes canonicals to apps, then a majority-vote at the session level ensures all rows from the same scrolling session land on the same app. 8% noise is applied per-session. `_assign_interaction_format` (random platform picker) is deprecated.
- **Interaction formats** come from a predefined catalog (`PLATFORM_INTERACTION_FORMATS` in persona_agent.py) — single source of truth for `action` identifiers and `action_label` wording. The pipeline picks one entry verbatim per preference, never invents new wording. Facebook reactions (love/haha/wow/sad/care/angry), Instagram save-to-collection + DM-to-friend, Threads quote-repost, etc.
- **`@ai` comment actions live on SOCIAL APPS, not on the AI Chatbot.** `AT_AI_ACTIONS` (`at_ai_recommend_more`, `at_ai_stop_recommending`, etc.) model the user `@`-mentioning an in-feed AI in the *comment section* of a post on Instagram / Facebook / Threads — message starts with `@ai `. On the AI Chatbot, the user just chats naturally: `CHATBOT_TURN_ACTIONS` (`asked_followup`, `requested_more_detail`, etc.) carry a natural chat-turn `user_message` with NO `@ai` prefix (the user is already talking to the assistant).
- **No test-split label in data-gen output (R8).** `split` and `over_personalization_irrelevant` have been dropped from emitted preferences and `build_test_split` has been removed from the pipeline. The eval harness picks test moments dynamically from the full history by cutting at an arbitrary `T_test` (E2 at @ai-directive timestamps, E3/E4 at stratified days, E5 at short-term canonical mid-windows, etc.). Pre-flagging a train/test partition inside the data was redundant and constrained eval.
- **Output layout**: `backend/{user_id}/` subfolder per user, containing `profile.json` + one JSON per app (`instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`, **`ai_studio.json`**) + `ai_studio_memory.json` (generation-time cross-session state, not consumed by eval) + `calendar.json` (modification stream). Each app JSON is a list of **interaction events** sorted by `source_timestamp` ascending. Each event represents one content engagement (one source CSV row) and contains a nested `preferences` list of surviving inferred preferences. Events with zero surviving preferences are omitted (for AI_Studio events: also dropped if Step 18b's conversation generation failed or Step Z's `no_harmful_content` floor failed). The same canonical text naturally appears across multiple events (preserving real-world repetition). `profile.json` retains a flat unique preference list. Trending platform content is embedded directly in social app JSONs as `feed_visible` events with `is_trending=True` — no separate `trending.json` file.
- **Hidden personas (Step 7)**: after profile generation, cross-row hashtag patterns are clustered by the LLM to infer deeper motivational layers. 11 discovered types grounded in behavioral science: `personality_trait`, `aspiration`, `emotional_pattern`, `identity_anchor` (overt + covert signals), `intimate_interest` (specific objects/aesthetics), `intellectual_curiosity`, `private_hobby`, `parasocial_attachment` (≥15 rows with one figure), `compensatory_need` (privacy_ratio >0.7), `covert_concern` (specific worries/fears/pressures — health anxiety, financial stress, parenting worry, relationship insecurity, body-image pressure), `medical_aesthetic_concern` (active medication / aesthetic-medicine / hormone / weight-loss regimen). Each cluster validated: ≥ 40 rows, ≥ 3 days (waived for `intimate_interest` when evidence overlaps the pre-screened intimate hashtag set; reduced to ≥ 15 rows / ≥ 2 days for `medical_aesthetic_concern`). Every cluster also carries `first_seen_ts` / `last_seen_ts` derived from its evidence rows. Total discovered cluster count stays 8–15. Each **preference** in app JSONs carries `hidden_persona_labels` linking it to matching hidden personas via hashtag overlap. `hidden_persona_summary` saved in `profile.json`. Total persona-type count = **11 discovered + 1 synthetic = 12**.
- **Sensitive life events (Step 9b — synthetic, LLM-personalized injection)**: every user gets exactly ONE additional `sensitive_life_event` hidden persona bundling **1–3 episodes** drawn from `SENSITIVE_LIFE_EVENT_TOPIC_MENU` (15 topics — divorce, breakup, surgery, gender/sexuality exploration, parent conflict, miscarriage, job loss, addiction recovery, mental health diagnosis, custody dispute, fertility struggle, death in family, chronic illness diagnosis, abuse recovery, financial collapse). Distinct from `covert_concern` (ongoing worry) by being *episodic* with `[first_seen_ts, last_seen_ts]` + `active_window_end = last_seen_ts + 14 days`. A mini-tier LLM call (`personalize_sensitive_life_event_prompt`) picks topics that fit the user's profile + hidden personas + top hashtags, demands diversity across themes, and writes ALL text from scratch (`label_fragment`, `specific_situation`, `evidence_hashtags`, `exemplar_persona_items`). **No template fallback** — if the LLM call fails, the user simply gets no `sensitive_life_event` persona. Cluster carries `is_synthetic: true`, `privacy_ratio: 1.0`, and an `events: [...]` list with one entry per episode. `_privacy_flagged()` in the personalization rubric treats `sensitive_life_event` as a privacy-flagged type alongside `intimate_interest` / `covert_concern` / `compensatory_need` / `medical_aesthetic_concern`. The Step 21 audit (`audit_persona_safety`) explicitly skips synthetic clusters since they're gated by their own active_window, not by recent organic engagement.
- **Sensitive-event evidence-row planting (Step 21b — pipeline-side LLM call)**: because `profile.json` is firewalled from the eval agent in every mode (snapshot doesn't write it; MCP overlay strips `hidden_personas`; longctx renderers omit profile preface), the agent would never see the sensitive_life_event signal otherwise. So in `save_to_backend`, after per-app event lists are built, **2–4 LLM-generated implicit_positive engagement rows are planted per episode** on a chosen social app (rotating across episodes for multi-event users). Rows carry `source_hashtags` from the episode's `evidence_hashtags` (≥ 2 must overlap; backfilled if the LLM drifts), an `interaction_format.action` sampled verbatim from `PLATFORM_INTERACTION_FORMATS[app]["implicit_positive"]`, LLM-written `content.title` + `content.caption`, empty `preferences[]`, and `_planted_sensitive_event` topic tag for traceability. These are visible to the agent in time-masked snapshots/MCP feeds; the `over_personalization_sensitive_event` eval (in EVAL.md) tests whether the agent leans on them in response to a benign off-topic query.
- **No more overpersonalization holdout** — removed; underlying data stays, just no special label.
- **Input CSV schema** matches `facebook/gistbench` columns only: `interaction_type, user_id, object_id, interaction_time, object_text`. No `dataset`/`ds`.
- **Ad injection (Step 20)**: social-app events whose hashtags are commerce-adjacent (see `HASHTAG_TO_AD_CATEGORY` in persona_agent.py) are eligible for conversion into sponsored ads. `AD_INJECTION_RATE = 0.06` of eligible events become ads, split 70% `clicked_ad` (explicit_positive) / 20% `dismissed_ad` / 10% `hidden_ad` (both explicit_negative). The content is regenerated via `synthesize_ad_content_prompt` to carry an `ad_metadata` block with invented sponsor name, fixed-vocabulary `ad_category` (11 values), fixed `cta_label` (6 values), `cta_destination_kind` (5 values), and `disclosure_label: "Sponsored"`. Invariant: `event.is_ad == true` iff `event.interaction_format.action ∈ AD_ACTIONS` (`clicked_ad`, `hidden_ad`, `dismissed_ad`). Chatbot never carries ad actions.
- **Time horizon + stop conditions (Step 4)**: each surviving canonical carries `time_horizon ∈ {"short_term", "long_term"}` (default `long_term`). Short-term is gated by `(span_days/obs_window) ≤ 0.35 AND n_rows < 8 AND category ∈ SHORT_TERM_ALLOWED_CATEGORIES` (travel, event_prep, purchase_intent, how_to, medical_consultation, trip). Short-term uses `XREF_THRESHOLD_SHORT_TERM = 3.0` instead of the 20/50 long-term interpolation. A mini-tier LLM pass (Step 4) confirms or demotes short-term candidates and emits a structured `stop_condition: {type, description, expected_stop_ts}` (type ∈ event/date/mastery/relocation). The LLM cannot promote long→short (guards against weak signals bypassing the short-term floor). Emitted in app JSONs on every preference; `stop_condition` only present when horizon is short_term.
- **Cross-polarity contradiction gate (Step 7)**: positive and negative cross-ref pipelines are run independently but their outputs are now cross-checked. Candidate pos/neg pairs must share `HASHTAG_OVERLAP_MIN = 2` hashtags, then an LLM confirms semantic opposition. Confirmed contradictory pairs must pass the temporal-precedent rule: the later-emerging stance survives only if `MIN_STANCE_FLIP_PRIOR = 3` (long_term) or `MIN_STANCE_FLIP_PRIOR_SHORT = 1` (short_term) same-polarity rows preceded the first opposite-polarity row. Failed pairs drop the later canonical; the survivor's `update_history` gains a `"contradicted"` entry with `resolution: "suppressed_insufficient_precedent"`. Passed pairs add mutual `"contradicted"` entries with `resolution: "stance_shift_with_precedent"`. Fixes a bug where an explicit_negative event appeared 1h after an implicit_positive with no prior positive evidence.
- **Per-session geolocation (Step 15)** + **calendar modification stream (Step 16)**: each event carries `event_location: {city, region, country, lat, lon, precision}` shared by all rows in its session. A mini-tier LLM call per user assigns home + up to 2 travel cities with ≥90% home-share (8-day window means most users are home-only). `backend/{uid}/calendar.json` holds a `{"modifications": [...]}` stream of CRUD events (`added`/`updated`/`removed`) on calendar entries. 5–10 total entries per user, scattered at realistic timestamps. Entries are a mix of preference-linked (40%) and plausible-noise daily activities (60%) — dentist, haircut, work review — NOT always tied to social hashtags. The calendar state at any T is derived by folding modifications with `ts ≤ T`, so eval tasks can time-mask it the same way as events. Mini-tier is also used for Step 13 (app routing).
