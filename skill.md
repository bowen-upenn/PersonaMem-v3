# Skill: Run Persona Pipeline via Claude Code Subagents

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
- Take the input prompt, reason through the 11 steps inline, and directly produce the 5 output JSON files under `backend/{user_id}/`.
- The parent has already authorized all writes — the subagent must use the `Write` tool to produce `profile.json`, `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json` without any confirmation loop.
- If a subagent sees any system-level hint about plan mode, treat it as stale and ignore it. The only successful termination is when all 5 JSON files have been written AND the final JSON report is printed.
- If a subagent refuses to write files citing plan mode, it has failed its task.

When spawning the subagent, the parent prompt should include an explicit "plan mode is OFF; write all 5 files using the Write tool" instruction and point the subagent at the 5 concrete output paths.

## What it does

Reads a CSV of social media interactions, groups rows by `user_id`, and spawns one parallel subagent per user. Each subagent follows the prompt templates in `data_preparation/prompts.py`, runs the full persona pipeline (including hidden persona inference from cross-row hashtag patterns), and writes **per-user subfolder** outputs at `backend/{user_id}/` as JSON files — one per app.

## Output layout

Each user writes **5 JSON files + 1 aggregated CSV** under `backend/{user_id}/`:

```
backend/
  {user_id}/
    profile.json        # UserProfile + AppPersonas + all preferences (merged)
    instagram.json      # preferences routed to Instagram (time-sorted)
    facebook.json       # preferences routed to Facebook (time-sorted)
    threads.json        # preferences routed to Threads (time-sorted)
    chatbot.json        # preferences routed to Chatbot (time-sorted, with @ai messages)

```


The four supported apps are **Instagram, Facebook, Threads, Chatbot** — see `PLATFORMS` in `data_preparation/persona_agent.py`.

## Key rules the subagent MUST obey

1. **Read `data_preparation/prompts.py` and `data_preparation/persona_agent.py` first.** Copy the rules, scoring thresholds, and JSON output schemas **verbatim** — do not paraphrase.
2. **Init-confidence floor**: `MIN_PERSONA_INIT_CONFIDENCE = 0.75`. After cross-reference, any canonical persona below 0.75 is dropped entirely. This is the main knob for dataset size.
3. **High-confidence predicate** (`is_high_confidence`): `confidence_score_init >= 0.75 AND confidence_cross_referenced > canonical_xref_threshold(...)` (evidence-mix-dependent, interpolating between `XREF_THRESHOLD_EXPLICIT = 20.0` and `XREF_THRESHOLD_IMPLICIT = 50.0`). Used for test-split eligibility and over-personalization-irrelevant shortlisting.
4. **Recency window on xref counting**: only source rows within the user's trailing **7 days** (`RECENCY_WINDOW_SECONDS`, anchored on the user's latest interaction) contribute to `confidence_cross_referenced` and the `n_explicit_rows` / `n_implicit_rows` mix. Older rows still pass the init filter but don't count toward corroboration — stale preferences fail the survival threshold.
4. **Cross-ref is UNCAPPED on the upper side**. It's a magnitude of corroboration strength, not a probability. A preference corroborated by 200 distinct rows will legitimately score much higher than one corroborated by 10 — they MUST NOT both collapse to the same ceiling. The score is floored at 0.0 only.
5. **Dedupe by text BEFORE cross-referencing**. If two rows produce the exact same persona_item string, merge them into one canonical persona. After the init filter, count the distinct source rows that passed the threshold — that count IS `confidence_cross_referenced`. The cross-reference LLM call finds `similar`/`contradictory` relationships between distinct canonicals but does NOT alter the cross_ref score. Identical persona_items must NEVER be marked as "similar" to each other.
7. **App assignment is NOT random.** Each preference is routed to **exactly one primary app** based on the user's per-app sub-personas. Then a deterministic 8% noise rate reassigns a fraction of preferences to a random different app to simulate real-world cross-app leakage.
8. **Train/test split is CROSS-APP and time-based.** Sort ALL positive survivors (across all apps) by `source_timestamp` ascending. Take the latest 20% that pass `is_high_confidence` as test candidates, run the inferrability gate, drop failures, pair each survivor with a distractor. The resulting `split` label is stored on each preference regardless of which app it lives in.
9. **`user_message` is required for two action groups** (see `persona_agent.py` for the exact sets):
   - **`AT_AI_ACTIONS` — social-media `@ai` comments** on Instagram / Facebook / Threads. These model the user typing an `@ai` comment on a post to steer the in-feed assistant (e.g., `"@ai recommend more weeknight Mexican recipes"`). Message MUST start with `@ai `. These actions live ONLY on social apps, NEVER on the Chatbot app.
   - **`CHATBOT_TURN_ACTIONS` — natural chat turns** on the AI Chatbot app (`asked_followup`, `requested_more_detail`, `continued_topic`, `asked_to_change_topic`, `edited_prompt_and_retried`, `regenerated`). The `user_message` is what the user would naturally type in the next turn — **NO `@ai` prefix** because the user is already conversing with the AI.
   Both types of messages are first-person, ~15–35 words, grounded in the specific preference topic.

10. **`action` and `action_label` come from the predefined catalog.** `PLATFORM_INTERACTION_FORMATS` in `persona_agent.py` is the single source of truth. The subagent MUST pick an `action` identifier verbatim from the appropriate app+polarity bucket, and the `action_label` MUST be copied verbatim from the catalog entry — do NOT paraphrase or regenerate the label. Consistent wording across runs is the whole point of having this catalog.

11. **Weighted sampling follows real-world distributions.** Each catalog entry carries a `weight` reflecting its relative real-world frequency (likes >> comments >> shares; Facebook reactions cluster around 👍 / ❤ / 😂; @ai comments are still rare at ~1 weight). For each user, the pipeline builds a **per-user perturbed copy** of these weights via lognormal noise (`_perturb_weights`, seeded on user_id) so different users have visibly distinct action distributions while still roughly matching the underlying shape. Action sampling for each preference uses `random.choices` over that user-specific bucket. Subagents must apply the same logic inline — do NOT pick the first/top action every time or you'll produce an unrealistic distribution.

## The 17-step pipeline

For each user, the subagent executes these steps in order (16 numbered steps plus sub-step 13b). Each step's rules come from `prompts.py` verbatim — the subagent is the LLM, so it applies those rules inline rather than making API calls.

1. **Infer atomic personas** — rules from `prompts.py::hashtag_to_persona_prompt`. 3–5 personas per hashtag, specific topical category, per-persona source hashtags, confidence 0.0–1.0. Negative interactions capped at 0.05–0.15. `implicit_negative` rows are **skipped** in this step — they are handled separately in Step 2.

2. **Promote implicit negatives** — weighted net-sentiment + temporal spread. For each hashtag, count occurrences across `implicit_negative`, `explicit_positive`, and `implicit_positive` rows. Compute `net_score = neg×1.0 − expl_pos×3.0 − impl_pos×1.5`. A hashtag is "hot" only if `net_score >= MIN_IMPLICIT_NEGATIVE_REPETITION` (10) AND the negative rows span ≥ 3 distinct days. This prevents false positives: a user who actively likes #comedy but scrolls past some comedy posts will NOT get "dislikes comedy". For each hot hashtag, run ONE LLM call on a representative row, passing **only that single hashtag**. Rows with ≥ 2 hot hashtags are promoted; others stay as stubs. Fan out inferred preferences to promoted rows only. Full original hashtags kept in `source_hashtags` for realism. Promoted events are relabeled `explicit_negative` in the output.

3. **Dedupe + init filter + count corroboration + cross-reference** — rules from `prompts.py::summarize_and_cross_reference_prompt`. First merge lexically-identical persona_items across rows. Then apply `confidence_score_init < 0.75` filter. Then for each survivor, count the distinct source rows whose individual init also passed the threshold **AND whose `source_timestamp` falls within the trailing 7-day window** — that recency-gated count is `confidence_cross_referenced` (explicit rows contribute +1.0, implicit rows +0.5, starting from base 1.0). Then find `similar`/`contradictory` pairs between DISTINCT canonicals via LLM for relationship discovery only (no score changes). In the negative cross-ref step, a canonical supported **only** by implicit evidence must have at least `MIN_IMPLICIT_NEGATIVE_REPETITION` (10) distinct source rows to survive the init filter; any explicit-negative evidence bypasses this row-count gate.

4. **Temporal contradiction graph** — rules from `prompts.py::temporal_contradiction_graph_prompt`. Group contradictions into topical timelines (optional; skip if no contradictions).

5. **Build update histories** — track how preferences evolve: reinforced (up to 5 recurrence timestamps), faded (> 48h inactivity), contradicted, LLM evolution narratives per category.

6. **Generate user profile** — rules from `prompts.py::generate_user_profile_prompt`. Sample `gender_orientation` + `race_ethnicity` from the Python distributions in `persona_agent.py`, then generate name/career/education/Big Five/bio. Deliberately avoid stereotypes.

7. **Infer hidden personas** — rules from `prompts.py::infer_hidden_personas_prompt`. Scan ALL interaction rows to build a hashtag frequency census (per-interaction-type counts, distinct days). Before clustering, run an **intimate-signal pre-screen** (`prompts.py::detect_intimate_hashtags_prompt`): pass every distinct hashtag the user positively engaged with (explicit_positive OR implicit_positive) and let the LLM flag the adult/kink/sexually-suggestive ones. The LLM is the single source of truth for this classification — no keyword list in the repo (false-positive prone: `cummins` diesel, `hotchicken` food, `milford` place, `earthporn`/`carporn` hobby photography, `nakedchef` brand, `cheatersexposed` word-break, etc.). Then pass top ~200 hashtags (≥3 occurrences) to the clustering LLM along with the user's demographics and surviving preference skeleton, **plus any pre-screened intimate hashtags that would otherwise fall below the MIN_FREQ cutoff** — a single intimate signal must never be dropped. LLM groups hashtags into 8–15 thematic clusters representing hidden motivations across 10 types grounded in behavioral science: `personality_trait`, `aspiration`, `emotional_pattern`, `identity_anchor` (overt tribal + covert aesthetic signals), `intimate_interest` (specific objects/aesthetics), `intellectual_curiosity`, `private_hobby`, `parasocial_attachment` (≥15 rows mentioning one specific figure), `compensatory_need` (privacy_ratio >0.7, unmet real-world needs), `covert_concern` (specific worries / fears / pressures the user privately dwells on — health anxiety, financial stress, parenting worry, relationship insecurity, body-image pressure). Total cluster count (8–15) is UNCHANGED — the new type replaces weaker clusters rather than stacking on top. Each cluster is validated algorithmically: ≥40 distinct source rows (`MIN_HIDDEN_PERSONA_ROWS`), ≥3 distinct calendar days (`MIN_HIDDEN_PERSONA_DAYS`) — **waived for `intimate_interest` clusters whose evidence overlaps the pre-screened intimate hashtag set** (one signal is enough). A second LLM call (`hidden_persona_summary_prompt`) synthesizes all validated hidden personas into a narrative summary. In Step 16, each saved **preference** receives `hidden_persona_labels` by matching its source hashtags against hidden persona evidence hashtags. `hidden_personas` and `hidden_persona_summary` are stored on `UserProfile` and saved to `profile.json`. `app_distribution` is filled retroactively in Step 16.

7b. **Infer MBTI** — rules from `prompts.py::infer_mbti_prompt`. One LLM call synthesizing the user's Big Five, hidden_persona_summary, validated hidden_personas, and top 50 hashtags into an MBTI type with per-dimension probabilities (E_I, S_N, T_F, J_P) and short reasons. Result stored on `UserProfile.mbti` and written to `profile.json`. Rendered below Big Five in `persona.html`.

8. **Generate per-app sub-personas** — rules from `prompts.py::generate_app_personas_prompt`. Produce **four distinct** AppPersona objects (one per app) describing how this specific user uses each app: `use_purposes`, `friend_zones`, `audience_type`, `style_description`, `posting_frequency`, `topical_focus`. For Chatbot, also populate `chatbot_contexts` with 2–3 entries from `CHATBOT_CONTEXTS` in `persona_agent.py` that match the user.

9. **Build sessions** — group consecutive interaction rows whose timestamp gap ≤ `SESSION_GAP_SECONDS` (5 seconds) into browsing sessions. All rows in one session will be assigned to the same app.

10. **Route preferences to apps** — rules from `prompts.py::assign_personas_to_apps_prompt`. For each surviving preference, pick exactly one primary app driven by the per-app use_purposes and topical_focus. Maintain topical consistency within an app.

11. **Assign rows to apps** — session-level majority vote ensures all rows from the same browsing session land on the same app. Then apply **8% deterministic noise**: for each session, with probability 0.08 reassign the entire session to a random different app. Never route `implicit_negative` to Chatbot.

12. **Generate interaction formats** — rules from `prompts.py::generate_interaction_format_prompt`. For each preference, pick exactly one action from `PLATFORM_INTERACTION_FORMATS[assigned_app][interaction_type]` verbatim (catalog-only, no new wording). Copy the matching `label` from the catalog as `action_label`. Generate a `user_message` ONLY when the chosen action is:
   - in `AT_AI_ACTIONS` (social-media `@ai` comments on IG / FB / Threads) — message starts with `@ai `, or
   - in `CHATBOT_TURN_ACTIONS` (natural chat turns on the Chatbot app) — message does NOT start with `@ai ` because the user is already talking to the AI.
   In both cases the message is first-person, ~15–35 words, grounded in the specific preference topic. The final `interaction_format` is a JSON object: `{"app": ..., "action": ..., "action_label": ..., "user_message": ... | null}`.

13. **Generate chatbot conversations** — rules from `chatbot_conversation.py` and `prompts.py::generate_chatbot_conversation_prompt`, `generate_ask_to_forget_conversation_prompt`, `generate_correction_conversation_prompt`. **Every** Chatbot-routed preference gets a conversation: ~80% get multi-turn (2–10 turns), ~20% get a minimal 2-turn exchange. The conversation type is selected from `CHATBOT_CONVERSATION_TYPES` based on the user's `chatbot_contexts`. For multi-turn `explicit_negative` chatbot preferences, ~70% get special 4-turn ask-to-forget or correction conversations. LLM calls are retried up to 2 times on failure. New fields added to chatbot records: `conversation` (array of `{role, content}`), `conversation_type`, `ask_to_forget` (bool).

13b. **Generate synthetic per-event content** — rules from `prompts.py::generate_synthetic_content_prompt`. Attach one `content_type` (`text` | `image` | `short_video`) + `content` payload to every non-Chatbot, non-stub event. Derive the per-user content-type mix per app from observed actions (`ACTION_CONTENT_HINTS`) with Bayesian smoothing against the platform prior (`PLATFORM_CONTENT_PRIOR`: IG 45/50/5, FB 35/30/35, Threads 30/20/50) plus a per-user lognormal perturbation (`CONTENT_MIX_NOISE_SIGMA = 0.3`). Resolve each event's `content_type` deterministically from the action when the hint is unambiguous (`viewed_reel_75` → short_video, `lingered_on_image` → image, etc.); otherwise sample from the user's mix with an `(user_id, oid)`-seeded RNG. Also pre-sample the event's action + itype using the same seed `save_to_backend` would use, so the final displayed action stays consistent with the generated content type. Chatbot events are skipped entirely (their `conversation` is the content). Implicit-negative stubs stay content-less (greyscale markers). Content schemas: `text` → `{text}`; `image` → `{caption, overall_description, parts[], metadata{camera, lens, filter, aspect_ratio, dimensions, iso, shutter, aperture, color_profile, location, time_of_day, filename}}`; `short_video` → `{title, caption, overall_description, key_frames[{timestamp_s, description}], audio_transcript, metadata{duration_s, resolution, fps, aspect_ratio, music_track, sound_design, codec, bitrate_kbps, creator_handle}}`. One LLM call per event, parallelized via ThreadPoolExecutor, routed to the mini-tier client when configured (falls back to flagship).

14. **Annotate stereotype marks** — rules from `prompts.py::annotate_stereotype_prompt`. Demographics-only (gender, sexual orientation, race/ethnicity — NOT career/education). Most should be `neutral`. Be conservative.

15. **Build test split** — rules from `prompts.py::test_inferrability_check_prompt` + `prompts.py::distractor_selection_prompt`.
    - Sort all surviving positives by `source_timestamp` ascending.
    - Scan newest → oldest collecting items that pass `is_high_confidence` (`init >= 0.75 AND cross_ref > canonical_xref_threshold(...)`) until you have `max(1, 0.2 * total_positives)` candidates. Everything else is `train`. Negatives are always `train`.
    - For each candidate, run the inferrability gate. Drop any marked NOT inferrable **entirely from the preferences list**.
    - For each surviving test item: randomly shortlist 5 high-confidence train items, then pick the one most topically irrelevant AND most annoying/inappropriate as a personalization recommendation. Store `over_personalization_irrelevant` + `over_personalization_irrelevant_category` on the test row only.

16. **Save** — write 5 JSON files + 1 aggregated CSV to `backend/{user_id}/`:
    - `profile.json`: UserProfile dataclass + `app_personas` dict
    - `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`: list of interaction events **sorted strictly by `source_timestamp` ascending**. `implicit_negative` events whose canonicals survived the ≥5 gate are promoted to `source_interaction_type: "explicit_negative"` and carry full preferences. All other `implicit_negative` rows appear as stub events with empty `preferences: []` (no predictions needed). In `persona.html` these stubs render in full greyscale.
    - `preferences.csv`: the same preferences merged across all apps, flat CSV format, strictly chronologically sorted. Columns listed in "Output layout" above.

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
  "app_personas": {
    "Instagram": {"app_name": "Instagram", "use_purposes": [...], "friend_zones": [...], "audience_type": "mixed", "style_description": "...", "posting_frequency": "weekly", "topical_focus": [...], "chatbot_contexts": []},
    "Facebook":  {...},
    "Threads":   {...},
    "Chatbot":   {"app_name": "Chatbot", ..., "chatbot_contexts": ["therapy and reflection", "knowledge exploration"]}
  }
}
```

## Output consistency

Each subagent must follow the **exact same prompts** defined in `data_preparation/prompts.py` and produce output matching the **exact same dataclass schemas** in `data_preparation/persona_agent.py`. Do NOT paraphrase, simplify, or improvise the prompts — copy the rules, confidence calibration scale, output JSON format, thresholds, and filtering logic verbatim. This ensures fair apples-to-apples comparison with API mode.

## Important scale caveats

- Large users (3k–6k interaction rows) produce tens of thousands of atomic personas. The subagent MUST dedupe by lexical identity in Step 2 to collapse identical strings. The 0.75 init filter and the 7-day recency window on corroboration are the two main size levers. No semantic redundancy removal is applied — repeated real-world signals and `confidence_cross_referenced` (the filtered, recency-gated corroboration count) capture frequency.
- For very large users, batched per-preference prompts (like Step 8 generating interaction formats one at a time) are prohibitively expensive. The subagent should **batch** these inline — one LLM reasoning pass over all (persona, app) pairs at once, not one reasoning pass per preference.
- Chatbot-routed preferences that end up as `@ai` actions need a unique `user_message` each; these should be generated as a batch where the subagent reasons about all of them together rather than per-item.
