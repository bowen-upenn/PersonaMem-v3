# Project Rules

## Auto-Commit
- After completing a key milestone, create a git commit with a descriptive message
- Do not batch multiple unrelated changes into one commit
- Key milestones include: new features, pipeline changes, bug fixes, data reprocessing with code changes
- Always push to the current branch after committing

## Persona Pipeline
- Default mode is Claude Code subagents (not API). See skill.md for the full 11-step specification.
- When asked to "reprocess persona data", spawn one subagent per user in parallel.
- **Cross-referencing** only applies across different interaction rows (different source_object_id), never within the same row. Identical persona_items are merged into a canonical BEFORE cross-referencing.
- **`confidence_cross_referenced`** = count of distinct source rows that independently produced this canonical AND individually passed `MIN_PERSONA_INIT_CONFIDENCE`. Computed AFTER the init filter. The LLM cross-ref step discovers `similar`/`contradictory` relationships but does NOT change this score.
- **Init filter**: `MIN_PERSONA_INIT_CONFIDENCE = 0.5`. Anything below 0.5 is dropped after cross-ref, regardless of cross_ref score or relationship type.
- **Semantic redundancy removal** runs after the 0.5 filter: LLM-clusters same-meaning-different-wording preferences; keeps the highest-scored representative per cluster; drops the rest.
- **High-confidence predicate** (for test-split + distractor eligibility): `init >= 0.5 AND cross_ref > 0.5`. Single source of truth is `is_high_confidence` in persona_agent.py. Thresholds are tentative; retune empirically once real-scale stats land.
- Stereotype marks are based on demographics only (gender, sexual orientation, race/ethnicity) — not career or education.
- **Per-user AppPersonas**: each user gets four distinct sub-personas (one per app — Instagram, Facebook, Threads, Chatbot) describing their use purposes, friend zones, audience type, style, posting frequency, and topical focus. Chatbot also carries 2–3 `chatbot_contexts`.
- **App routing is NOT random**: each surviving preference is assigned to exactly one primary app via LLM, driven by the per-app sub-personas. A deterministic 8% noise rate simulates cross-app leakage. `_assign_interaction_format` (random platform picker) is deprecated.
- **Interaction formats** come from a predefined catalog (`PLATFORM_INTERACTION_FORMATS` in persona_agent.py) — single source of truth for `action` identifiers and `action_label` wording. The pipeline picks one entry verbatim per preference, never invents new wording. Facebook reactions (love/haha/wow/sad/care/angry), Instagram save-to-collection + DM-to-friend, Threads quote-repost, etc.
- **`@ai` comment actions live on SOCIAL APPS, not on the AI Chatbot.** `AT_AI_ACTIONS` (`at_ai_recommend_more`, `at_ai_stop_recommending`, etc.) model the user `@`-mentioning an in-feed AI in the *comment section* of a post on Instagram / Facebook / Threads — message starts with `@ai `. On the AI Chatbot, the user just chats naturally: `CHATBOT_TURN_ACTIONS` (`asked_followup`, `requested_more_detail`, etc.) carry a natural chat-turn `user_message` with NO `@ai` prefix (the user is already talking to the assistant).
- **Train/test split is cross-app and time-based**: the latest 20% of each user's high-confidence positives (globally, regardless of app) are test candidates. LLM inferrability gate removes candidates that can't be predicted from the earlier history. Each surviving test item is paired with one LLM-picked `over_personalization_irrelevant` item (a correct-but-topically-irrelevant preference) drawn from a Python-shortlisted 5 high-confidence train items.
- **Output layout**: `backend/{user_id}/` subfolder per user, containing `profile.json` + one JSON per app (`instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`) + an aggregated `preferences.csv` that merges all apps into one flat time-sorted file (old-style view, kept for downstream tools). Each app JSON and the CSV are sorted strictly by `source_timestamp` ascending.
- **No more overpersonalization holdout** — removed; underlying data stays, just no special label.
- **Input CSV schema** matches `facebook/gistbench` columns only: `interaction_type, user_id, object_id, interaction_time, object_text`. No `dataset`/`ds`.
