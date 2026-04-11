# Project Rules

## Auto-Commit
- After completing a key milestone, create a git commit with a descriptive message
- Do not batch multiple unrelated changes into one commit
- Key milestones include: new features, pipeline changes, bug fixes, data reprocessing with code changes
- Always push to the current branch after committing

## Persona Pipeline
- Default mode is Claude Code subagents (not API). See skill.md for the full 11-step specification.
- When asked to "reprocess persona data", spawn one subagent per user in parallel.
- **Cross-referencing** only applies across different interaction rows (different source_object_id), never within the same row. Identical persona_items are merged into a canonical with `corroboration_count` BEFORE cross-referencing — they must NEVER be marked as "similar" to themselves.
- Scoring: `+0.1` per similar pair (both sides), `-0.1` per contradictory pair (older side only). Floor `0.0`, **cap `1.0`**. Base cross_ref starts at `0.1 * (corroboration_count - 1)` from the lexical merge.
- **Strict init filter**: `MIN_PERSONA_INIT_CONFIDENCE = 0.8`. Anything below 0.8 is dropped after cross-ref, regardless of cross_ref score or relationship type.
- **Semantic redundancy removal** runs after the 0.8 filter: LLM-clusters same-meaning-different-wording preferences; keeps the highest-scored representative per cluster; drops the rest.
- **High-confidence predicate** (for test-split + distractor eligibility): `init >= 0.8 AND cross_ref > 0.5`. Single source of truth is `is_high_confidence` in persona_agent.py. Thresholds are tentative; retune empirically once real-scale stats land.
- Stereotype marks are based on demographics only (gender, sexual orientation, race/ethnicity) — not career or education.
- **Per-user AppPersonas**: each user gets four distinct sub-personas (one per app — Instagram, Facebook, Threads, Chatbot) describing their use purposes, friend zones, audience type, style, posting frequency, and topical focus. Chatbot also carries 2–3 `chatbot_contexts`.
- **App routing is NOT random**: each surviving preference is assigned to exactly one primary app via LLM, driven by the per-app sub-personas. A deterministic 8% noise rate simulates cross-app leakage. `_assign_interaction_format` (random platform picker) is deprecated.
- **Interaction formats** are now richer and app-specific: Facebook reactions (love/haha/wow/sad/care/angry), Instagram save-to-collection + DM-to-friend, Threads quote-repost, Chatbot `@ai` steering directives (`at_ai_recommend_more`, `at_ai_stop_recommending`, etc.). For `@ai` actions, the pipeline generates a natural-language `user_message` grounded in the specific preference.
- **Train/test split is cross-app and time-based**: the latest 20% of each user's high-confidence positives (globally, regardless of app) are test candidates. LLM inferrability gate removes candidates that can't be predicted from the earlier history. Each surviving test item is paired with one LLM-picked hard-negative distractor drawn from a Python-shortlisted 5 high-confidence train items.
- **Output layout**: `backend/{user_id}/` subfolder per user, containing `profile.json` + one JSON per app (`instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`) + an aggregated `preferences.csv` that merges all apps into one flat time-sorted file (old-style view, kept for downstream tools). Each app JSON and the CSV are sorted strictly by `source_timestamp` ascending.
- **No more overpersonalization holdout** — removed; underlying data stays, just no special label.
- **Input CSV schema** matches `facebook/gistbench` columns only: `interaction_type, user_id, object_id, interaction_time, object_text`. No `dataset`/`ds`.
