# Project Rules

## Auto-Commit
- After completing a key milestone, create a git commit with a descriptive message
- Do not batch multiple unrelated changes into one commit
- Key milestones include: new features, pipeline changes, bug fixes, data reprocessing with code changes
- Always push to the current branch after committing

## Persona Pipeline
- Default mode is Claude Code subagents (not API). See skill.md for the full specification.
- When asked to "reprocess persona data", spawn one subagent per user in parallel.
- Cross-referencing only applies across different interaction rows (different source_object_id), never within the same row.
- Confidence scores use the full 0.0-1.0 range. Filter threshold: init < 0.5 AND cross_ref <= 0.0.
- **High-confidence predicate** (for test split + distractor eligibility): init >= 0.5 AND cross_ref > 0.5. Defined once as `is_high_confidence` in persona_agent.py. Thresholds are tentative; retune empirically.
- Stereotype marks are based on demographics only (gender, sexual orientation, race/ethnicity) — not career or education.
- **Train/test split**: every preference carries a `split` label. The latest 20% of each user's *high-confidence* positives go through an LLM inferrability gate (can they be inferred from the earlier 80%?). Items that fail the gate are removed entirely. Each surviving test item is paired with one LLM-picked hard-negative distractor drawn from a Python-shortlisted 5 high-confidence train items.
- **No more overpersonalization holdout** — that mechanism has been removed; the underlying preferences stay in output, they just lost their special label.
- Output per user: `_preferences.csv` (all filtered rows, sorted early → latest, with `split` + `distractor_persona_item` + `distractor_category` columns) + `_profile.csv` (all users).
- CSV schema matches `facebook/gistbench` columns only: `interaction_type, user_id, object_id, interaction_time, object_text`. No `dataset`/`ds`.
