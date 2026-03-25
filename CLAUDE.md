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
- Confidence scores use the full 0.0-1.0 range. Filter threshold: init < 0.5 AND cross_ref == 0.0.
- Stereotype marks are based on demographics only (gender, sexual orientation, race/ethnicity) — not career or education.
- Output per user: _preferences.csv (filtered rows only) + _profile.csv (positive users only).
