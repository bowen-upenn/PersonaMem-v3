# Project Rules

## Documentation Map (how the .md files in this repo fit together)
- `CLAUDE.md` (this file) — session rules ONLY: high-level workflow, management policies, and critical reminders every Claude Code session must follow. Deep content lives in the files below; never duplicate it here.
- `README.md` — public-facing quickstart: what the benchmark is, setup, building personas, running evals. Keep concise and free of internal run history.
- `skill.md` — single source of truth for the persona pipeline: the step-by-step subagent specification + the "Design invariants & reference" section (canonical constants, gates, quotas, schemas). Read before any pipeline work. Default generation mode is Claude Code subagents (one per user, in parallel) — not the API.
- `DESIGN.md` — design rationale: WHY each pipeline mechanism exists. Update on any design change (see Design Document below).
- `EVAL.md` — evaluation methodology: task definitions, modes, scoring, judge protocol.
- `AUDIT.md` — data-quality audit methodology: known failure modes, detection methods, existing automated checks. Methodology only — never findings from a specific run.
- `results/RESULTS_SECTION.md` — paper results prose (kept local, not tracked in git).
- `docs/` — internal scratch notes; gitignored, never shipped.

## Auto-Commit
- After completing a key milestone, create a git commit with a descriptive message
- Do not batch multiple unrelated changes into one commit
- Key milestones include: new features, pipeline changes, bug fixes, data reprocessing with code changes
- Always push to the current branch after committing

## Regeneration & Evaluation
- **NEVER** start any of the following without explicitly asking the user first and receiving a clear "yes":
  - Persona pipeline (`scripts/run_persona_pipeline.py`)
  - Query regeneration (`scripts/prepare_eval_data.py`)
  - Evaluation runs (`evaluation/run_eval.py`)
- LLM calls are expensive and other sessions may be editing code concurrently. Even if the user says "regen" or "run eval" in the same message as other instructions, **ask before launching**.
- When running a regen (via `scripts/run_persona_pipeline.py` or similar), always redirect stdout + stderr to files under `/tmp/persona_regen/{user_id}.{stdout,stderr}` AND launch a tmux session named `persona{user_id}` that tails both so the user can watch progress live. tqdm progress bars write to stderr, so stderr is the stream that carries real-time per-row progress. Example:
  ```bash
  mkdir -p /tmp/persona_regen
  /usr/bin/time -v python scripts/run_persona_pipeline.py --user_id 115 --verbose \
      > /tmp/persona_regen/115.stdout 2> /tmp/persona_regen/115.stderr &
  tmux new-session -d -s persona115 -x 220 -y 50 \
      "tail -F /tmp/persona_regen/115.stdout /tmp/persona_regen/115.stderr"
  ```
  Then tell the user: `tmux attach -t persona115` (read-only: add `-r`). Detach with `Ctrl-b d`.

## Cleanup of Logs & Backups
- Regularly propose cleaning up unused/stale files that accumulate during work: regen/eval logs (e.g. under `/tmp/persona_regen/`, `/tmp/eval_regen/`), one-off backup directories (e.g. `backend/_v1_backup/`), scratch dumps, and superseded temp artifacts.
- **NEVER `rm` (or otherwise delete/overwrite) log or backup files without explicitly asking the user first and receiving a clear "yes".** List exactly what will be removed (paths + rough sizes) and why it is safe to delete, then wait for approval. This applies even when cleanup is mentioned in the same message as other instructions — ask before deleting.
- Stale logs are a known audit footgun (e.g. tracebacks in old `*_eval.stderr` files misread as live errors); flag them for cleanup rather than silently leaving or silently removing them.

## Shared Results HTML (single-writer lock)
- `results/aggregate/html/results_tables.html` is written by MANY scripts (`results/_scripts/render_final_tables.py`, the NIAH/memory section renderers, ad-hoc `patch_*.py`) and by MULTIPLE concurrent sessions. With no lock, two writers race: the file gets **duplicated** (whole document doubled) or a section renderer carries a **stale copy of another section forward** (this is how the Hidden-persona Accuracy row kept reverting to old values).
- **Whenever you update `results_tables.html`, acquire the single-writer lock first — only one session/script may edit it at a time.** Hold the lock across the entire read-modify-write, not just the write.
- Python: `from _htmllock import html_lock` then `with html_lock(): html = open(HTML).read(); ...; open(HTML, "w").write(html)`. The helper is `results/_scripts/_htmllock.py` (flock on `results/aggregate/html/.results_tables.lock`, 180s timeout then raises rather than risk an overwrite). `render_final_tables.py` and `mark_top2_bolds.py` already use it; any new writer MUST too.
- Bash one-off edits:
  ```bash
  exec 9>results/aggregate/html/.results_tables.lock
  flock -w 180 9 || { echo "results_tables.html locked by another session"; exit 1; }
  python3 results/_scripts/<writer>.py        # do the edit
  flock -u 9
  ```
- First three tables (Accuracy / Latency / Total tokens) bold the **top two** cells per row (Accuracy = 2 highest; Latency & tokens = 2 lowest). `mark_top2_bolds.py` re-applies this; keep it top-2, not top-1.

## Design Document
- Whenever you make changes to the pipeline design (new features, changed thresholds, altered logic, new steps, etc.), update `DESIGN.md` accordingly. Keep it clean and concise — match the existing style.

## Audit Document
- Whenever you fix a quality problem in generated data or in benchmark queries (e.g. silent task-type loss, un-substituted template placeholders, jargon leaks, unfair test pairs, missing length floors, GT-shape drift, etc.), update `AUDIT.md` so future audits know about the failure mode.
  - Add the symptom + its detection method (substring blocklist, programmatic check, sample-row reading) to the appropriate slice (A / B / C) or the cross-cutting mechanical scan.
  - If you added a new automated check inside the pipeline (validator, gate, post-process scrub), list it in the "Existing automated checks" tables with location + what it catches.
  - If you intentionally chose not to fix a finding (by-design behavior), add it to "Known false positives — do not flag" with a one-line reason.
- `AUDIT.md` is methodology only — it contains no findings from any specific audit run. Findings go in a separate report.

## Data Quality Iteration (standing instruction)
- Continuously improving generated-data quality is a standing user priority. REMEMBER every piece of user feedback about data quality — expectations, complaints, accepted fixes — and apply it across sessions (save non-obvious feedback to auto-memory so it persists).
- Whenever the user is not satisfied with data quality (or flags a suspect sample), iterate:
  1. **Diagnose** on concrete generated rows — read the actual backend data; never guess from code alone.
  2. **Fix** — prefer BOTH a surgical script (in `scripts/`) that patches already-shipped data AND a pipeline fix (`persona_agent.py` / `prompts.py` / `skill.md`) so the failure cannot recur. Ask before any regen or eval run per the rules above.
  3. **Iterate into `AUDIT.md`** — add the symptom + detection method (and any new automated check) per the Audit Document section, so future audits catch this failure mode automatically.
  4. **Sync the other docs** — update `DESIGN.md` if the design changed, and `skill.md` if constants/gates/schemas changed.
- Repeat the loop until the user confirms satisfaction; each round's failure mode must land in `AUDIT.md` before the loop closes.
