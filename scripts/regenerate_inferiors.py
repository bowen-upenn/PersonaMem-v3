"""Re-emit only the `inferior_response` field on broken (example, inferior)
pairs in an already-built benchmark.

Why this exists: when only the inferior generator changes (new prompts,
new validator), there's no need to re-run the expensive example-generation
LLM call for every sample. This script:

  - loads `benchmark/{user_id}/queries.csv` (canonical store)
  - reuses each instance's existing example_response, user_query, and the
    saved `inferior_response.flaw_kind` / `flaw_evidence`
  - for ranking tasks: recomputes inferior deterministically (no LLM call)
  - for non-ranking broken pairs: calls `_generate_inferior` with the
    new prompt + validator + retry loop
  - leaves already-passing pairs alone
  - writes back to queries.csv and re-dumps `backend/{user_id}/test.json`

Usage:
    python scripts/regenerate_inferiors.py 115 [--llm-model haiku]
                                                 [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
csv.field_size_limit(sys.maxsize)

from evaluation.llm_postprocess import (
    _RANKING_TASKS,
    _TASKS_NO_FOIL,
    _compute_ranking_inferior,
    _generate_inferior,
    _validate_inferior,
)
from evaluation.task_registry import normalize_task_type
from scripts.prepare_eval_data import (
    COLUMNS,
    QUERIES_CSV_VERSION,
    _make_blind_check_llm,
    _project_row,
)


def _load_queries_csv(user_id: str) -> tuple[list[dict], list[dict]]:
    """Returns (rows_in_order, instances_in_order). Rows preserve the
    on-disk CSV row dicts so we can rewrite them with minimal churn."""
    path = Path("benchmark") / user_id / "queries.csv"
    if not path.exists():
        raise FileNotFoundError(f"no queries.csv at {path}")
    rows: list[dict] = []
    instances: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            rows.append(row)
            inst = json.loads(row["instance_json"])
            inst["task_id"] = normalize_task_type(inst.get("task_id") or row.get("task_type", ""))
            instances.append(inst)
    return rows, instances


def _write_queries_csv(user_id: str, rows: list[dict], instances: list[dict]) -> Path:
    """Rewrite queries.csv with updated instance_json. Preserves seq order
    and other columns from the original rows."""
    path = Path("benchmark") / user_id / "queries.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# queries_csv_version={QUERIES_CSV_VERSION}\n")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row, inst in zip(rows, instances):
            seq = int(row.get("seq", 0))
            tt = row.get("task_type", "")
            ts = int(row.get("ts") or 0)
            writer.writerow(_project_row(seq, tt, inst, user_id, ts))
    return path


def _is_ranking(task_id: str) -> bool:
    return normalize_task_type(task_id) in _RANKING_TASKS


def _is_no_foil(task_id: str) -> bool:
    return normalize_task_type(task_id) in _TASKS_NO_FOIL


def _classify_pair(example: str, inferior: str) -> tuple[bool, str]:
    return _validate_inferior(example, inferior)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id")
    parser.add_argument("--llm-model", default="haiku",
                        help="model for inferior LLM rewrites (default: haiku)")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write changes; just report what would happen")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many LLM regen attempts (0 = no limit). "
                             "Ranking recomputations are not affected by --limit.")
    args = parser.parse_args()
    user_id = args.user_id

    rows, instances = _load_queries_csv(user_id)
    print(f"[{user_id}] loaded {len(instances)} instances from queries.csv")

    # Stats.
    n_skipped_good = 0
    n_skipped_no_foil = 0
    n_skipped_no_pair = 0
    n_ranking_recomputed = 0
    n_llm_regen_attempted = 0
    n_llm_regen_succeeded = 0
    n_llm_regen_dropped = 0

    llm = None  # lazy: only build if any non-ranking sample needs it

    for inst in instances:
        task_id = normalize_task_type(inst.get("task_id") or inst.get("task_type", ""))
        if _is_no_foil(task_id):
            n_skipped_no_foil += 1
            continue
        example = inst.get("example_response") or ""
        inf_obj = inst.get("inferior_response") or {}
        inf_text = inf_obj.get("text") if isinstance(inf_obj, dict) else inf_obj
        inf_text = inf_text or ""
        if not example:
            n_skipped_no_pair += 1
            continue

        # Already-passing pair → leave alone.
        if inf_text:
            ok, _ = _classify_pair(example, inf_text)
            if ok:
                n_skipped_good += 1
                continue

        # Broken or missing inferior — regenerate.
        if _is_ranking(task_id):
            new_text = _compute_ranking_inferior(inst, task_id)
            # Edge case: when the instance has no personalization signal
            # (e.g. at_ai_directive_followup with empty positive_indices
            # AND empty carveout_indices), the inversion collapses to the
            # same order as the example. Drop the foil rather than emit a
            # meaningless identical pair.
            if new_text and new_text != example:
                inst["inferior_response"] = {
                    "text": new_text,
                    "flaw_kind": "ranking_inversion",
                    "flaw_evidence": {"_from": "deterministic_ranking_inversion"},
                }
                inst.pop("inferior_drop_reason", None)
                n_ranking_recomputed += 1
            else:
                inst["inferior_response"] = None
                inst["inferior_drop_reason"] = "ranking_inversion_collapsed_to_gold"
                n_llm_regen_dropped += 1
            continue

        # Non-ranking → LLM rewrite. Reuse the saved flaw_kind/evidence
        # if present; otherwise we cannot regen (need flaw context).
        flaw_kind = (inf_obj.get("flaw_kind") if isinstance(inf_obj, dict) else "") or ""
        flaw_evidence = (inf_obj.get("flaw_evidence") if isinstance(inf_obj, dict) else None)
        if not flaw_kind or not flaw_evidence:
            n_skipped_no_pair += 1
            continue

        if llm is None and not args.dry_run:
            llm = _make_blind_check_llm(args.llm_model)
        if args.dry_run:
            n_llm_regen_attempted += 1
            continue

        if args.limit > 0 and n_llm_regen_attempted >= args.limit:
            # Hit the user-supplied LLM-call cap. Leave the rest for a
            # later run. We don't break out of the loop because we still
            # want to recompute ranking-task pairs (free) and tally
            # skipped buckets accurately.
            continue

        user_query = inst.get("user_query") or inst.get("query") or ""
        n_llm_regen_attempted += 1
        new_text = _generate_inferior(
            llm, example, flaw_kind, flaw_evidence, task_id,
            user_query=user_query,
        )
        if new_text:
            inst["inferior_response"] = {
                "text": new_text,
                "flaw_kind": flaw_kind,
                "flaw_evidence": flaw_evidence,
            }
            inst.pop("inferior_drop_reason", None)
            n_llm_regen_succeeded += 1
        else:
            # Validator rejected all 3 attempts. Drop the foil.
            inst["inferior_response"] = None
            inst["inferior_drop_reason"] = "validator_failed_after_3_attempts"
            n_llm_regen_dropped += 1

    print()
    print(f"[{user_id}] regen summary:")
    print(f"  already-good pairs (left alone):    {n_skipped_good}")
    print(f"  no-foil tasks (left alone):         {n_skipped_no_foil}")
    print(f"  missing example or flaw evidence:   {n_skipped_no_pair}")
    print(f"  ranking pairs recomputed:           {n_ranking_recomputed}")
    print(f"  LLM regen attempted:                {n_llm_regen_attempted}")
    print(f"    succeeded:                        {n_llm_regen_succeeded}")
    print(f"    dropped (validator gave up):      {n_llm_regen_dropped}")

    if args.dry_run:
        print()
        print("(dry-run — no files written)")
        return 0

    out = _write_queries_csv(user_id, rows, instances)
    print(f"[{user_id}] wrote {out}")

    # Re-dump test.json so the rendered output reflects the regenerated
    # inferiors. Mirrors what `repostprocess_user.py` does on completion.
    try:
        from data_preparation.visualize import dump_test_samples_json
        print(f"[{user_id}] {dump_test_samples_json(user_id)}")
    except Exception as exc:
        print(f"[{user_id}] dump_test_samples_json failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
