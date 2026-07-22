#!/usr/bin/env python3
"""Surgical add of the 3 previously-zero-emit task types to test.json.

short_vs_long_term_lifecycle, new_suggestions_recsys, new_suggestions_chatbot
emitted ZERO instances for every persona because of two builder bugs (fixed in
the preceding commit). This script rebuilds ONLY those three task types for the
given personas and APPENDS them to each user's backend/{uid}/test.json WITHOUT
touching any other row — every existing query_id is preserved so run_eval
--resume / prior results CSVs stay valid.

Formatting goes through the SAME canonical path the full builder uses
(`_project_row` -> `dump_test_samples_json(precomputed_rows=...)`), so the
appended records are schema-identical to a normal build. The merge is
idempotent: any pre-existing rows of these three task types are dropped before
re-appending.

c1e (new_suggestions) needs the discovery LLM (Azure gpt-5.5, the default
BUILDER_LLM_MODEL); e5 (short_vs_long) is LLM-free.

Usage:
  python scripts/add_missing_tasks_surgical.py --users "1 2 3 5 6 8 9 10 13 14" [--apply]
Without --apply: DRY (builds everything incl. LLM, reports counts, writes nothing).
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from evaluation.backend_query import BackendQuery                       # noqa: E402
from evaluation.inference_utils import load_test_items                  # noqa: E402
from evaluation.build_benchmark import build_c1e_new_suggestions       # noqa: E402
from evaluation.tasks.e5_horizon_lifecycle import (                     # noqa: E402
    build_e5_horizon_lifecycle,
)
from evaluation import task_distribution as _task_dist                  # noqa: E402
from data_preparation.visualize import dump_test_samples_json           # noqa: E402

# Import prepare_eval_data as a module to reuse its row helpers verbatim.
_spec = importlib.util.spec_from_file_location(
    "prepare_eval_data", "scripts/prepare_eval_data.py")
ped = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ped)

TASK_TYPES = (
    "short_vs_long_term_lifecycle",
    "new_suggestions_recsys",
    "new_suggestions_chatbot",
)
BACKEND = "backend"


def build_user(uid: str, llm) -> dict:
    """Build the 3 task buckets for one persona (capped per task_distribution)."""
    bq = BackendQuery(BACKEND)
    test_items = load_test_items(BACKEND, uid)
    e5 = build_e5_horizon_lifecycle(bq, uid, rng_seed=0)
    c1e = build_c1e_new_suggestions(bq, uid, test_items, discovery_llm=llm, rng_seed=0)
    buckets = {
        "short_vs_long_term_lifecycle": e5,
        "new_suggestions_recsys": c1e.get("new_suggestions_recsys", []),
        "new_suggestions_chatbot": c1e.get("new_suggestions_chatbot", []),
    }
    return _task_dist.apply_caps(buckets, rng_seed=0)


def project_rows(uid: str, buckets: dict) -> list[dict]:
    """Flatten buckets -> _project_row dicts, with :append: query_ids."""
    rows: list[dict] = []
    seq = 0
    for task_type, bucket in buckets.items():
        if not isinstance(bucket, list):
            continue
        for inst in bucket:
            if not isinstance(inst, dict):
                continue
            ts = ped._extract_ts(inst)
            if ts <= 0:
                ts = ped._synthesize_ts_for_no_ts_instance(inst, task_type, buckets, uid)
            if ts <= 0:
                print(f"    [{uid}] skip {task_type} (no timestamp): "
                      f"{inst.get('instance_id', '?')}")
                continue
            row = ped._project_row(seq, task_type, inst, uid, ts)
            iid = row["instance_id"]
            row["query_id"] = f"{uid}:append:{iid}"  # surgical-append marker
            rows.append(row)
            seq += 1
    return rows


def merge_into_test_json(uid: str, new_records: list[dict], apply: bool) -> str:
    path = Path(BACKEND) / uid / "test.json"
    existing = json.loads(path.read_text())
    kept = [r for r in existing if r.get("task_type") not in TASK_TYPES]
    dropped = len(existing) - len(kept)
    merged = kept + new_records
    msg = (f"  u{uid}: existing={len(existing)} dropped_old={dropped} "
           f"+new={len(new_records)} -> total={len(merged)}")
    if apply:
        Path(str(path) + ".bak_missing_tasks").write_text(json.dumps(existing))
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        msg += "  [WRITTEN]"
    else:
        msg += "  [DRY]"
    return msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="space-separated user ids")
    ap.add_argument("--apply", action="store_true", help="write changes (else dry)")
    args = ap.parse_args()
    users = args.users.split()

    llm = ped._build_llm_client()  # Azure gpt-5.5 (BUILDER_LLM_MODEL default)
    if llm is None:
        print("[add_missing_tasks] FATAL: no discovery LLM client — c1e needs it.")
        return 2

    summary: dict[str, dict] = {}
    for uid in users:
        try:
            buckets = build_user(uid, llm)
            counts = {tt: len(buckets.get(tt, [])) for tt in TASK_TYPES}
            rows = project_rows(uid, buckets)
            with tempfile.TemporaryDirectory() as td:
                tmp = str(Path(td) / "new.json")
                dump_test_samples_json(uid, output_path=tmp, precomputed_rows=rows)
                new_records = json.loads(Path(tmp).read_text())
            print(f"  u{uid}: built {counts}  -> {len(new_records)} records")
            print(merge_into_test_json(uid, new_records, args.apply))
            summary[uid] = {"counts": counts, "records": len(new_records)}
        except Exception as exc:
            import traceback
            print(f"  u{uid}: ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary[uid] = {"error": str(exc)}

    print("\n=== SUMMARY ===")
    tot = {tt: 0 for tt in TASK_TYPES}
    for uid, s in summary.items():
        if "counts" in s:
            for tt in TASK_TYPES:
                tot[tt] += s["counts"][tt]
    print("per-task totals:", tot)
    print("personas with error:", [u for u, s in summary.items() if "error" in s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
