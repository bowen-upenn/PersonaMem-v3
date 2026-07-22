#!/usr/bin/env python3
"""Surgical add of new_suggestions_chatbot (chatbot-only fresh-suggestion task)
into each persona's test.json. Touches ONLY that task_type — every other row
(including the short_vs_long_term_lifecycle rows already being evaluated) is
preserved. Idempotent: existing new_suggestions_* rows are dropped before
re-appending. Gold is built with the discovery LLM (Azure gpt-5.5).

Usage:
  python scripts/add_chatbot_suggestions.py --users "<id id ...>" [--apply]
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
from evaluation.build_benchmark import build_c1e_new_suggestions        # noqa: E402
from evaluation import task_distribution as _task_dist                  # noqa: E402
from data_preparation.visualize import dump_test_samples_json           # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "prepare_eval_data", "scripts/prepare_eval_data.py")
ped = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ped)

# Drop any stale new_suggestions_* (incl. the retired recsys) before re-adding.
DROP_TYPES = {"new_suggestions_chatbot", "new_suggestions_recsys"}
EMIT_TYPE = "new_suggestions_chatbot"
BACKEND = "backend"


def build_chatbot(uid: str, llm) -> list[dict]:
    bq = BackendQuery(BACKEND)
    ti = load_test_items(BACKEND, uid)
    c1e = build_c1e_new_suggestions(bq, uid, ti, discovery_llm=llm, rng_seed=0)
    buckets = {EMIT_TYPE: c1e.get(EMIT_TYPE, [])}
    return _task_dist.apply_caps(buckets, rng_seed=0).get(EMIT_TYPE, [])


def project_rows(uid: str, insts: list[dict]) -> list[dict]:
    rows, seq = [], 0
    for inst in insts:
        ts = ped._extract_ts(inst)
        if ts <= 0:
            ts = ped._synthesize_ts_for_no_ts_instance(inst, EMIT_TYPE, {EMIT_TYPE: insts}, uid)
        if ts <= 0:
            print(f"    [{uid}] skip (no ts): {inst.get('instance_id','?')}")
            continue
        row = ped._project_row(seq, EMIT_TYPE, inst, uid, ts)
        row["query_id"] = f"{uid}:append:{row['instance_id']}"
        rows.append(row)
        seq += 1
    return rows


def merge(uid: str, new_records: list[dict], apply: bool) -> str:
    path = Path(BACKEND) / uid / "test.json"
    existing = json.loads(path.read_text())
    kept = [r for r in existing if r.get("task_type") not in DROP_TYPES]
    merged = kept + new_records
    msg = (f"  u{uid}: existing={len(existing)} dropped_old={len(existing)-len(kept)} "
           f"+new={len(new_records)} -> total={len(merged)}")
    if apply:
        Path(str(path) + ".bak_chatbot_sugg").write_text(json.dumps(existing))
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        msg += "  [WRITTEN]"
    else:
        msg += "  [DRY]"
    return msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    llm = ped._build_llm_client()
    if llm is None:
        print("FATAL: no discovery LLM client"); return 2
    total = 0
    for uid in args.users.split():
        try:
            insts = build_chatbot(uid, llm)
            rows = project_rows(uid, insts)
            with tempfile.TemporaryDirectory() as td:
                tmp = str(Path(td) / "new.json")
                dump_test_samples_json(uid, output_path=tmp, precomputed_rows=rows)
                new_records = json.loads(Path(tmp).read_text())
            print(f"  u{uid}: built {len(insts)} chatbot -> {len(new_records)} records")
            print(merge(uid, new_records, args.apply))
            total += len(new_records)
        except Exception as exc:
            import traceback
            print(f"  u{uid}: ERROR {type(exc).__name__}: {exc}"); traceback.print_exc()
    print(f"\n=== TOTAL new_suggestions_chatbot records: {total} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
