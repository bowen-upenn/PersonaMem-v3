"""Re-run llm_postprocess on an existing `benchmark/{user_id}/queries.csv`.

Use when you've changed the postprocess prompts / task-flaw routing /
example-gen guidance and want to refresh the gold + foil text WITHOUT
rebuilding the whole benchmark (which would re-run blind_check + E6
discovery and lose any existing E6 instances).

Usage:
    python scripts/repostprocess_user.py 115
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
csv.field_size_limit(sys.maxsize)

from evaluation.backend_query import BackendQuery
from evaluation.llm_postprocess import postprocess_benchmark
from evaluation.task_registry import normalize_task_type
from scripts.prepare_eval_data import (
    COLUMNS,
    QUERIES_CSV_VERSION,
    _extract_ts,
    _inst_field,
    _make_blind_check_llm,
    _project_row,
    _secondary_sort_key,
    _synthesize_ts_for_no_ts_instance,
)


def _load_existing_bm(user_id: str) -> dict:
    """Reconstruct the in-memory benchmark dict from queries.csv. Applies
    OLD_TO_NEW normalization so renamed task types route to current logic."""
    csv_in = Path("benchmark") / user_id / "queries.csv"
    if not csv_in.exists():
        raise FileNotFoundError(f"no queries.csv at {csv_in}")
    bm: dict[str, list] = defaultdict(list)
    with csv_in.open() as f:
        rdr = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rdr:
            inst = json.loads(row["instance_json"])
            tt = normalize_task_type(row.get("task_type", ""))
            inst["task_id"] = normalize_task_type(inst.get("task_id") or tt)
            bm[tt].append(inst)
    return dict(bm)


def _emit_csv(user_id: str, bm: dict) -> Path:
    pairs: list[tuple[str, dict, int]] = []
    for task_type, bucket in bm.items():
        if not isinstance(bucket, list):
            continue
        for inst in bucket:
            ts = _extract_ts(inst)
            if ts <= 0:
                ts = _synthesize_ts_for_no_ts_instance(
                    inst, task_type, bm, user_id
                )
            if ts <= 0:
                continue
            pairs.append((task_type, inst, ts))

    def sort_key(item):
        tt, inst, ts = item
        iid = _inst_field(
            inst, "instance_id", "pair_id", "scenario_id", "test_id",
            default=f"{tt}_na",
        )
        return (ts, _secondary_sort_key(iid))

    pairs.sort(key=sort_key)

    csv_out = Path("benchmark") / user_id / "queries.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# queries_csv_version={QUERIES_CSV_VERSION}\n")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for seq, (tt, inst, ts) in enumerate(pairs):
            writer.writerow(_project_row(seq, tt, inst, user_id, ts))
    return csv_out


def main(user_id: str, blind_check_model: str = "haiku") -> int:
    bm = _load_existing_bm(user_id)
    print(f"[{user_id}] loaded {sum(len(v) for v in bm.values())} instances "
          f"across {len(bm)} task types")

    bq = BackendQuery(backend_dir="backend")
    llm = _make_blind_check_llm(blind_check_model)

    postprocess_benchmark(
        bm, bq, user_id,
        self_check_llm=llm,
        inferior_llm=llm,
        verbose=True,
    )

    out = _emit_csv(user_id, bm)
    print(f"[{user_id}] wrote {out}")

    # Re-render persona.html + dump test.json so the visualization picks
    # up the new gold/foil text without a follow-up command.
    try:
        from data_preparation.visualize import dump_test_samples_json, generate_persona_html
        print(f"[{user_id}] {dump_test_samples_json(user_id)}")
        print(f"[{user_id}] {generate_persona_html(user_id)}")
    except Exception as exc:
        print(f"[{user_id}] post-render failed: {exc}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/repostprocess_user.py <user_id>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
