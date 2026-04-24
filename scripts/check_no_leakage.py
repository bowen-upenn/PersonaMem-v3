"""Audit a run's overlay + backend views for temporal leakage.

Invariant: at any query with `ts = t_k`, the only events visible to the
agent must satisfy `source_timestamp <= t_k`. The overlay writes.jsonl
is allowed to contain records with `sim_timestamp > t_k` — but only if
they were produced by later queries in the same run, NOT by this query.
We verify by walking the writes.jsonl in append order and checking each
write's sim_timestamp is consistent with the producing query's t_test.

This does NOT exercise the backend reader — BackendQuery already enforces
`since_timestamp` on every read path — but it catches:

  1. Overlay writes stamped at a ts SMALLER than their producing query
     (writes that would appear to predate the user's own action),
  2. Overlay writes whose sim_timestamp > the following query's t_test
     (overshoots that skip a query's window),
  3. Writes whose sim_timestamp is absent or wall-clock (pre-patch regression).

Usage:
  python scripts/check_no_leakage.py benchmark/115/runs/<ts>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10_000_000)


def _load_queries(user_id: str) -> list[dict]:
    p = REPO_ROOT / "benchmark" / user_id / "queries.csv"
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _load_writes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def audit_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        print(f"[check_no_leakage] run_dir does not exist: {run_dir}", file=sys.stderr)
        return {"status": "error", "error": "missing_run_dir"}

    user_id = run_dir.parent.parent.name  # benchmark/{uid}/runs/{ts}
    queries = _load_queries(user_id)
    q_ts_by_id: dict[str, int] = {}
    for r in queries:
        try:
            q_ts_by_id[r["query_id"]] = int(r["ts"])
        except Exception:
            continue
    t_sorted = sorted(int(r["ts"]) for r in queries if r.get("ts"))
    if not t_sorted:
        return {"status": "error", "error": "no_query_timestamps"}

    writes = _load_writes(run_dir / "writes.jsonl")
    issues: list[dict] = []

    # Per-t_test grouping: a valid run stamps writes at t_test + 1..N for the
    # _producing_ query, then moves on when PM3_T_TEST changes.
    prev_sim_ts_per_t: dict[int, int] = {}
    for rec in writes:
        sim_ts = rec.get("sim_timestamp")
        tool = rec.get("tool", "")
        app = rec.get("app", "")
        # 1. Every write must carry a sim_timestamp (post-patch invariant).
        if sim_ts is None:
            issues.append({
                "kind": "missing_sim_timestamp",
                "tool": tool, "app": app,
                "note": "write predates the overlay patch or env was missing",
            })
            continue

        # 2. sim_ts must fall inside a valid (t_prev, t_prev + K] window of
        #    some query's t_test. We approximate by finding the closest
        #    query t_test that is <= sim_ts and checking sim_ts - t < 3600.
        floor_t = None
        for t in t_sorted:
            if t <= sim_ts:
                floor_t = t
            else:
                break
        if floor_t is None:
            issues.append({
                "kind": "sim_ts_before_any_query",
                "sim_ts": sim_ts, "tool": tool, "app": app,
            })
            continue

        offset = sim_ts - floor_t
        # Each write is stamped at t + 1 + k, where k = # of prior writes
        # within the same query. A huge offset (> 3600s = 1hr) suggests
        # the producing query wasn't the one at floor_t — possibly an
        # overshoot that would make the write visible in a later query
        # unexpectedly. Note: later queries legitimately see these writes;
        # the concern is whether the write belongs to that floor_t cohort.
        if offset > 3600:
            issues.append({
                "kind": "suspicious_offset",
                "sim_ts": sim_ts, "floor_t_test": floor_t,
                "offset_seconds": offset, "tool": tool, "app": app,
            })
            continue

        # 3. Monotonicity within a t_test cohort.
        last = prev_sim_ts_per_t.get(floor_t)
        if last is not None and sim_ts < last:
            issues.append({
                "kind": "sim_ts_non_monotonic",
                "floor_t_test": floor_t, "prev": last, "curr": sim_ts,
                "tool": tool, "app": app,
            })
        prev_sim_ts_per_t[floor_t] = sim_ts

    # Results summary stats
    results_csv = run_dir / "results.csv"
    n_results = 0
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8") as f:
            n_results = sum(1 for _ in csv.DictReader(f))

    summary = {
        "run_dir": str(run_dir),
        "user_id": user_id,
        "n_queries": len(queries),
        "n_results": n_results,
        "n_writes": len(writes),
        "n_issues": len(issues),
        "issues": issues[:50],  # cap for readability
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="benchmark/{uid}/runs/{ts}")
    ap.add_argument("--json", action="store_true", help="print full JSON instead of short summary")
    args = ap.parse_args()

    out = audit_run(Path(args.run_dir))
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(
            f"[check_no_leakage] user={out.get('user_id')}  queries={out.get('n_queries')}  "
            f"results={out.get('n_results')}  writes={out.get('n_writes')}  "
            f"issues={out.get('n_issues')}"
        )
        for iss in out.get("issues", [])[:10]:
            print("  !", json.dumps(iss, ensure_ascii=False))
    return 0 if out.get("n_issues", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
