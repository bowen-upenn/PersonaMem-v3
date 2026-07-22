#!/usr/bin/env python3
"""Study 1 (ablation): run the per-query QA data-quality auto-verification
over one persona's REAL test set and collect per-criterion pass rates.

The shipping CLI (`scripts/audit_benchmark_queries.py`) reads
`benchmark/{uid}/queries.csv`, which only exists for built benchmarks.
The data the eval actually consumes is `backend/{uid}/test.json` — a list
of instance dicts whose `instance_full` field is the runner's `inst`
(exactly how `evaluation/run_eval.py::_load_queries` projects them). This
driver feeds those same instances to `evaluation.audit_query_quality.
audit_query`, so the audit is byte-for-byte the one the pipeline runs.

Outputs (under --out_dir):
  audit_rows.jsonl            one row per query (all dimension verdicts)
  audit_summary.json          per-criterion + per-task pass/fail/skip + rates

Usage:
  python results/_scripts/run_qa_audit.py --user_id 1 --model gpt-5.5
  python results/_scripts/run_qa_audit.py --user_id 1 --limit 2   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.audit_query_quality import audit_query  # noqa: E402
from evaluation.backend_query import BackendQuery  # noqa: E402


def _flatten(item: dict, user_id: str) -> dict:
    """Project a test.json row into the runner's `inst` view, matching
    run_eval._load_queries: inst = instance_full, with user_query / task_type
    / query_id promoted, and user_id injected (it lives outside the row)."""
    inst = dict(item.get("instance_full") or item)
    inst.setdefault("task_type", item.get("task_type") or inst.get("task_id") or "")
    inst.setdefault("task_id", inst.get("task_type"))
    inst.setdefault("query_id", item.get("query_id") or inst.get("test_id") or "")
    if item.get("user_query") and not inst.get("user_query"):
        inst["user_query"] = item["user_query"]
    inst["user_id"] = str(user_id)
    # source_timestamp is sometimes a stringified int in the JSON dump.
    ts = inst.get("source_timestamp")
    if isinstance(ts, str) and ts.isdigit():
        inst["source_timestamp"] = int(ts)
    return inst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--model", default=os.getenv("AUDIT_MODEL", "gpt-5.5"))
    ap.add_argument("--rate_limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backend_dir", default="backend")
    args = ap.parse_args()

    test_path = Path(args.backend_dir) / args.user_id / "test.json"
    items = json.loads(test_path.read_text())
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir or f"results/audit/qa_audit_p{args.user_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from query_llm import QueryLLM

    class _LLM:
        """Dual-interface shim. The audit dims are inconsistent: some call
        ``llm(prompt)`` (need a callable), others ``llm.query_llm(prompt)``
        (need the object). QueryLLM is not callable, so a bare object makes
        the ``llm(prompt)`` dims (completeness / schema_sanity /
        sensitive_probe_placement / response_quality) silently fall through.
        This wrapper satisfies both."""
        def __init__(self, q):
            self._q = q

        def __call__(self, prompt):
            return self._q.query_llm(prompt)

        def query_llm(self, prompt, *a, **k):
            return self._q.query_llm(prompt, *a, **k)

    llm = _LLM(QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit))
    try:
        bq = BackendQuery(args.backend_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa_audit] WARN: BackendQuery init failed ({exc}); "
              f"tool_call_validity + frame_consistency will self-skip")
        bq = None

    print(f"[qa_audit] auditing {len(items)} queries for persona {args.user_id} "
          f"with model={args.model}")

    # per (dimension) and per (task_type, dimension) tallies
    by_dim: dict = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    by_task_dim: dict = defaultdict(lambda: defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0}))
    fail_examples: dict = defaultdict(list)

    rows_path = out_dir / "audit_rows.jsonl"
    t0 = time.time()
    # audit_query per item is independent; bq reads are in-memory (safe to
    # share), QueryLLM is thread-safe. Fan items across a thread pool and
    # fold the tallies under a lock so progress flushes live.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fh = rows_path.open("w")
    lock = threading.Lock()
    done = [0]

    def _audit_one(item):
        inst = _flatten(item, args.user_id)
        return audit_query(inst, llm, query_id=inst.get("query_id", ""), bq=bq)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_audit_one, it) for it in items]):
            res = fut.result()
            with lock:
                fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                for d in res.dimensions:
                    slot = by_dim[d.name]
                    tslot = by_task_dim[res.task_type][d.name]
                    if d.skipped:
                        slot["skipped"] += 1
                        tslot["skipped"] += 1
                    elif d.passed:
                        slot["passed"] += 1
                        tslot["passed"] += 1
                    else:
                        slot["failed"] += 1
                        tslot["failed"] += 1
                        if len(fail_examples[d.name]) < 5:
                            fail_examples[d.name].append({
                                "query_id": res.query_id,
                                "task_type": res.task_type,
                                "reason": (d.reason or "")[:240],
                            })
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(items):
                    print(f"[qa_audit]   {done[0]}/{len(items)}  "
                          f"({time.time()-t0:.0f}s)", flush=True)
    fh.close()

    def _rate(slot: dict):
        ev = slot["passed"] + slot["failed"]
        return (slot["passed"] / ev) if ev else None

    dim_summary = {}
    for name, slot in by_dim.items():
        dim_summary[name] = {**slot, "evaluated": slot["passed"] + slot["failed"],
                             "pass_rate": _rate(slot)}
    task_summary = {}
    for tt, dims in by_task_dim.items():
        task_summary[tt] = {n: {**s, "pass_rate": _rate(s)} for n, s in dims.items()}

    # overall (micro) across all evaluated (non-skipped) dimension checks
    tot_pass = sum(s["passed"] for s in by_dim.values())
    tot_fail = sum(s["failed"] for s in by_dim.values())
    tot_eval = tot_pass + tot_fail

    summary = {
        "user_id": args.user_id,
        "model": args.model,
        "n_queries": len(items),
        "overall": {
            "evaluated_checks": tot_eval,
            "passed": tot_pass,
            "failed": tot_fail,
            "pass_rate": (tot_pass / tot_eval) if tot_eval else None,
        },
        "by_dimension": dim_summary,
        "by_task_dimension": task_summary,
        "fail_examples": dict(fail_examples),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[qa_audit] overall pass rate: "
          f"{summary['overall']['pass_rate']*100:.1f}% "
          f"({tot_pass}/{tot_eval} checks)  in {time.time()-t0:.0f}s")
    print("[qa_audit] per-dimension:")
    for name in sorted(dim_summary):
        s = dim_summary[name]
        pr = f"{s['pass_rate']*100:.1f}%" if s["pass_rate"] is not None else "—"
        print(f"    {name:28s} pass={s['passed']:3d} fail={s['failed']:2d} "
              f"skip={s['skipped']:3d}  rate={pr}")
    print(f"[qa_audit] wrote {rows_path} + audit_summary.json")


if __name__ == "__main__":
    main()
