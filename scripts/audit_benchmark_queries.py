#!/usr/bin/env python3
"""Per-query LLM quality audit for a built benchmark.

Reads `benchmark/{user_id}/queries.csv`, parses each row's
`instance_json`, and runs the dimensions defined in
`evaluation/audit_query_quality.py` against each query using a mini-tier
LLM (default: gpt-5.4-mini).

Outputs:
  benchmark/{user_id}/runs/{ts}/audit_queries.jsonl       (one row per query)
  benchmark/{user_id}/runs/{ts}/audit_queries_summary.json (per-task / per-dim pass rates)
  benchmark/{user_id}/runs/{ts}/audit_queries_summary.md   (human-readable table)

Differs from `scripts/audit_test_queries.py` (which is a deterministic,
schema-level distribution audit with no LLM calls).

Usage:
  python scripts/audit_benchmark_queries.py --user_id 115
  python scripts/audit_benchmark_queries.py --user_id 115 --task personalized_recommendation
  python scripts/audit_benchmark_queries.py --user_id 115 --limit 10  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.audit_query_quality import audit_query, QueryAuditResult
from evaluation.backend_query import BackendQuery


def _load_queries(csv_path: Path) -> list[dict]:
    # instance_json fields can blow past the default 128 KB CSV limit.
    csv.field_size_limit(sys.maxsize)
    rows: list[dict] = []
    with csv_path.open() as f:
        sniff = f.readline()
        if not sniff.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _flatten_row(row: dict) -> dict:
    """Merge top-level CSV columns and the parsed instance_json so the
    audit dimensions can read either source uniformly."""
    inst_json_raw = row.get("instance_json") or "{}"
    try:
        inst = json.loads(inst_json_raw)
    except json.JSONDecodeError:
        inst = {}
    inst = dict(inst)
    # Top-level fields override only when missing in the nested JSON.
    for k, v in row.items():
        if k == "instance_json":
            continue
        inst.setdefault(k, v)
    inst["task_type"] = row.get("task_type") or inst.get("task_type") or inst.get("task_id") or ""
    inst["query_id"] = row.get("query_id") or inst.get("query_id") or inst.get("instance_id") or ""
    return inst


def _build_llm(model: str, rate_limit: int):
    from query_llm import QueryLLM
    return QueryLLM({"models": {"llm_model": model}}, rate_limit_per_min=rate_limit)


def _render_markdown(summary: dict, total: int, model: str) -> str:
    lines = [
        f"# Per-query Audit Summary",
        f"- model: `{model}`",
        f"- total queries: {total}",
        "",
        "| task_type | dim | passed | failed | skipped | pass_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task_type in sorted(summary):
        for dim_name in sorted(summary[task_type]):
            slot = summary[task_type][dim_name]
            pr = slot.get("pass_rate")
            pr_str = f"{pr*100:.1f}%" if pr is not None else "—"
            lines.append(
                f"| {task_type} | {dim_name} | {slot['passed']} | "
                f"{slot['failed']} | {slot['skipped']} | {pr_str} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--queries_csv", default=None,
                        help="Path to queries.csv (default: benchmark/{user_id}/queries.csv)")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: benchmark/{user_id}/runs/{ts})")
    parser.add_argument("--task", default=None,
                        help="If set, audit only this task_type")
    parser.add_argument("--limit", type=int, default=None,
                        help="Audit at most N queries (smoke test)")
    parser.add_argument("--model", default=os.getenv("AUDIT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--rate_limit", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true",
                        help="Don't call the LLM — just enumerate dimension applicability")
    parser.add_argument("--backend_dir", default="backend",
                        help="Path to the backend root used by the tool-call validator's "
                             "dry-run executor (default: backend)")
    parser.add_argument("--skip-tool-call-validity", action="store_true",
                        help="Disable the tool_call_validity dimension (skips MCP read "
                             "dry-runs + supportability LLM judge for agentic / E3 / E6 tasks)")
    args = parser.parse_args()

    csv_path = Path(args.queries_csv) if args.queries_csv else (
        Path("benchmark") / args.user_id / "queries.csv"
    )
    if not csv_path.exists():
        sys.exit(f"queries.csv not found at {csv_path}")
    rows = _load_queries(csv_path)
    if args.task:
        rows = [r for r in rows if r.get("task_type") == args.task]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("No queries to audit (after --task / --limit filtering)")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("benchmark") / args.user_id / "runs" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Run only deterministic dims; LLM-backed dims will return errors.
        # Useful to confirm the script wires up correctly without burning calls.
        class _NullLLM:
            def query_llm(self, prompt):
                raise RuntimeError("dry_run mode — LLM disabled")
        llm = _NullLLM()
        print(f"[audit] dry_run — auditing {len(rows)} queries with no LLM calls")
    else:
        llm = _build_llm(args.model, args.rate_limit)
        print(f"[audit] auditing {len(rows)} queries with model={args.model}")

    # `bq` enables _dim_tool_call_validity to dry-run MCP read tools at each
    # instance's t_test against the local backend. None disables it.
    bq: BackendQuery | None = None
    if not args.skip_tool_call_validity:
        try:
            bq = BackendQuery(args.backend_dir)
            print(f"[audit] tool_call_validity ENABLED (backend_dir={args.backend_dir})")
        except Exception as exc:
            print(f"[audit] WARN: could not init BackendQuery from {args.backend_dir!r}: {exc}; "
                  f"tool_call_validity will self-skip")
            bq = None
    else:
        print("[audit] tool_call_validity DISABLED (--skip-tool-call-validity)")

    results: list[QueryAuditResult] = []
    summary: dict = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0}))
    out_jsonl = out_dir / "audit_queries.jsonl"
    with out_jsonl.open("w") as f:
        for i, row in enumerate(rows):
            inst = _flatten_row(row)
            res = audit_query(inst, llm, query_id=inst["query_id"], bq=bq)
            results.append(res)
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            for d in res.dimensions:
                slot = summary[res.task_type][d.name]
                if d.skipped:
                    slot["skipped"] += 1
                elif d.passed:
                    slot["passed"] += 1
                else:
                    slot["failed"] += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(f"[audit]   {i+1}/{len(rows)}")

    # Compute pass rates and emit summary files.
    summary_clean: dict = {}
    for task_type, dims in summary.items():
        bucket = summary_clean.setdefault(task_type, {})
        for dim_name, slot in dims.items():
            evaluated = slot["passed"] + slot["failed"]
            slot["pass_rate"] = (slot["passed"] / evaluated) if evaluated else None
            bucket[dim_name] = dict(slot)

    (out_dir / "audit_queries_summary.json").write_text(
        json.dumps({
            "user_id": args.user_id,
            "model": args.model,
            "total_queries": len(rows),
            "queries_csv": str(csv_path),
            "summary": summary_clean,
        }, ensure_ascii=False, indent=2)
    )
    md = _render_markdown(summary_clean, total=len(rows), model=args.model)
    (out_dir / "audit_queries_summary.md").write_text(md)

    print(f"\n[audit] wrote results to {out_dir}")
    print(f"[audit] summary:\n{md}")


if __name__ == "__main__":
    main()
