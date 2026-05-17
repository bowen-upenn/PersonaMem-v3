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


def _load_queries(csv_path: Path) -> tuple[list[dict], str, list[str]]:
    """Read queries.csv. Returns (rows, header_comment, fieldnames).

    The CSV starts with a `# queries_csv_version=...` comment line which
    must be preserved verbatim on rewrite. The DictReader sees the first
    actual line as the header.
    """
    # instance_json fields can blow past the default 128 KB CSV limit.
    csv.field_size_limit(sys.maxsize)
    rows: list[dict] = []
    header_comment = ""
    with csv_path.open() as f:
        sniff = f.readline()
        if sniff.startswith("#"):
            header_comment = sniff
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)
    return rows, header_comment, fieldnames


def _rewrite_queries_csv(
    csv_path: Path,
    raw_rows: list[dict],
    header_comment: str,
    fieldnames: list[str],
    updates: dict[int, dict],
) -> None:
    """Rewrite `queries.csv` with `updates` applied.

    `raw_rows` is the original parsed rows (from `_load_queries`).
    `updates[row_idx]` maps the row index to the regenerated instance
    dict — we re-serialize it into the `instance_json` column.
    """
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if header_comment:
            f.write(header_comment if header_comment.endswith("\n") else header_comment + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(raw_rows):
            out_row = dict(row)
            if i in updates:
                out_row["instance_json"] = json.dumps(
                    updates[i], ensure_ascii=False, separators=(",", ":"),
                )
            writer.writerow(out_row)


def _parse_instance(row: dict) -> dict:
    """Parse the row's `instance_json` column as-is, without merging top-
    level CSV columns. This is the dict we write back to the CSV after
    regen — keeping it clean of the top-level wrapper fields.
    """
    inst_json_raw = row.get("instance_json") or "{}"
    try:
        return json.loads(inst_json_raw)
    except json.JSONDecodeError:
        return {}


def _flatten_row(row: dict, raw_inst: dict | None = None) -> dict:
    """Merge top-level CSV columns and the parsed instance_json so the
    audit dimensions can read either source uniformly. This DOES NOT
    mutate `raw_inst` — pass a separate dict for the rewrite path.
    """
    inst = dict(raw_inst if raw_inst is not None else _parse_instance(row))
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
    parser.add_argument("--no_regen", action="store_true",
                        help="Disable the auto-regenerate path. When set, rows that fail "
                             "the inferior_axis_check are flagged in the JSONL only — "
                             "the CSV is NOT rewritten. Mirrors the legacy flag-only "
                             "behavior for cheap audit-only runs.")
    parser.add_argument("--max_regen_calls", type=int, default=100,
                        help="Cap on auto-regenerate LLM calls per audit run. Default "
                             "100; raise for large benchmarks, lower for cost-control "
                             "smoke tests.")
    parser.add_argument("--regen_model", default=os.getenv("AUDIT_REGEN_MODEL", "gpt-5.4-mini"),
                        help="Model used by the regen path's _generate_inferior call. "
                             "Default: gpt-5.4-mini (same as the audit model).")
    args = parser.parse_args()

    csv_path = Path(args.queries_csv) if args.queries_csv else (
        Path("benchmark") / args.user_id / "queries.csv"
    )
    if not csv_path.exists():
        sys.exit(f"queries.csv not found at {csv_path}")
    raw_rows, header_comment, fieldnames = _load_queries(csv_path)
    # Track original row indices so we can rewrite the right rows after regen.
    rows: list[tuple[int, dict]] = list(enumerate(raw_rows))
    if args.task:
        rows = [(i, r) for (i, r) in rows if r.get("task_type") == args.task]
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

    # Build the regen LLM client lazily on first use (only when needed).
    regen_llm: callable | None = None
    regen_bq: BackendQuery | None = bq
    if not args.no_regen and not args.dry_run:
        # Reuse the audit BackendQuery when available; otherwise spin one up
        # because `regenerate_inferior_for_instance` reads per-app events to
        # build the persona context (top_categories / recent_negatives / etc).
        if regen_bq is None:
            try:
                regen_bq = BackendQuery(args.backend_dir)
            except Exception as exc:
                print(f"[audit] WARN: regen disabled — could not init BackendQuery "
                      f"from {args.backend_dir!r}: {exc}")
                regen_bq = None
        if regen_bq is not None:
            try:
                from query_llm import QueryLLM as _QLM
                _regen_qlm = _QLM(
                    {"models": {"llm_model": args.regen_model}},
                    rate_limit_per_min=args.rate_limit,
                )

                def _regen_llm(prompt: str) -> str:
                    try:
                        return _regen_qlm.query_llm(prompt) or ""
                    except Exception:
                        return ""
                regen_llm = _regen_llm
                print(f"[audit] auto-regen ENABLED (model={args.regen_model}, "
                      f"max_calls={args.max_regen_calls})")
            except Exception as exc:
                print(f"[audit] WARN: regen disabled — could not init regen LLM: {exc}")
                regen_llm = None
    else:
        if args.no_regen:
            print("[audit] auto-regen DISABLED (--no_regen)")

    results: list[QueryAuditResult] = []
    summary: dict = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0}))
    csv_updates: dict[int, dict] = {}  # row_idx → updated instance dict
    regen_calls_used = 0
    n_regen_attempted = 0
    n_regen_success = 0
    n_regen_exhausted = 0

    out_jsonl = out_dir / "audit_queries.jsonl"
    n_synth_filled = 0
    with out_jsonl.open("w") as f:
        for i, (row_idx, row) in enumerate(rows):
            # `raw_inst` is the unmerged parsed instance_json — the dict we
            # write back to the CSV after regen. `inst` is the flattened
            # view (parsed JSON + top-level CSV columns) used only by the
            # audit dimensions for reading.
            raw_inst = _parse_instance(row)
            inst = _flatten_row(row, raw_inst=raw_inst)

            # Synthesis pass for task families that the standard build
            # pipeline doesn't generate example/inferior pairs for
            # (multi-query repetition clusters + sensitive-event silence).
            # Deterministic template-based — no LLM call. Fills BOTH
            # fields, stages the row for CSV rewrite.
            if not raw_inst.get("inferior_response") and not raw_inst.get("example_response"):
                from evaluation.llm_postprocess import (
                    synthesize_special_task_example_inferior,
                )
                synth = synthesize_special_task_example_inferior(
                    raw_inst, inst.get("task_type") or "",
                )
                if synth:
                    raw_inst["example_response"] = synth["example_response"]
                    raw_inst["inferior_response"] = synth["inferior_response"]
                    # Mirror into the flattened view so the dim sees them.
                    inst["example_response"] = synth["example_response"]
                    inst["inferior_response"] = synth["inferior_response"]
                    csv_updates[row_idx] = raw_inst
                    n_synth_filled += 1

            res = audit_query(inst, llm, query_id=inst["query_id"], bq=bq)

            # Auto-regenerate when the per-task axis check failed and the
            # regen path is enabled + under the cap.
            axis_dim = next(
                (d for d in res.dimensions
                 if d.name == "inferior_axis_check" and not d.skipped),
                None,
            )
            if (axis_dim is not None and not axis_dim.passed
                    and regen_llm is not None
                    and regen_calls_used < args.max_regen_calls):
                from evaluation.llm_postprocess import regenerate_inferior_for_instance
                n_regen_attempted += 1
                # Feed the audit's failure reason back as `axis_hint` so the
                # LLM rewrite sees the specific reason it failed last time.
                axis_hint = (
                    f"{axis_dim.reason or ''} "
                    f"The foil MUST commit the labeled failure axis."
                ).strip()
                regen_calls_used += 1
                new_inferior = regenerate_inferior_for_instance(
                    inst=inst,
                    task_id=inst["task_type"],
                    bq=regen_bq,
                    user_id=str(inst.get("user_id") or args.user_id),
                    inferior_llm=regen_llm,
                    axis_hint=axis_hint,
                )
                if new_inferior:
                    # Re-audit the regenerated foil. If it passes, replace
                    # the per-row result's axis dim with the new outcome
                    # and stage the row for CSV rewrite. If it fails again,
                    # leave the original audit row in place and tag the
                    # outcome as "exhausted" so a human can hand-fix.
                    inst["inferior_response"] = new_inferior
                    res2 = audit_query(inst, llm, query_id=inst["query_id"], bq=bq)
                    axis_dim2 = next(
                        (d for d in res2.dimensions
                         if d.name == "inferior_axis_check"),
                        None,
                    )
                    regen_outcome = "ok" if (axis_dim2 and axis_dim2.passed) else "still_failing"
                    if regen_outcome == "ok":
                        n_regen_success += 1
                        res = res2  # adopt the new result for summary stats
                        # Mutate the RAW parsed instance for the CSV rewrite —
                        # avoids leaking the top-level wrapper columns
                        # (query_id, seq, task_family, mcp_tools_allowed, etc.)
                        # back into instance_json.
                        raw_inst["inferior_response"] = new_inferior
                        csv_updates[row_idx] = raw_inst
                    else:
                        n_regen_exhausted += 1
                    # Annotate the JSONL row so reviewers can trace the regen.
                    for d in res.dimensions:
                        if d.name == "inferior_axis_check":
                            d.reason = (
                                f"{d.reason} | regen_outcome={regen_outcome}"
                                f"; regen_attempts=1"
                            )[:240]
                            break
                else:
                    n_regen_exhausted += 1
                    for d in res.dimensions:
                        if d.name == "inferior_axis_check":
                            d.reason = (
                                f"{d.reason} | regen_outcome=no_new_foil; "
                                f"regen_attempts=1"
                            )[:240]
                            break

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
                print(f"[audit]   {i+1}/{len(rows)}  "
                      f"regen={n_regen_attempted}/ok={n_regen_success}/exhausted={n_regen_exhausted}")

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
            "regen": {
                "enabled": regen_llm is not None,
                "max_calls": args.max_regen_calls,
                "calls_used": regen_calls_used,
                "attempts": n_regen_attempted,
                "succeeded": n_regen_success,
                "exhausted": n_regen_exhausted,
            },
            "summary": summary_clean,
        }, ensure_ascii=False, indent=2)
    )
    md = _render_markdown(summary_clean, total=len(rows), model=args.model)
    (out_dir / "audit_queries_summary.md").write_text(md)

    # Rewrite queries.csv if any rows were regenerated successfully.
    if csv_updates:
        _rewrite_queries_csv(csv_path, raw_rows, header_comment, fieldnames, csv_updates)
        print(f"[audit] rewrote {len(csv_updates)} row(s) in {csv_path} "
              f"(regen succeeded)")

    print(f"\n[audit] wrote results to {out_dir}")
    if n_synth_filled:
        print(f"[audit] synthesized example/inferior for {n_synth_filled} "
              f"missing-foil row(s)")
    if regen_llm is not None:
        print(f"[audit] regen: attempted={n_regen_attempted} ok={n_regen_success} "
              f"exhausted={n_regen_exhausted} calls_used={regen_calls_used}/"
              f"{args.max_regen_calls}")
    print(f"[audit] summary:\n{md}")


if __name__ == "__main__":
    main()
