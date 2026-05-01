#!/usr/bin/env python3
"""Audit backend/{uid}/test.json — flag broken or mislabeled queries.

Modes:
  # one-shot audit
  python scripts/audit_test_queries.py --user_id 115 --phase before

  # diff two prior phase reports
  python scripts/audit_test_queries.py --user_id 115 --diff before after

The audit is read-only and uses no LLM calls. Mislabel detection draws
on the existing ``blind_check_score`` already saved by build_benchmark.

Outputs:
  backend/{uid}/test_audit_{phase}.md
  backend/{uid}/test_audit_{phase}.json
  backend/{uid}/test_audit_diff.md   (only with --diff)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the project root importable when running this file directly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.audit_rules import (
    Finding,
    distribution_findings,
    run_per_record_rules,
    task_count_table,
    TASK_TARGETS,
)
from evaluation.task_distribution import DATA_DEPENDENT_TASKS


def _load_records(test_json_path: str) -> list[dict]:
    with open(test_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ascii_bar(n: int, max_n: int, width: int = 40) -> str:
    if max_n <= 0:
        return ""
    fill = int(round((n / max_n) * width))
    return "█" * fill + "·" * (width - fill)


def _flag_marker(n: int, lo: int, hi: int, tt: str = "") -> str:
    if hi == 0 and n == 0:
        return ""  # task type not in TASK_TARGETS — silent
    if tt in DATA_DEPENDENT_TASKS and n < lo:
        return "data-dependent"
    if n < lo:
        return f"⚠ under min ({lo})"
    if n > hi:
        return f"⚠ over max ({hi})"
    return "ok"


def _render_markdown(records: list[dict], findings: list[Finding], phase: str) -> str:
    by_rule: dict[str, list[Finding]] = defaultdict(list)
    by_severity: Counter = Counter()
    for f in findings:
        by_rule[f.rule].append(f)
        by_severity[f.severity] += 1

    lines: list[str] = []
    lines.append(f"# Test-query audit — phase `{phase}`")
    lines.append("")
    lines.append(f"- total records: **{len(records)}**")
    lines.append(f"- total findings: **{len(findings)}** "
                 f"(high {by_severity['high']}, medium {by_severity['medium']}, low {by_severity['low']})")
    lines.append("")

    # Distribution summary
    lines.append("## Distribution")
    lines.append("")
    lines.append("| task_type | count | target [min, max] | status |")
    lines.append("|---|---:|:---:|:---|")
    rows = task_count_table(records)
    max_n = max((n for _, n, _, _ in rows), default=1)
    for tt, n, lo, hi in rows:
        bar = _ascii_bar(n, max_n)
        flag = _flag_marker(n, lo, hi, tt)
        lines.append(f"| `{tt}` | {n} | [{lo}, {hi}] | {flag} `{bar}` |")
    lines.append("")

    qk_counts = Counter(r.get("query_kind") for r in records)
    eb_counts = Counter(r.get("expected_behavior") for r in records)
    lines.append(f"- query_kind: " + ", ".join(f"`{k}`={v}" for k, v in qk_counts.most_common()))
    lines.append(f"- expected_behavior: " + ", ".join(f"`{k}`={v}" for k, v in eb_counts.most_common()))
    lines.append("")

    # Findings grouped by rule
    lines.append("## Findings")
    lines.append("")
    rule_order = sorted(by_rule.keys(), key=lambda k: (-len(by_rule[k]), k))
    for rule in rule_order:
        rule_findings = by_rule[rule]
        sev = max((f.severity for f in rule_findings), key=lambda s: {"high": 3, "medium": 2, "low": 1}[s])
        lines.append(f"### `{rule}` ({len(rule_findings)} occurrences, max severity: {sev})")
        lines.append("")
        # Show breakdown by task_type
        by_tt = Counter(f.task_type for f in rule_findings)
        lines.append("| task_type | count |")
        lines.append("|---|---:|")
        for tt, n in by_tt.most_common():
            lines.append(f"| `{tt}` | {n} |")
        lines.append("")
        lines.append("Sample (up to 5):")
        lines.append("")
        for f in rule_findings[:5]:
            lines.append(f"- **{f.query_id}** ({f.task_type}, {f.severity}, "
                         f"action: {f.suggested_action}) — {f.message}")
        if len(rule_findings) > 5:
            lines.append(f"- … and {len(rule_findings) - 5} more (see test_audit_{phase}.json for full list)")
        lines.append("")

    return "\n".join(lines) + "\n"


def _diff_reports(uid: str, phase_a: str, phase_b: str, backend_dir: str) -> str:
    """Compare two audit JSON files and emit a diff Markdown report."""
    base = Path(backend_dir) / str(uid)
    pa = base / f"test_audit_{phase_a}.json"
    pb = base / f"test_audit_{phase_b}.json"
    a = json.loads(pa.read_text())
    b = json.loads(pb.read_text())

    a_findings = a.get("findings") or []
    b_findings = b.get("findings") or []
    a_by_rule = Counter(f["rule"] for f in a_findings)
    b_by_rule = Counter(f["rule"] for f in b_findings)
    rules = set(a_by_rule) | set(b_by_rule)

    lines: list[str] = []
    lines.append(f"# Audit diff: `{phase_a}` → `{phase_b}` (user {uid})")
    lines.append("")
    lines.append(f"- {phase_a}: {len(a_findings)} findings over {a.get('record_count', 0)} records")
    lines.append(f"- {phase_b}: {len(b_findings)} findings over {b.get('record_count', 0)} records")
    lines.append("")

    lines.append("## Findings by rule")
    lines.append("")
    lines.append(f"| rule | {phase_a} | {phase_b} | delta |")
    lines.append("|---|---:|---:|---:|")
    for rule in sorted(rules):
        na = a_by_rule.get(rule, 0)
        nb = b_by_rule.get(rule, 0)
        delta = nb - na
        sign = "+" if delta > 0 else ("" if delta == 0 else "")
        lines.append(f"| `{rule}` | {na} | {nb} | {sign}{delta} |")
    lines.append("")

    a_dist = a.get("by_task") or {}
    b_dist = b.get("by_task") or {}
    all_tt = sorted(set(a_dist) | set(b_dist))
    lines.append("## Distribution by task_type")
    lines.append("")
    lines.append(f"| task_type | {phase_a} | {phase_b} | delta |")
    lines.append("|---|---:|---:|---:|")
    for tt in all_tt:
        na = a_dist.get(tt, 0)
        nb = b_dist.get(tt, 0)
        delta = nb - na
        lines.append(f"| `{tt}` | {na} | {nb} | {delta:+d} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _run_audit(uid: str, phase: str, benchmark_dir: str, backend_dir: str) -> tuple[str, str]:
    test_json = os.path.join(backend_dir, str(uid), "test.json")
    if not os.path.exists(test_json):
        raise FileNotFoundError(
            f"{test_json} not found. Run `python scripts/dump_test_json.py --user_id {uid}` first."
        )
    records = _load_records(test_json)
    findings: list[Finding] = []
    findings.extend(run_per_record_rules(records))
    findings.extend(distribution_findings(records))

    by_task = Counter(r["task_type"] for r in records)
    out_md_path = os.path.join(backend_dir, str(uid), f"test_audit_{phase}.md")
    out_json_path = os.path.join(backend_dir, str(uid), f"test_audit_{phase}.json")

    md = _render_markdown(records, findings, phase)
    Path(out_md_path).write_text(md, encoding="utf-8")

    payload = {
        "user_id": str(uid),
        "phase": phase,
        "record_count": len(records),
        "by_task": dict(by_task),
        "findings": [f.as_dict() for f in findings],
    }
    Path(out_json_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_md_path, out_json_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user_id", required=True)
    p.add_argument("--phase", default="snapshot",
                   help="label for this audit run, e.g. before / after / snapshot")
    p.add_argument("--diff", nargs=2, metavar=("PHASE_A", "PHASE_B"), default=None,
                   help="compare two prior phase reports instead of running a new one")
    p.add_argument("--benchmark_dir", default="benchmark")
    p.add_argument("--backend_dir", default="backend")
    args = p.parse_args()

    if args.diff:
        md = _diff_reports(args.user_id, args.diff[0], args.diff[1], args.backend_dir)
        out = os.path.join(args.backend_dir, str(args.user_id), "test_audit_diff.md")
        Path(out).write_text(md, encoding="utf-8")
        print(f"wrote {out}")
        return 0

    md_path, json_path = _run_audit(args.user_id, args.phase, args.benchmark_dir, args.backend_dir)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    # Quick stdout summary
    payload = json.loads(Path(json_path).read_text())
    by_rule = Counter(f["rule"] for f in payload["findings"])
    print(f"\n{len(payload['findings'])} findings over {payload['record_count']} records:")
    for rule, n in by_rule.most_common():
        print(f"  {n:4d}  {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
