#!/usr/bin/env python
"""Recompute the pr-rubric headline from STORED per-dim judge scores — no LLM.

Use case: a rubric re-focus that only changes WHICH stored dims count toward
the headline (and their primary/secondary roles), not the dim definitions —
e.g. the 2026-06-12 v3 focus (each task = primary + ≤1 secondary; guardrail
dims dropped from scoring). Rows already carry every per-dim score as
`pr_{dim}_score` and every hard rule as `pr_{rule}_violated`, so the new
headline is pure arithmetic:

    new = 0.8*primary + 0.2*mean(secondaries)   (primary alone if no secondary)
    new = 0.0 if any APPLICABLE hard rule violated
    primary missing -> mean of available applicable dims (pr.score fallback)

Mirrors personalization_rubric.score()'s aggregation tail exactly; the dim
roles come live from APPLICABILITY / _PRIMARY_POSITIVE_OVERRIDE so this script
never drifts from the rubric module.

Updates pr_query_score_0_10 / pr_combined_personalization_score /
pr_primary_dim / pr_primary_dim_score in metrics_json IN PLACE (tmp+replace),
stamps pr_rubric_version, and takes a one-time backup per file.

Usage:
    python scripts/recompute_pr_headline.py --results_dirs results/agent_tools_sonnet4.6,results/mem0_gpt5.5
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10 ** 9)

from evaluation.personalization_rubric import (  # noqa: E402
    APPLICABILITY, HARD_RULE_DIMS, PENALTY_CHECKS, POSITIVE_DIMS,
    _PRIMARY_POSITIVE_OVERRIDE,
)
from evaluation.task_registry import normalize_task_type  # noqa: E402

RUBRIC_VERSION = "v3_focused_20260612"
COLS = ["query_id", "seq", "user_id", "task_type", "ts", "metrics_json",
        "status", "duration_ms", "error", "agent_response"]


def recompute_row(task_type: str, m: dict) -> bool:
    """Recompute headline keys in metrics dict m. Returns True if updated."""
    tt = normalize_task_type(task_type)
    applicable = APPLICABILITY.get(tt)
    if not applicable or m.get("pr_combined_personalization_score") is None:
        return False
    pos = [d for d in applicable if applicable[d] and d in POSITIVE_DIMS]
    hard = [d for d in applicable if applicable[d] and d in HARD_RULE_DIMS]
    if not pos:
        return False
    primary = _PRIMARY_POSITIVE_OVERRIDE.get(tt, pos[0])
    if primary not in pos:
        primary = pos[0]

    def dim(d):
        v = m.get(f"pr_{d}_score")
        return float(v) if isinstance(v, (int, float)) else None

    p = dim(primary)
    secs = [dim(d) for d in pos if d != primary]
    secs = [s for s in secs if s is not None]
    if p is not None:
        final = 0.8 * p + 0.2 * (sum(secs) / len(secs)) if secs else p
    else:
        avail = [dim(d) for d in pos]
        avail = [a for a in avail if a is not None]
        if not avail:
            return False  # none of the focused dims were ever judged
        final = sum(avail) / len(avail)

    # Penalty checks: deduction = weight × (10 − stored score)/10. Stored rows
    # were judged with the avoidance-framed 0-10 dims (10 = clean), so the
    # historical per-dim scores convert directly into deductions. Missing
    # stored score → no deduction (benefit of the doubt).
    penalty_points = 0.0
    for d, w in PENALTY_CHECKS.get(tt, {}).items():
        s = dim(d)
        if s is not None:
            penalty_points += float(w) * (10.0 - s) / 10.0
    m["pr_penalty_points"] = round(penalty_points, 2)
    final = max(0.0, float(final) - penalty_points)

    violated = any(m.get(f"pr_{r}_violated") in (1, 1.0, True, "True", "true")
                   for r in hard)
    score = 0.0 if violated else round(float(final), 2)
    m["pr_primary_dim"] = primary
    m["pr_primary_dim_score"] = p
    m["pr_query_score_0_10"] = score
    m["pr_combined_personalization_score"] = score
    m["pr_combined_max_possible"] = 10.0
    m["pr_rubric_version"] = RUBRIC_VERSION
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dirs", required=True,
                    help="Comma-separated results dirs (each containing {uid}/results.csv)")
    ap.add_argument("--backup_root", default="results/_prerefocus_backup_20260612")
    args = ap.parse_args()

    for rd in [Path(d.strip()) for d in args.results_dirs.split(",") if d.strip()]:
        for rfile in sorted(rd.glob("*/results.csv")):
            uid = rfile.parent.name
            rows = list(csv.DictReader(open(rfile)))
            n_upd = 0
            for row in rows:
                try:
                    m = json.loads(row.get("metrics_json") or "{}")
                except Exception:
                    continue
                if recompute_row(row.get("task_type") or "", m):
                    row["metrics_json"] = json.dumps(m, ensure_ascii=False)
                    n_upd += 1
            if not n_upd:
                print(f"[recompute] {rfile}: 0 pr rows — skipped")
                continue
            bdir = Path(args.backup_root) / rd.name / uid
            bdir.mkdir(parents=True, exist_ok=True)
            if not (bdir / "results.csv").exists():
                shutil.copy2(rfile, bdir / "results.csv")
            tmp = str(rfile) + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                w.writeheader()
                for row in rows:
                    w.writerow(row)
            Path(tmp).replace(rfile)
            print(f"[recompute] {rfile}: {n_upd} rows -> {RUBRIC_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
