#!/usr/bin/env python
"""Re-judge saved eval responses with the NEW unified scoring (pr.score).

Cheap path: reuses the model responses already in
`results/llm_longctx/{uid}/results.csv` (no re-running the model under test),
rebuilds each instance's ground truth the same way the runners do
(`personalization_rubric.build_source_a`), and re-scores with the new unified
judge. Aggregates `query_score_0_10` per task_type across personas.

Only tasks that flow through `pr.score` (i.e. present in
`personalization_rubric.APPLICABILITY`) are re-judged — others are unchanged by
the scoring redesign. Repetition tasks (cluster-based fatigue) are skipped.

Usage:
    python scripts/rejudge_existing.py --users 105,115,209,229,282 \
        [--judge_model gpt-5.5] [--no-judge] [--limit_per_task N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from evaluation import personalization_rubric as pr
from evaluation.backend_query import BackendQuery
from evaluation.personalization_rubric import APPLICABILITY


def _build_gt(bq, uid, inst):
    """Rebuild the ground truth a runner would pass to pr.score."""
    full = inst.get("instance_full") or {}
    tt = inst.get("task_type")
    ts = int(inst.get("ts") or full.get("t_test") or full.get("source_timestamp") or 0)
    user_query = inst.get("user_query") or full.get("user_query") or full.get("query") or ""
    hashtags = full.get("source_hashtags") or inst.get("source_hashtags") or []

    if tt == "over_personalization_context_shift":
        q = full.get("query") or user_query
        gt = pr.build_source_a(bq, uid, ts, query_text=q)
        gt["query_text"] = f'{q}\n\n[Scenario — {full.get("name","")}: {full.get("notes","")}]'
        forbidden = [it for it in (full.get("forbidden_items") or []) if it.get("persona_item")]
        if forbidden:
            gt["scenario_off_limits_preferences"] = [
                {"persona_item": it.get("persona_item", ""), "category": it.get("category", "")}
                for it in forbidden
            ]
        if full.get("carve_out"):
            gt["user_opted_out_topic"] = full["carve_out"].get("topic", "")
        return gt
    return pr.build_source_a(bq, uid, ts, query_text=user_query, query_hashtags=hashtags)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="105,115,209,229,282")
    ap.add_argument("--results_dir", default="results/llm_longctx")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--judge_model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-5.5"))
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="Plumbing dry-run: rebuild GT but don't call the judge LLM.")
    ap.add_argument("--limit_per_task", type=int, default=None)
    ap.add_argument("--out", default="/tmp/eval_regen/rejudge_summary.json")
    args = ap.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]

    judge_client = None
    if args.judge:
        from query_llm import QueryLLM
        judge_client = QueryLLM({"models": {"llm_model": args.judge_model}}, rate_limit_per_min=50)

    scope = set(APPLICABILITY.keys())
    # cluster-based fatigue tasks aren't re-judgeable per-row here
    scope -= {"over_personalization_repetition_recsys", "over_personalization_repetition_chatbot",
              "new_suggestions_recsys", "new_suggestions_chatbot"}

    # task_type -> list of query_score_0_10
    by_task: dict[str, list[float]] = defaultdict(list)
    per_task_seen: dict[str, int] = defaultdict(int)
    n_total = n_scored = n_skip = 0

    for uid in users:
        rfile = Path(args.results_dir) / uid / "results.csv"
        tfile = Path(args.backend_dir) / uid / "test.json"
        if not rfile.exists() or not tfile.exists():
            print(f"[skip] user {uid}: missing results or test.json", file=sys.stderr)
            continue
        saved = {}
        for row in csv.DictReader(open(rfile)):
            if (row.get("status") or "").strip() == "ok" and row.get("agent_response"):
                saved[row["query_id"]] = row["agent_response"]
        instances = json.load(open(tfile))
        bq = BackendQuery(args.backend_dir)

        for inst in instances:
            tt = inst.get("task_type")
            if tt not in scope:
                continue
            qid = inst.get("query_id")
            resp = saved.get(qid)
            if not resp:
                n_skip += 1
                continue
            if args.limit_per_task and per_task_seen[f"{uid}:{tt}"] >= args.limit_per_task:
                continue
            per_task_seen[f"{uid}:{tt}"] += 1
            n_total += 1
            try:
                gt = _build_gt(bq, uid, inst)
                out = pr.score(tt, resp, gt, judge_client=judge_client)
                s = out.get("query_score_0_10")
                if isinstance(s, (int, float)):
                    by_task[tt].append(float(s))
                    n_scored += 1
                    prim = out.get("primary_dim")
                    viol = out.get("hard_rule_violations") or []
                    print(f"[{n_scored:4d}] u{uid} {tt:38s} score={s:5.2f} "
                          f"primary={prim}={out.get('primary_dim_score')} "
                          f"{'VIOL:'+','.join(viol) if viol else ''}",
                          file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[err] {uid} {qid} {tt}: {exc}", file=sys.stderr, flush=True)
        print(f"[done] user {uid}: scored so far {n_scored}/{n_total}", file=sys.stderr)

    # Aggregate
    rows = []
    for tt in sorted(by_task):
        vals = by_task[tt]
        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append({"task_type": tt, "n": len(vals),
                     "avg_score_0_10": round(avg, 2), "avg_pct": round(avg * 10, 1)})
    all_vals = [v for vs in by_task.values() for v in vs]
    micro = (sum(all_vals) / len(all_vals)) if all_vals else 0.0
    macro = (sum(r["avg_score_0_10"] for r in rows) / len(rows)) if rows else 0.0
    summary = {"users": users, "n_scored": n_scored, "n_skipped_no_response": n_skip,
               "judge": bool(args.judge), "judge_model": args.judge_model,
               "micro_avg_0_10": round(micro, 2), "micro_avg_pct": round(micro * 10, 1),
               "macro_avg_0_10": round(macro, 2), "macro_avg_pct": round(macro * 10, 1),
               "by_task": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(f"\n{'task_type':42s} {'n':>4s} {'avg/10':>7s} {'avg%':>6s}")
    print("-" * 64)
    for r in rows:
        print(f"{r['task_type']:42s} {r['n']:>4d} {r['avg_score_0_10']:>7.2f} {r['avg_pct']:>6.1f}")
    print("-" * 64)
    print(f"{'MICRO (row-weighted)':42s} {n_scored:>4d} {micro:>7.2f} {micro*10:>6.1f}")
    print(f"{'MACRO (task-weighted)':42s} {len(rows):>4d} {macro:>7.2f} {macro*10:>6.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
