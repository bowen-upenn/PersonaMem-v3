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
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread pool size for parallel judge calls.")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task_type allowlist; only re-judge these "
                         "(intersected with the judge-based scope).")
    ap.add_argument("--out", default="/tmp/eval_regen/rejudge_summary.json")
    ap.add_argument("--write_back", action="store_true",
                    help="Update each re-judged row's metrics_json IN PLACE in "
                         "results.csv (all pr_* keys replaced with the fresh "
                         "pr.score output). A one-time backup of each touched "
                         "results.csv is taken under --backup_root first.")
    ap.add_argument("--backup_root", default="results/_prejudge_backup_20260612")
    ap.add_argument("--dump_prompts", default=None,
                    help="Capture mode: append every judge prompt to this JSONL "
                         "(no LLM calls) for subagent fan-out.")
    ap.add_argument("--replay_map", default=None,
                    help="Replay mode: JSONL of {prompt, raw} subagent answers; "
                         "fed back through pr.score for identical scoring.")
    args = ap.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]

    judge_client = None
    if args.dump_prompts:
        # Capture mode: write every judge prompt pr.score builds (no LLM call),
        # so harness subagents can answer them in parallel. Returns "" so the
        # row produces no score this pass (we only want the prompts).
        import threading as _th
        _dlock = _th.Lock()
        _dfh = open(args.dump_prompts, "a")

        class _CaptureClient:
            def query_llm(self, prompt, *a, **k):
                with _dlock:
                    _dfh.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
                    _dfh.flush()
                return ""
        judge_client = _CaptureClient()
    elif args.replay_map:
        # Replay mode: feed precomputed subagent answers back through pr.score for
        # IDENTICAL parsing/aggregation. Keyed by exact prompt text.
        _rmap = {}
        for _ln in open(args.replay_map):
            _r = json.loads(_ln)
            if _r.get("prompt") is not None:
                _rmap[_r["prompt"]] = _r.get("raw", "") or ""

        class _ReplayClient:
            def query_llm(self, prompt, *a, **k):
                return _rmap.get(prompt, "")
        judge_client = _ReplayClient()
        print(f"[replay] loaded {len(_rmap)} answers", file=sys.stderr)
    elif args.judge:
        jm = args.judge_model
        if jm.startswith("claude") or jm in ("opus", "sonnet", "haiku"):
            # Claude judges have no Azure/API deployment in this env; drive them
            # through `claude -p` (same path the persona-1 judge study used).
            sys.path.insert(0, str(ROOT / "results/_scripts"))
            from run_qa_audit_opus import ClaudeLLM
            alias = {"claude-opus-4.8": "opus", "claude-sonnet-4.6": "sonnet"}.get(jm, jm)
            judge_client = ClaudeLLM(alias)
        else:
            from query_llm import QueryLLM
            judge_client = QueryLLM({"models": {"llm_model": jm}}, rate_limit_per_min=50)

    scope = set(APPLICABILITY.keys())
    # cluster-based fatigue tasks aren't re-judgeable per-row here
    scope -= {"over_personalization_repetition_recsys", "over_personalization_repetition_chatbot",
              "new_suggestions_recsys", "new_suggestions_chatbot"}
    if args.tasks:
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        scope &= want
        print(f"[rejudge] task filter -> {sorted(scope)}", file=sys.stderr)

    # task_type -> list of query_score_0_10
    by_task: dict[str, list[float]] = defaultdict(list)
    per_task_seen: dict[str, int] = defaultdict(int)
    n_total = n_scored = n_skip = 0

    # Phase 1 (sequential): collect work items, pre-building ground truth.
    # GT building reads backend JSON, so keep it off the worker threads (the
    # parallel section then only does the judge LLM call + scoring).
    work: list[tuple] = []  # (uid, tt, qid, resp, gt)
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
            try:
                gt = _build_gt(bq, uid, inst)
            except Exception as exc:
                print(f"[err-gt] {uid} {qid} {tt}: {exc}", file=sys.stderr, flush=True)
                continue
            # Runner-path parity: agentic runners score the EXTRACTED compose
            # text (final_answer/response/summary/reply_to_user), not the raw
            # JSON-wrapped agent output (agentic_tasks.py:328-341). Mirror that
            # here or the judge sees a JSON wrapper the runner never scored.
            if tt.startswith("agentic_"):
                from data_preparation.utils import extract_json_from_response
                parsed = extract_json_from_response(resp)
                if isinstance(parsed, dict):
                    picked = (parsed.get("final_answer") or parsed.get("response")
                              or parsed.get("summary") or parsed.get("reply_to_user")
                              or resp or "")
                else:
                    picked = resp or ""
                resp = picked if isinstance(picked, str) else json.dumps(picked, ensure_ascii=False)
            work.append((uid, tt, qid, resp, gt))
    n_total = len(work)
    print(f"[rejudge] {n_total} rows to score with {args.workers} workers", file=sys.stderr)

    # Phase 2 (parallel): the judge LLM call dominates wall-time. Fan out over a
    # thread pool — QueryLLM's internal rate limiter (50/min) still caps the
    # request rate; threads just hide the per-call latency.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def _score_one(item):
        uid, tt, qid, resp, gt = item
        try:
            out = pr.score(tt, resp, gt, judge_client=judge_client)
            return uid, tt, qid, out, None
        except Exception as exc:
            return uid, tt, qid, None, str(exc)

    rescored: dict[str, dict[str, dict]] = defaultdict(dict)  # uid -> qid -> pr out
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_score_one, it) for it in work]
        for fut in as_completed(futs):
            uid, tt, qid, out, err = fut.result()
            if err is not None:
                print(f"[err] {uid} {qid} {tt}: {err}", file=sys.stderr, flush=True)
                continue
            s = out.get("query_score_0_10")
            if isinstance(s, (int, float)):
                with lock:
                    by_task[tt].append(float(s))
                    rescored[uid][qid] = out
                    n_scored += 1
                    cur = n_scored
                prim = out.get("primary_dim")
                viol = out.get("hard_rule_violations") or []
                print(f"[{cur:4d}/{n_total}] u{uid} {tt:38s} score={s:5.2f} "
                      f"primary={prim}={out.get('primary_dim_score')} "
                      f"{'VIOL:'+','.join(viol) if viol else ''}",
                      file=sys.stderr, flush=True)

    # Write-back: replace each re-judged row's pr_* metrics in results.csv.
    # Backup is one-time (cp -n semantics); rewrite is tmp + os.replace so a
    # crash mid-write never truncates the live file.
    if args.write_back and rescored:
        import shutil
        for uid, updates in sorted(rescored.items()):
            rfile = Path(args.results_dir) / uid / "results.csv"
            bdir = Path(args.backup_root) / Path(args.results_dir).name / uid
            bdir.mkdir(parents=True, exist_ok=True)
            if not (bdir / "results.csv").exists():
                shutil.copy2(rfile, bdir / "results.csv")
            rows_all = list(csv.DictReader(open(rfile)))
            cols = ["query_id", "seq", "user_id", "task_type", "ts", "metrics_json",
                    "status", "duration_ms", "error", "agent_response"]
            n_upd = 0
            for row in rows_all:
                out = updates.get(row.get("query_id"))
                if out is None:
                    continue
                try:
                    m = json.loads(row.get("metrics_json") or "{}")
                except Exception:
                    m = {}
                # Strip stale pr_* (dims dropped by the rubric must not linger),
                # then merge the fresh scalar copies — same encoding as the
                # runners ({f"pr_{k}": v for scalar v}).
                m = {k: v for k, v in m.items() if not k.startswith("pr_")}
                m.update({f"pr_{k}": v for k, v in out.items()
                          if isinstance(v, (int, float, str))})
                row["metrics_json"] = json.dumps(m, ensure_ascii=False)
                n_upd += 1
            tmp = str(rfile) + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for row in rows_all:
                    w.writerow(row)
            os.replace(tmp, rfile)
            print(f"[write_back] {rfile}: {n_upd} rows updated "
                  f"(backup: {bdir / 'results.csv'})", file=sys.stderr, flush=True)

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

    # Per-item sidecar (for inter-judge agreement metrics): one row per scored
    # item with its query_score_0_10, keyed by uid+qid+task.
    if rescored:
        side = str(args.out) + ".items.jsonl"
        with open(side, "w") as fh:
            for uid, qmap in sorted(rescored.items()):
                for qid, o in qmap.items():
                    s = o.get("query_score_0_10")
                    if isinstance(s, (int, float)):
                        fh.write(json.dumps({"uid": uid, "qid": qid,
                                 "task": o.get("task_id"), "score": float(s)}) + "\n")

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
