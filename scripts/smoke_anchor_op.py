#!/usr/bin/env python
"""A/B smoke test: does ANCHORING an over-personalization probe right after the
most-misleading preference make the model over-personalize (genuinely harder)?

No regen needed — everything is EXTRACTED FROM EXISTING DATA:
  * op queries come from the existing `over_personalization_chatbot_text`
    instances in `backend/{uid}/test.json`,
  * the "most-misleading" pref P is picked by the mini LLM from the user's real
    positive preferences (each carries its expression `source_timestamp`),
  * the anchored context is just the time-masked snapshot at `t = P_ts + 1`
    (P becomes the freshest thing the model sees; later events are hidden),
  * the unanchored context is the snapshot at the probe's original timestamp
    (full history, the current setup).

Both arms run the SAME model under test (llm_longctx) and are judged against
`{P}` with the unified rubric (over_personalization primary; avoid_leak /
privacy_leak hard rules). The contrast isolates the effect of P-freshness.

Usage:
    python scripts/smoke_anchor_op.py --users <id,id,...> \
        --model gpt-5.5 --judge_model gpt-5.5 --pick_model gpt-5.4-mini \
        --limit_per_user 6 --workers 6 --out /tmp/eval_regen/anchor_ab.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from evaluation import personalization_rubric as pr
from evaluation import prompts as prompts_mod
from evaluation import build_benchmark as bb
from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import SnapshotCache
from data_preparation.utils import extract_json_from_response

OP_TASK = "over_personalization_chatbot_text"


def _parse_response(raw: str) -> str:
    parsed = extract_json_from_response(raw)
    if isinstance(parsed, dict):
        return parsed.get("response") or raw
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="comma-separated persona ids")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5", help="Model under test (llm_longctx baseline).")
    ap.add_argument("--judge_model", default="gpt-5.5")
    ap.add_argument("--pick_model", default="gpt-5.4-mini", help="Mini LLM that picks the most-misleading pref.")
    ap.add_argument("--limit_per_user", type=int, default=6)
    ap.add_argument("--max_prior_turns", type=int, default=6,
                    help="Max chatbot conversation turns to prepend as the anchored prior context.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--context_budget", type=int, default=None)
    ap.add_argument("--max_anchor_gap_frac", type=float, default=1.0,
                    help="Only anchor when P's last occurrence is within this fraction "
                         "of the timeline from the end (1.0 = no recency filter). Controls "
                         "the context-truncation confound.")
    ap.add_argument("--out", default="/tmp/eval_regen/anchor_ab.json")
    args = ap.parse_args()
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    from query_llm import QueryLLM
    pick_llm = QueryLLM({"models": {"llm_model": args.pick_model}}, rate_limit_per_min=200)
    baseline = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=200)
    judge = QueryLLM({"models": {"llm_model": args.judge_model}}, rate_limit_per_min=200)

    # ---- Phase 1 (sequential, fast): per-user context (pre-warm BackendQuery
    # so worker threads only do dict reads) + collect the op pick-tasks.
    user_ctx: dict[str, dict] = {}
    pick_tasks: list[dict] = []
    for uid in users:
        bq = BackendQuery(args.backend_dir)
        tfile = Path(args.backend_dir) / uid / "test.json"
        if not tfile.exists():
            print(f"[skip] {uid}: no test.json", file=sys.stderr); continue
        instances = json.load(open(tfile))
        tss = [int(e.get("source_timestamp") or 0)
               for app in bb.APPS for e in bq.get_events(user_id=uid, app=app, since_timestamp=10**12)]
        latest = max(tss) if tss else 0
        span = max(1, latest - (min(tss) if tss else 0))
        pos = bq.get_preferences(uid, latest + 1, polarity="positive")
        p_last_ts = defaultdict(int)
        for p in pos:
            pi = (p.get("persona_item") or "").strip()
            if pi:
                p_last_ts[pi] = max(p_last_ts[pi], int(p.get("source_timestamp") or 0))
        # Index the user's real chatbot conversation events so an op query can be
        # placed right AFTER the chat where pref P was discussed (the salient
        # over-personalization trap).
        chat_convs = []
        for e in bq.get_events(user_id=uid, app="chatbot", since_timestamp=10**12):
            turns = [{"role": t.get("role", "?"), "content": t.get("content", "")}
                     for t in (e.get("conversation") or []) if isinstance(t, dict)]
            if not turns:
                continue
            chat_convs.append({
                "ts": int(e.get("source_timestamp") or 0),
                "tags": {str(h).lstrip("#").lower() for h in (e.get("source_hashtags") or [])},
                "items": {(p.get("persona_item") or "").strip()
                          for p in (e.get("preferences") or []) if isinstance(p, dict)},
                "turns": turns,
            })
        user_ctx[uid] = {"bq": bq, "latest": latest, "span": span, "pos": pos,
                         "p_last_ts": p_last_ts, "chat_convs": chat_convs}
        op = [i for i in instances if i.get("task_type") == OP_TASK and i.get("user_query")][:args.limit_per_user]
        for n, inst in enumerate(op):
            pick_tasks.append({"uid": uid, "query": inst["user_query"],
                               "orig_ts": int(inst.get("source_timestamp") or latest),
                               "qid": inst.get("query_id") or inst.get("source_object_id") or f"{uid}:{n}"})

    # ---- Phase 2 (parallel): pick the most-misleading P per op query.
    skipped_no_P = 0
    lock = threading.Lock()

    def _match_conv(ctx, P):
        """Best chatbot conversation that discussed P: prefer one whose embedded
        preferences include P (exact), else max hashtag overlap with P's tags.
        Returns the conv dict or None."""
        pi = P["persona_item"].strip()
        ptags = {str(h).lstrip("#").lower() for h in (P.get("source_hashtags") or [])}
        exact = [c for c in ctx["chat_convs"] if pi in c["items"]]
        if exact:
            return max(exact, key=lambda c: c["ts"])
        if ptags:
            scored = [(len(ptags & c["tags"]), c["ts"], c) for c in ctx["chat_convs"] if ptags & c["tags"]]
            if scored:
                return max(scored, key=lambda x: (x[0], x[1]))[2]
        return None

    def _pick(task):
        ctx = user_ctx[task["uid"]]
        shortlist = bb._shortlist_candidate_prefs(task["query"], [], ctx["pos"])
        P = bb._pick_misleading_pref_for_query(pick_llm, task["query"], shortlist)
        if not P or not P.get("persona_item"):
            return ("no_P", None)
        # carry P's hashtags (from the picked occurrence) for conv matching
        Pfull = {"persona_item": P["persona_item"], "category": P.get("category", ""),
                 "source_hashtags": P.get("source_hashtags", [])}
        conv = _match_conv(ctx, Pfull)
        if conv is None:
            return ("no_conv", None)
        p_ts = ctx["p_last_ts"].get(P["persona_item"].strip(), int(P.get("source_timestamp") or 0))
        return ("ok", {**task, "P": {"persona_item": Pfull["persona_item"], "category": Pfull["category"]},
                       "latest_ts": ctx["latest"], "p_ts": p_ts,
                       "conv_ts": conv["ts"], "conv_turns": conv["turns"][-args.max_prior_turns:],
                       "gap_frac": round((ctx["latest"] - conv["ts"]) / ctx["span"], 3)})

    picks: list[dict] = []
    skip_reasons = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for status, r in pool.map(_pick, pick_tasks):
            if r is None:
                skip_reasons[status] += 1
            else:
                picks.append(r)

    work: list[dict] = []
    for pk in picks:
        # AN: op query placed right AFTER the chatbot conversation about P
        #     (conversation is the salient prior context). UN: full history,
        #     no prior conversation — the current setup.
        work.append({**pk, "arm": "unanchored", "t": pk["orig_ts"], "prior": []})
        work.append({**pk, "arm": "anchored",   "t": pk["conv_ts"] + 1, "prior": pk["conv_turns"]})
    print(f"[anchor_ab] {len(work)} work items "
          f"({len(work)//2} A/B pairs across {len(users)} users; "
          f"skipped: {dict(skip_reasons)})", file=sys.stderr)

    # ---- Phase 3 (parallel): build context, run model, judge against {P}.
    snapcache = SnapshotCache(max_entries=128, mode="llm_longctx")
    done = [0]

    def _run(item):
        uid, t = item["uid"], item["t"]
        bq = user_ctx[uid]["bq"]
        hist, stats = snapcache.get_or_build(bq, uid, t, args.model, args.context_budget)
        prompt = prompts_mod.chatbot_control_prompt(item["query"], item.get("prior") or [], hist)
        raw = baseline.query_llm(prompt) or ""
        resp = _parse_response(raw)
        gt = pr.build_source_a(bq, uid, t, query_text=item["query"])
        gt["scenario_off_limits_preferences"] = [item["P"]]
        sc = pr.score(OP_TASK, resp, gt, judge_client=judge)
        out = dict(item)
        out["score"] = sc.get("query_score_0_10")
        out["over_pers"] = sc.get("over_personalization_score")
        out["hard_viol"] = sc.get("hard_rule_violations") or []
        out["ctx_tokens"] = stats.get("total_tokens")
        out["response"] = resp
        out["judge_reasoning"] = sc.get("judge_reasoning", "")
        return out

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run, it) for it in work]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[err] {exc}", file=sys.stderr); continue
            results.append(r)
            with lock:
                done[0] += 1
                cur = done[0]
            s = r.get("score")
            if isinstance(s, (int, float)):
                print(f"[{cur:3d}/{len(work)}] {r['arm']:10s} u{r['uid']} score={s:5.2f} "
                      f"prior={len(r.get('prior') or [])}turns ctx={r['ctx_tokens']:>7} "
                      f"P={r['P']['persona_item'][:42]!r}",
                      file=sys.stderr, flush=True)

    # ---- Aggregate.
    by_arm = defaultdict(list)
    for r in results:
        if isinstance(r.get("score"), (int, float)):
            by_arm[r["arm"]].append(float(r["score"]))
    summary = {"users": users, "model": args.model, "judge_model": args.judge_model,
               "n_pairs": len(work) // 2, "by_arm": {}}
    for arm in ("unanchored", "anchored"):
        v = by_arm.get(arm, [])
        summary["by_arm"][arm] = {"n": len(v),
                                  "avg_0_10": round(sum(v) / len(v), 2) if v else None,
                                  "avg_pct": round(sum(v) / len(v) * 10, 1) if v else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": results}, indent=2))

    print(f"\n{'arm':12s} {'n':>4s} {'avg/10':>7s} {'avg%':>6s}", file=sys.stderr)
    print("-" * 34, file=sys.stderr)
    for arm in ("unanchored", "anchored"):
        a = summary["by_arm"][arm]
        if a["n"]:
            print(f"{arm:12s} {a['n']:>4d} {a['avg_0_10']:>7.2f} {a['avg_pct']:>6.1f}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
