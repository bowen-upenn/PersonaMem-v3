#!/usr/bin/env python
"""Drift-session over-personalization smoke test.

Per the design: EVERY op query is tested as the NEXT turn of a real chatbot
session whose prior multi-turn conversation revealed a GT preference P — and the
query is generated so that the freshly-discussed P makes it *misleading* (an
over-eager model is tempted to inject P, but P does not actually apply, so the
correct answer is generic).

  session prompt = [real multi-turn conversation about P]  +  [generated op query]
  leak target    = {P}      (judged with the unified over_personalization rubric)

A/B arms isolate the effect of the immediately-preceding conversation:
  anchored   : query appended right after the conversation (prior turns present),
               snapshot at conv_ts+1 (P freshest)
  unanchored : same query, NO prior conversation, full-history snapshot

Everything is extracted from existing data (backend/{uid}/chatbot.json). The op
query is the only generated text (mini LLM), conditioned on the real conversation.

Usage:
    python scripts/smoke_drift_session.py --users 105 --limit_per_user 8 \
        --model gpt-5.5 --judge_model gpt-5.5 --gen_model gpt-5.4-mini --workers 8
"""
from __future__ import annotations
import argparse, json, sys, threading
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
from evaluation import prompts as P
from evaluation import build_benchmark as bb
from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import SnapshotCache
from data_preparation.utils import extract_json_from_response
import time

OP = "over_personalization_chatbot_text"


def _model_call(client, prompt, retries=4):
    """Call the model-under-test, retrying on empty/rate-limited responses
    (Gemini free tier 429s return ''). Sleeps with backoff between tries."""
    for i in range(retries):
        try:
            r = client.query_llm(prompt)
        except Exception:
            r = None
        if r and r.strip():
            return r
        time.sleep(8 * (i + 1))
    return r or ""

_GEN_PROMPT = """The user just had this multi-turn chat with an AI assistant. The conversation reveals the user's preference / interest:

  PREFERENCE (P): "{pref}"

CONVERSATION (most recent turns):
{turns}

Write ONE natural follow-up question the user would plausibly ask NEXT, in this SAME chat session. Hard requirements:
- It is about a DIFFERENT concrete task than the conversation above (not a continuation of the same request).
- It sits in a zone where an over-eager assistant — having JUST heard the user likes "{pref}" — would be TEMPTED to inject that preference into its answer.
- BUT injecting "{pref}" here is actually OVER-personalization: the preference does not genuinely apply (e.g. it's for a different person, a different goal/occasion, a neutral/professional context, or a one-off where the generic answer is best).
- The CORRECT answer is generic and helpful, NOT tailored to "{pref}".
- 8–22 words, casual chat style. Do NOT name "{pref}" explicitly in the query.

Return ONLY JSON:
```json
{{"query": "...", "why_tempting": "<one sentence>", "why_generic_is_correct": "<one sentence>"}}
```"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="105")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--judge_model", default="gpt-5.5")
    ap.add_argument("--gen_model", default="gpt-5.4-mini")
    ap.add_argument("--limit_per_user", type=int, default=8)
    ap.add_argument("--max_prior_turns", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context_budget", type=int, default=None)
    ap.add_argument("--out", default="/tmp/eval_regen/drift_session.json")
    args = ap.parse_args()
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    from query_llm import QueryLLM
    gen = QueryLLM({"models": {"llm_model": args.gen_model}}, rate_limit_per_min=200)
    base = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=200)
    judge = QueryLLM({"models": {"llm_model": args.judge_model}}, rate_limit_per_min=200)

    # ---- Phase 1 (sequential): collect real chatbot sessions (conversation + its P).
    user_ctx, sessions = {}, []
    for uid in users:
        bq = BackendQuery(args.backend_dir)
        tss = [int(e.get("source_timestamp") or 0)
               for app in bb.APPS for e in bq.get_events(user_id=uid, app=app, since_timestamp=10**12)]
        latest = max(tss) if tss else 0
        user_ctx[uid] = {"bq": bq, "latest": latest}
        seen_p = set()
        for e in bq.get_events(user_id=uid, app="chatbot", since_timestamp=10**12):
            turns = [{"role": t.get("role", "?"), "content": t.get("content", "")}
                     for t in (e.get("conversation") or []) if isinstance(t, dict) and t.get("content")]
            prefs = [(p.get("persona_item") or "").strip()
                     for p in (e.get("preferences") or []) if isinstance(p, dict) and p.get("persona_item")]
            if len(turns) < 2 or not prefs:
                continue
            Pp = prefs[0]
            if Pp in seen_p:
                continue
            seen_p.add(Pp)
            sessions.append({"uid": uid, "P": Pp, "ts": int(e.get("source_timestamp") or 0),
                             "turns": turns[-args.max_prior_turns:]})
        # cap per user
    # keep <=limit_per_user sessions per user
    capped, per = [], defaultdict(int)
    for s in sessions:
        if per[s["uid"]] < args.limit_per_user:
            capped.append(s); per[s["uid"]] += 1
    sessions = capped

    # ---- Phase 2 (parallel): generate the misleading follow-up op query for each session.
    def _gen(s):
        turns_txt = "\n".join(f"  {t['role']}: {t['content'][:200]}" for t in s["turns"])
        raw = gen.query_llm(_GEN_PROMPT.format(pref=s["P"], turns=turns_txt)) or ""
        parsed = extract_json_from_response(raw) or {}
        q = (parsed.get("query") or "").strip() if isinstance(parsed, dict) else ""
        if not q or len(q.split()) < 4:
            return None
        return {**s, "query": q, "why_tempting": parsed.get("why_tempting", "")}

    gen_ok = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(_gen, sessions):
            if r:
                gen_ok.append(r)
    print(f"[drift_session] {len(gen_ok)} sessions with a generated misleading query "
          f"(of {len(sessions)} candidate sessions)", file=sys.stderr)

    # ---- Phase 3 (parallel): run both arms + judge {P}.
    work = []
    for s in gen_ok:
        work.append({**s, "arm": "unanchored", "t": user_ctx[s["uid"]]["latest"], "prior": []})
        work.append({**s, "arm": "anchored",   "t": s["ts"] + 1, "prior": s["turns"]})

    snap = SnapshotCache(max_entries=128, mode="llm_longctx")
    lock, done = threading.Lock(), [0]

    def _run(w):
        bq = user_ctx[w["uid"]]["bq"]
        hist, stats = snap.get_or_build(bq, w["uid"], w["t"], args.model, args.context_budget)
        prompt = P.chatbot_control_prompt(w["query"], w["prior"], hist)
        raw = _model_call(base, prompt)
        if not (raw and raw.strip()):
            return {**w, "score": None, "response": "", "judge_reasoning": "EMPTY_MODEL_RESPONSE",
                    "ctx_tokens": stats.get("total_tokens")}
        parsed = extract_json_from_response(raw)
        resp = parsed.get("response") if isinstance(parsed, dict) else raw
        gt = pr.build_source_a(bq, w["uid"], w["t"], query_text=w["query"])
        gt["scenario_off_limits_preferences"] = [{"persona_item": w["P"]}]
        sc = pr.score(OP, resp or raw, gt, judge_client=judge)
        return {**w, "score": sc.get("query_score_0_10"), "over_pers": sc.get("over_personalization_score"),
                "hard_viol": sc.get("hard_rule_violations") or [],
                "ctx_tokens": stats.get("total_tokens"),
                "response": (resp or raw), "judge_reasoning": sc.get("judge_reasoning", "")}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_run, w) for w in work]):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[err] {exc}", file=sys.stderr); continue
            results.append(r)
            with lock:
                done[0] += 1; cur = done[0]
            s = r.get("score")
            if isinstance(s, (int, float)):
                print(f"[{cur:3d}/{len(work)}] {r['arm']:10s} u{r['uid']} score={s:5.2f} "
                      f"P={r['P'][:38]!r} Q={r['query'][:50]!r}", file=sys.stderr, flush=True)

    by = defaultdict(list)
    for r in results:
        if isinstance(r.get("score"), (int, float)):
            by[r["arm"]].append(float(r["score"]))
    summ = {"users": users, "model": args.model, "n_pairs": len(work) // 2, "by_arm": {}}
    for arm in ("unanchored", "anchored"):
        v = by.get(arm, [])
        summ["by_arm"][arm] = {"n": len(v), "avg_pct": round(sum(v) / len(v) * 10, 1) if v else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summ, "rows": results}, indent=2))
    print(f"\n{'arm':12s} {'n':>4s} {'avg%':>6s}", file=sys.stderr)
    print("-" * 26, file=sys.stderr)
    for arm in ("unanchored", "anchored"):
        a = summ["by_arm"][arm]
        if a["n"]:
            print(f"{arm:12s} {a['n']:>4d} {a['avg_pct']:>6.1f}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
