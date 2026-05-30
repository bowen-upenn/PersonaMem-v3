#!/usr/bin/env python3
"""Re-run ONLY Step 28 (generate_feed_posts) + Step 29
(infer_proactive_trigger_candidates) for a batch of users, with bounded
concurrency. Reads each user's pre-written backend/{uid}/.trending_search_cache.json
(see scripts/build_trending_cache.py) so Step 28 injects trending feed content
on the cache hit — no live web_search needed.

Why a batch driver: trending content + complete proactive candidates live in
persona DATA, so they must be (re)generated per user. generate_feed_posts is
idempotent (drops prior feed_visible events, regenerates friend + trending);
infer_proactive_trigger_candidates re-derives all trigger types. Each worker
gets its OWN QueryLLM with a low rate limit so the aggregate stays well under
Azure throttling at any concurrency.

Usage:
    python scripts/run_feed_backfill_batch.py --user_ids 1 2 105 ... \
        --workers 5 --rate_limit 12 --model gpt-5-chat
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass


def _backfill_one(uid: str, backend_dir: str, model: str, rate_limit: int,
                  skip_28: bool, skip_29: bool) -> tuple:
    from data_preparation.persona_agent import PersonaAgent
    from query_llm import QueryLLM
    t0 = time.time()
    try:
        client = QueryLLM({"models": {"llm_model": model}},
                          rate_limit_per_min=rate_limit)
        agent = PersonaAgent(user_id=str(uid), llm_client=client,
                             backend_dir=backend_dir, verbose=False)
        if not skip_28:
            agent.generate_feed_posts()
        if not skip_29:
            agent.infer_proactive_trigger_candidates()
        return (uid, "ok", round(time.time() - t0, 1), None)
    except Exception as e:  # noqa: BLE001
        import traceback
        return (uid, "FAIL", round(time.time() - t0, 1),
                f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user_ids", nargs="+", required=True)
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5-chat")
    ap.add_argument("--rate_limit", type=int, default=12,
                    help="Per-worker QueryLLM rate limit (workers*rate_limit "
                         "is the aggregate ceiling).")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip_step_28", action="store_true")
    ap.add_argument("--skip_step_29", action="store_true")
    args = ap.parse_args()

    uids = [str(u) for u in args.user_ids]
    print(f"[feed-backfill] {len(uids)} users, workers={args.workers}, "
          f"rate_limit={args.rate_limit}/worker, model={args.model}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_backfill_one, u, args.backend_dir, args.model,
                          args.rate_limit, args.skip_step_28, args.skip_step_29): u
                for u in uids}
        for fut in as_completed(futs):
            uid, status, secs, err = fut.result()
            results.append((uid, status))
            print(f"[feed-backfill] user {uid} -> {status} ({secs}s)"
                  + (f"\n    {err}" if err else ""), flush=True)

    ok = sum(1 for _, s in results if s == "ok")
    fails = [u for u, s in results if s != "ok"]
    print(f"[feed-backfill] DONE: {ok}/{len(uids)} ok"
          + (f"; FAILED: {fails}" if fails else ""), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
