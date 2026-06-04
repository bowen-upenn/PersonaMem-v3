#!/usr/bin/env python3
"""Run ONLY the new Step 28 (feed posts generation) + Step 29 (proactive
trigger candidate inference, with new gather helpers) for a single user.

Skips all other pipeline steps — relies on the user's existing backend
being complete. Useful for incrementally adding feed-react + overactive-
check data to a user without re-running the full 29-step pipeline.

Usage:
    python scripts/run_feed_posts_for_user.py --user_id 115 [--model gpt-5.5]

Trending search cache: if backend/{uid}/.trending_search_cache.json
already has trends for the user's months/platforms, the pipeline reads
from it and skips the WebSearch step. Pre-populate the cache manually
to avoid API search costs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--skip_step_28", action="store_true",
                    help="Skip the feed-posts step (only run proactive trigger inference).")
    ap.add_argument("--skip_step_29", action="store_true",
                    help="Skip proactive trigger inference (only run feed-posts).")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    from data_preparation.persona_agent import PersonaAgent
    from query_llm import QueryLLM

    llm_client = QueryLLM(
        {"models": {"llm_model": args.model}},
        rate_limit_per_min=args.rate_limit,
    )

    # Instantiate PersonaAgent against the existing backend. We don't need to
    # call load_interactions() because the two steps we're about to run both
    # read from disk, not from in-memory pipeline state.
    agent = PersonaAgent(
        user_id=args.user_id,
        llm_client=llm_client,
        backend_dir=args.backend_dir,
        verbose=args.verbose,
    )

    if not args.skip_step_28:
        print(f"\n=== Step 28: generate_feed_posts for user {args.user_id} ===\n")
        agent.generate_feed_posts()
    else:
        print("(skipping Step 28)")

    if not args.skip_step_29:
        print(f"\n=== Step 29: infer_proactive_trigger_candidates for user {args.user_id} ===\n")
        agent.infer_proactive_trigger_candidates()
    else:
        print("(skipping Step 29)")

    # Print API usage summary.
    try:
        usage = llm_client.get_usage_totals()
        print(f"\n=== LLM usage ===\n{usage}\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
