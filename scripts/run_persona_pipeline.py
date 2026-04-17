#!/usr/bin/env python3
"""
CLI script for running the persona inference pipeline via OpenAI/Azure API.

For Claude Code mode, use Claude Code directly — see README.md.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from data_preparation.main import load_and_group_csv, process_single_user
from data_preparation.visualize import generate_persona_html
from data_preparation import utils


def main():
    parser = argparse.ArgumentParser(
        description="Run the PersonaMem persona pipeline via OpenAI/Azure API."
    )
    parser.add_argument("--input_csv", default="data/test_interactions.csv", help="Path to interactions CSV file (default: data/test_interactions.csv)")
    parser.add_argument("--backend_dir", default="backend", help="Directory for output CSVs (default: backend)")
    parser.add_argument("--user_id", default=None, help="Process only this user_id (default: all users)")
    parser.add_argument("--max_workers", type=int, default=1, help="Max parallel user workers (default: 1 — process users sequentially)")
    parser.add_argument("--parallel", type=int, default=50, help="Parallel LLM API calls per user (default: 50)")
    parser.add_argument("--model", default="gpt-5-chat", help="LLM model name (default: gpt-5-chat, uses AZURE_OPENAI_DEPLOYMENT_NAME from .env)")
    parser.add_argument("--rate_limit", type=int, default=50, help="API rate limit per minute (default: 50)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from query_llm import QueryLLM

    llm_args = {'models': {'llm_model': args.model}}
    llm_client = QueryLLM(llm_args, rate_limit_per_min=args.rate_limit)

    print(f"{utils.Colors.BOLD}PersonaMem Persona Pipeline{utils.Colors.ENDC}")
    print(f"  Input: {args.input_csv}")
    print(f"  Backend: {args.backend_dir}")
    print(f"  User workers: {args.max_workers}")
    print(f"  Parallel API calls: {args.parallel}")
    actual_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or args.model
    print(f"  Model: {actual_model}")
    print()

    grouped = load_and_group_csv(args.input_csv)

    if args.user_id:
        if args.user_id not in grouped:
            print(f"{utils.Colors.FAIL}User {args.user_id} not found in CSV.{utils.Colors.ENDC}")
            sys.exit(1)
        grouped = {args.user_id: grouped[args.user_id]}

    results = []
    if len(grouped) == 1:
        uid, rows = next(iter(grouped.items()))
        result = process_single_user(uid, rows, llm_client, args.backend_dir, args.verbose, args.parallel)
        results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_user, uid, rows, llm_client, args.backend_dir, args.verbose, args.parallel
                ): uid
                for uid, rows in grouped.items()
            }
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"{utils.Colors.FAIL}[User {uid}] Failed: {e}{utils.Colors.ENDC}")
                    results.append({"user_id": uid, "error": str(e)})

    # Generate visualizations
    for result in results:
        uid = result.get("user_id", "")
        if uid and "error" not in result:
            try:
                generate_persona_html(uid, args.backend_dir)
            except Exception as e:
                print(f"{utils.Colors.WARNING}[User {uid}] Visualization failed: {e}{utils.Colors.ENDC}")

    # Summary
    print(f"\n{utils.Colors.BOLD}=== Pipeline Summary ==={utils.Colors.ENDC}")
    for r in results:
        uid = r.get("user_id", "?")
        if "error" in r:
            print(f"  {utils.Colors.FAIL}[{uid}] ERROR: {r['error']}{utils.Colors.ENDC}")
        else:
            print(f"  {utils.Colors.OKGREEN}[{uid}] {r.get('total_cross_referenced', '?')} personas, "
                  f"{r.get('temporal_topics', '?')} temporal topics{utils.Colors.ENDC}")


if __name__ == "__main__":
    main()
