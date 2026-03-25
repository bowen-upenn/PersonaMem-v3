"""
Library entrypoint for the persona pipeline.

Provides functions for:
- Loading and grouping CSV data by user_id
- Processing users via API (parallel threads)
"""

from __future__ import annotations

import os
import sys
import csv
from collections import defaultdict
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_preparation.persona_agent import PersonaAgent
from data_preparation import utils, prompts


def load_and_group_csv(csv_path: str) -> dict[str, list[dict]]:
    """Load a CSV file and group rows by user_id."""
    grouped = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id", "")
            if uid:
                grouped[uid].append(row)
    print(f"{utils.Colors.OKBLUE}Loaded {sum(len(v) for v in grouped.values())} rows for {len(grouped)} users from {csv_path}{utils.Colors.ENDC}")
    return dict(grouped)


def process_single_user(
    user_id: str,
    rows: list[dict],
    llm_client,
    backend_dir: str = "backend",
    verbose: bool = False,
) -> dict:
    """Create a PersonaAgent for one user and run the full pipeline."""
    agent = PersonaAgent(
        user_id=user_id,
        llm_client=llm_client,
        backend_dir=backend_dir,
        verbose=verbose,
    )
    agent.load_interactions(rows)
    return agent.run_pipeline()


def process_all_users_api(
    csv_path: str,
    llm_client,
    backend_dir: str = "backend",
    max_workers: int = 4,
    verbose: bool = False,
) -> list[dict]:
    """Process all users in parallel using ThreadPoolExecutor (API mode)."""
    grouped = load_and_group_csv(csv_path)
    results = []

    if len(grouped) == 1:
        # Single user — no need for thread pool
        uid, rows = next(iter(grouped.items()))
        result = process_single_user(uid, rows, llm_client, backend_dir, verbose)
        results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_user, uid, rows, llm_client, backend_dir, verbose
                ): uid
                for uid, rows in grouped.items()
            }
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"{utils.Colors.FAIL}[User {uid}] Pipeline failed: {e}{utils.Colors.ENDC}")
                    results.append({"user_id": uid, "error": str(e)})

    return results


