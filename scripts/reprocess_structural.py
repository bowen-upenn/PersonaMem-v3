#!/usr/bin/env python3
"""
Reprocess existing persona data with structural pipeline changes.

Applies new output format (interaction events with nested preferences),
session-based app routing, and updated visualization — WITHOUT requiring
LLM API calls. Uses existing processed data as the base.

For full re-inference (including re-running LLM steps), use
run_persona_pipeline.py with an API key.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from data_preparation.persona_agent import (
    PersonaAgent, SESSION_GAP_SECONDS, PLATFORMS, _normalize_persona_text,
    CrossReferencedPersona, AtomicPersona, InteractionRow,
)
from data_preparation.visualize import generate_persona_html
from data_preparation import utils


def reprocess_user(user_id: str, csv_path: str, backend_dir: str = "backend"):
    """Reprocess a single user: load existing data, apply structural changes, save."""
    print(f"\n{utils.Colors.BOLD}[User {user_id}] Structural reprocessing...{utils.Colors.ENDC}")

    agent = PersonaAgent(user_id=user_id, backend_dir=backend_dir, verbose=True)

    # 1. Load existing processed data (old format)
    if not agent.load_from_backend():
        print(f"{utils.Colors.FAIL}[User {user_id}] No existing data found in {backend_dir}/{user_id}/{utils.Colors.ENDC}")
        return

    print(f"  Loaded: {len(agent.atomic_personas)} positive atomics, "
          f"{len(agent.negative_personas)} negative atomics, "
          f"{len(agent.cross_referenced_personas)} positive canonicals")

    # 2. Load raw CSV for session building
    grouped = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == str(user_id):
                grouped[user_id].append(row)

    if str(user_id) in grouped:
        agent.load_interactions(grouped[str(user_id)])
        print(f"  Raw CSV: {len(agent.interactions)} interaction rows")
    else:
        print(f"  {utils.Colors.WARNING}User {user_id} not found in CSV — skipping session building{utils.Colors.ENDC}")

    # 3. Build sessions from raw interactions
    agent._build_sessions()

    # 4. Build canonical groups from existing atomics
    pos_groups: dict[str, list] = defaultdict(list)
    for ap in agent.atomic_personas:
        key = _normalize_persona_text(ap.persona_item)
        if key:
            pos_groups[key].append(ap)
    agent._canonical_groups = dict(pos_groups)

    neg_groups: dict[str, list] = defaultdict(list)
    for ap in agent.negative_personas:
        key = _normalize_persona_text(ap.persona_item)
        if key:
            neg_groups[key].append(ap)
    agent._negative_canonical_groups = dict(neg_groups)

    # 5. Session-based row-to-app assignment
    # Use existing canonical app assignments for the majority vote
    agent._assign_rows_to_apps()

    # 6. Save in new interaction-event format
    output_dir = agent.save_to_backend()

    # 7. Generate updated visualization
    html_path = generate_persona_html(user_id, backend_dir)

    print(f"\n{utils.Colors.OKGREEN}[User {user_id}] Reprocessing complete!{utils.Colors.ENDC}")
    print(f"  Output: {output_dir}")
    print(f"  HTML:   {html_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Structural reprocessing of existing persona data")
    parser.add_argument("--user_id", required=True, help="User ID to reprocess")
    parser.add_argument("--csv", default="data/gistbench_sample_10users.csv", help="Path to raw CSV")
    parser.add_argument("--backend_dir", default="backend", help="Backend directory")
    args = parser.parse_args()

    reprocess_user(args.user_id, args.csv, args.backend_dir)


if __name__ == "__main__":
    main()
