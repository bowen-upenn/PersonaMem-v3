#!/usr/bin/env python3
"""Regenerate DM threads for a user using the new seed-anchored generator.

Walks backend/{uid}/{instagram,facebook,threads}.json and replaces the
existing DM-thread events with new ones produced by
data_preparation/extension_b/dm_threads.generate_dm_threads — which
anchors each thread to a real seed event and embeds forwarded content
in the messages.

Per-user cost: ~25 LLM calls (one per thread). Uses the same Azure
OpenAI deployment configured for the persona pipeline (env vars
AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT_NAME).

Usage:
  python scripts/regen_dms_with_real_seeds.py --user_id 115
  python scripts/regen_dms_with_real_seeds.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load .env so QueryLLM picks up Azure / OpenAI creds.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(_ROOT) / ".env", override=False)
except Exception:
    pass


def _build_llm_client():
    from query_llm import QueryLLM
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-5.5"
    client = QueryLLM({"models": {"llm_model": model}}, rate_limit_per_min=120)
    print(f"[regen_dms] LLM client ready (model={model})")
    return client


def regen_user(user_id: str, backend_dir: str, llm_client) -> dict:
    from data_preparation.extension_b.dm_threads import generate_dm_threads
    user_dir = Path(backend_dir) / str(user_id)
    profile = json.loads((user_dir / "profile.json").read_text())
    friends = profile.get("friends") or []

    stats = {"user_id": user_id, "by_app": {}}
    for app in ("instagram", "facebook", "threads"):
        path = user_dir / f"{app}.json"
        if not path.exists():
            continue
        evs = json.loads(path.read_text())
        # Strip existing DMs from the event list.
        non_dm_events = [e for e in evs if not e.get("is_dm")]
        n_old_dms = len(evs) - len(non_dm_events)

        # Generate fresh DMs anchored to real seed events.
        new_dms = generate_dm_threads(
            user_id=user_id,
            app=app,
            profile=profile,
            friends=friends,
            existing_events=non_dm_events,
            llm_client=llm_client,
            rng_seed=hash(f"{user_id}:{app}") % (2**31),
        )

        # Append + sort + write back.
        merged = sorted(non_dm_events + new_dms,
                        key=lambda e: int(e.get("source_timestamp") or 0))
        path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        stats["by_app"][app] = {"old": n_old_dms, "new": len(new_dms)}
        print(f"[{user_id}] {app}: replaced {n_old_dms} DMs → {len(new_dms)} new (seed-anchored)")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--user_id")
    grp.add_argument("--all", action="store_true")
    p.add_argument("--backend_dir", default="backend")
    args = p.parse_args()

    if args.all:
        backend = Path(args.backend_dir)
        user_ids = sorted(d.name for d in backend.iterdir()
                          if d.is_dir() and not d.name.startswith("_"))
    else:
        user_ids = [args.user_id]

    llm = _build_llm_client()
    for uid in user_ids:
        regen_user(uid, args.backend_dir, llm)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
