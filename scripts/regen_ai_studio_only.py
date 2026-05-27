#!/usr/bin/env python3
"""Regen ONLY Step 18B (AI Studio conversations) on an existing backend.

Saves ~18 min vs a full pipeline regen by reusing every other backend
artifact. Useful when only the SPT delta scaling, prompt, or LLM model
has changed and the rest of the pipeline output is still valid.

Loads backend/{uid}/ state via PersonaAgent.load_from_backend, then
patches in the fields that load_from_backend skips (hidden_personas,
ai_studio_persona, user_voice, mbti) by reading profile.json directly.
Derives _row_app + _canonical_groups from the loaded events.

Then calls agent.generate_ai_studio_conversations() — which clears
ai_studio.json + ai_studio_memory.json and re-emits them with fresh
SPT pacing. No other backend files are touched.

Usage:
  python scripts/regen_ai_studio_only.py --user_id 115 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from query_llm import QueryLLM
from data_preparation.persona_agent import PersonaAgent
from data_preparation import utils


def _patch_user_profile_from_disk(agent: PersonaAgent, uid: str, backend_dir: str) -> None:
    """load_from_backend only populates 8 UserProfile fields. AI Studio
    generation needs `hidden_personas`, `ai_studio_persona`, `user_voice`,
    and `mbti` too — those live in profile.json. Read them back in."""
    profile_path = Path(backend_dir) / uid / "profile.json"
    with profile_path.open() as f:
        profile = json.load(f)
    agent.user_profile.hidden_personas = profile.get("hidden_personas") or []
    agent.user_profile.ai_studio_persona = profile.get("ai_studio_persona") or {}
    agent.user_profile.user_voice = profile.get("user_voice") or {}
    agent.user_profile.mbti = profile.get("mbti") or {}


def _rebuild_row_app(agent: PersonaAgent, uid: str, backend_dir: str) -> None:
    """`generate_ai_studio_conversations` filters events by
    `self._row_app[oid] == 'AI_Studio'`. The router decisions are
    permanently encoded by which app's JSON each event lives in — so
    rebuild the dict by scanning each per-app file."""
    agent._row_app = {}
    for app_name in ("instagram", "facebook", "threads", "chatbot", "ai_studio"):
        path = Path(backend_dir) / uid / f"{app_name}.json"
        if not path.exists():
            continue
        with path.open() as f:
            events = json.load(f)
        # Normalize app name to the canonical title-case used by the router.
        canonical = {
            "instagram": "Instagram",
            "facebook": "Facebook",
            "threads": "Threads",
            "chatbot": "Chatbot",
            "ai_studio": "AI_Studio",
        }[app_name]
        for ev in events:
            oid = ev.get("source_object_id", "")
            if oid:
                agent._row_app[oid] = canonical


def _rebuild_canonical_groups(agent: PersonaAgent) -> None:
    """`generate_ai_studio_conversations` groups atomic personas by
    canonical via `self._canonical_groups`. Rebuild from atomic_personas."""
    from data_preparation.persona_agent import _normalize_persona_text
    groups = defaultdict(list)
    for ap in agent.atomic_personas:
        key = _normalize_persona_text(ap.persona_item)
        groups[key].append(ap)
    agent._canonical_groups = dict(groups)
    # Negative side too (mirrors the live pipeline state).
    neg_groups = defaultdict(list)
    for ap in agent.negative_personas:
        key = _normalize_persona_text(ap.persona_item)
        neg_groups[key].append(ap)
    agent._negative_canonical_groups = dict(neg_groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.5"))
    parser.add_argument("--mini_model", default="gpt-5.4-mini")
    parser.add_argument("--rate_limit", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    llm = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit)
    mini = QueryLLM({"models": {"llm_model": args.mini_model}}, rate_limit_per_min=args.rate_limit)

    agent = PersonaAgent(
        user_id=str(args.user_id),
        llm_client=llm,
        backend_dir=args.backend_dir,
        verbose=args.verbose,
        max_workers=50,
        llm_client_mini=mini,
    )
    if not agent.load_from_backend():
        print(f"backend/{args.user_id}/ not found or empty — aborting.")
        return 1

    _patch_user_profile_from_disk(agent, str(args.user_id), args.backend_dir)
    _rebuild_row_app(agent, str(args.user_id), args.backend_dir)
    _rebuild_canonical_groups(agent)

    if args.verbose:
        n_ai_studio_rows = sum(1 for a in agent._row_app.values() if a == "AI_Studio")
        print(f"[User {args.user_id}] reloaded; "
              f"hp={len(agent.user_profile.hidden_personas)}, "
              f"ai_persona={bool(agent.user_profile.ai_studio_persona)}, "
              f"atomics={len(agent.atomic_personas)}, "
              f"_row_app['AI_Studio']={n_ai_studio_rows}")

    agent.generate_ai_studio_conversations()
    # Step 18C audit is INLINE in Step 18B now — no need for a separate call.
    # But the records on disk are the final ones; nothing more to persist
    # since generate_ai_studio_conversations already wrote ai_studio.json
    # incrementally as part of the batched-parallel + inline-audit flow.

    print(f"[User {args.user_id}] AI Studio regen complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
