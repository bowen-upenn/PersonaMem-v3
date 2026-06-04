"""[DEV-ONLY surgical tool — NOT the production path.]

Refresh AI Studio outputs (steps 11C + 18B + 18C) on an existing
backend/{user_id}/ tree, without redoing the full ~20-minute persona
pipeline.

Operates on:
  - profile.json                  (re-runs Step 11C → ai_studio_persona)
  - ai_studio.json                (re-runs Step 18B → per-event conversations)
  - ai_studio_memory.json         (re-runs Step 18B → cross-session memory)
  - Step 18C audit drops events that fail the safety floor.

Does NOT touch:
  - instagram.json / facebook.json / threads.json / chatbot.json
  - calendar.json
  - any other profile fields besides `ai_studio_persona`

Usage:
    python scripts/refresh_ai_studio_conversations.py 115
    python scripts/refresh_ai_studio_conversations.py 115 --input_csv data/gistbench_sample_10users.csv

Equivalent to running scripts/refresh_ai_studio_persona.py (which only
does Step 11C) followed by Steps 18B + 18C from run_persona_pipeline.py,
but reuses the already-cached upstream state (atomic_personas,
cross_referenced_personas, _row_app) reconstructed from the saved
per-app JSONs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation import utils
from data_preparation.persona_agent import (
    HiddenPersona,
    PersonaAgent,
    UserProfile,
    _normalize_persona_text,
)
from query_llm import QueryLLM


# Supported app file names → canonical row_app label used by Step 18B.
_APP_FILES = {
    "instagram.json":  "Instagram",
    "facebook.json":   "Facebook",
    "threads.json":    "Threads",
    "chatbot.json":    "Chatbot",
    "ai_studio.json":  "AI_Studio",
}


def _load_user_csv_rows(input_csv: Path, user_id: str) -> list[dict]:
    if not input_csv.exists():
        return []
    out: list[dict] = []
    with input_csv.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("user_id", "")) == str(user_id):
                out.append(row)
    return out


def _reconstruct_row_app(user_dir: Path) -> dict[str, str]:
    """For each saved per-app JSON, map source_object_id → app label.
    Step 18B reads `self._row_app` to filter AI_Studio-routed atomics."""
    row_app: dict[str, str] = {}
    for fname, app_label in _APP_FILES.items():
        path = user_dir / fname
        if not path.exists():
            continue
        try:
            events = json.loads(path.read_text())
        except Exception:
            continue
        for ev in events:
            oid = ev.get("source_object_id")
            if oid:
                row_app[str(oid)] = app_label
    return row_app


def _reconstruct_canonical_groups(agent: PersonaAgent) -> dict[str, list]:
    """Group atomic_personas by their canonical (normalized) persona_item
    key. Step 18B uses this for fallback lookup when an atomic's exact
    text doesn't match the canonical's text. Empty dict is safe; building
    it from atomics improves lookup hit rate."""
    groups: dict[str, list] = {}
    for ap in agent.atomic_personas:
        key = _normalize_persona_text(ap.persona_item)
        groups.setdefault(key, []).append(ap)
    return groups


def _reload_existing_ai_studio_events(user_dir: Path) -> dict[str, dict]:
    """Load any existing backend/{uid}/ai_studio.json so we can preserve
    per-event metadata (interaction_format, content, etc.) when overlaying
    the freshly-regenerated conversation fields. Returns
    {source_object_id: full_event_dict}."""
    path = user_dir / "ai_studio.json"
    if not path.exists():
        return {}
    try:
        events = json.loads(path.read_text())
    except Exception:
        return {}
    return {str(e.get("source_object_id", "")): e for e in events if e.get("source_object_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--input_csv",
                    default="data/gistbench_sample_10users.csv",
                    help="Source CSV for interaction reconstruction")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--mini_model", default="gpt-5.4-mini")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=8,
                    help="Per-event LLM concurrency for Step 18B (sequential by design — kept low)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip_11c", action="store_true",
                    help="Skip Step 11C and reuse existing ai_studio_persona on profile.json")
    ap.add_argument("--max_ai_studio_events", type=int, default=None,
                    help="Cap the number of AI_Studio-routed events that Step 18B "
                         "generates conversations for. When set, the kept subset is "
                         "stratified chronologically (every Nth event in source_timestamp "
                         "order) so the full S1→S4 intimacy arc is preserved. The "
                         "remaining AI_Studio source_object_ids are dropped from "
                         "self._row_app for the duration of this run.")
    args = ap.parse_args()

    user_dir = Path(args.backend_dir) / args.user_id
    profile_path = user_dir / "profile.json"
    if not profile_path.exists():
        print(f"FAIL: {profile_path} not found")
        sys.exit(1)
    profile_dict = json.loads(profile_path.read_text())

    if not profile_dict.get("user_voice", {}).get("identity_spine"):
        print("FAIL: profile.json has no 4-layer user_voice — run the full pipeline first")
        sys.exit(1)

    # If we're going to re-run 11C, clear the cached ai_studio_persona so
    # `generate_ai_studio_persona()` actually fires (it's a no-op if the
    # field is already populated).
    if not args.skip_11c:
        profile_dict.pop("ai_studio_persona", None)

    # --- Build PersonaAgent + LLM clients ---
    llm_client = QueryLLM({"models": {"llm_model": args.model}},
                          rate_limit_per_min=args.rate_limit)
    llm_client_mini = QueryLLM({"models": {"llm_model": args.mini_model}},
                               rate_limit_per_min=args.rate_limit)
    agent = PersonaAgent(
        user_id=args.user_id,
        llm_client=llm_client,
        backend_dir=args.backend_dir,
        verbose=args.verbose,
        llm_client_mini=llm_client_mini,
        max_workers=args.parallel,
    )

    # --- Reconstruct UserProfile from profile.json ---
    hp_list: list[HiddenPersona] = []
    for h in profile_dict.get("hidden_personas", []) or []:
        if isinstance(h, dict):
            hp_list.append(HiddenPersona(**{
                k: v for k, v in h.items()
                if k in {f.name for f in HiddenPersona.__dataclass_fields__.values()}
            }))
    agent.user_profile = UserProfile(
        name=profile_dict.get("name", ""),
        gender=profile_dict.get("gender", ""),
        race_ethnicity=profile_dict.get("race_ethnicity", ""),
        career=profile_dict.get("career", ""),
        education=profile_dict.get("education", ""),
        big_five=profile_dict.get("big_five", {}) or {},
        bio=profile_dict.get("bio", ""),
        user_voice=profile_dict.get("user_voice", {}) or {},
        app_personas=profile_dict.get("app_personas", {}) or {},
        hidden_personas=hp_list,
        hidden_persona_summary=profile_dict.get("hidden_persona_summary", ""),
        mbti=profile_dict.get("mbti", {}) or {},
        mobility_class=profile_dict.get("mobility_class", ""),
        ai_studio_persona=profile_dict.get("ai_studio_persona", {}) or {},
    )

    # --- Load interactions (needed by 11C for hashtag extraction;
    #     also seeds agent.interactions which 18B references downstream) ---
    csv_rows = _load_user_csv_rows(Path(args.input_csv), args.user_id)
    agent.load_interactions(csv_rows)

    # --- Reconstruct atomic_personas + cross_referenced_personas from
    #     the saved per-app JSONs (load_from_backend reads everything in
    #     backend/{uid}/). ---
    if not agent.load_from_backend():
        print(f"FAIL: load_from_backend returned False for {user_dir}")
        sys.exit(1)

    # --- Reconstruct _row_app + _canonical_groups (Step 18B needs both) ---
    agent._row_app = _reconstruct_row_app(user_dir)
    agent._canonical_groups = _reconstruct_canonical_groups(agent)

    # Optional cap on AI_Studio event count. Stratify chronologically so
    # the kept subset spans the full timeline (preserves the S1→S4
    # intimacy arc; sequential cross-session memory in 18B still works
    # because it only references events within the kept set).
    if args.max_ai_studio_events is not None and args.max_ai_studio_events > 0:
        ai_studio_oids_in_order: list[str] = []
        for fname, app_label in _APP_FILES.items():
            if app_label != "AI_Studio":
                continue
            path = user_dir / fname
            if not path.exists():
                continue
            try:
                events = json.loads(path.read_text())
            except Exception:
                continue
            for ev in sorted(events,
                              key=lambda e: (int(e.get("source_timestamp") or 0),
                                              e.get("source_object_id", ""))):
                oid = ev.get("source_object_id")
                if oid:
                    ai_studio_oids_in_order.append(str(oid))
            break
        n_total = len(ai_studio_oids_in_order)
        cap = args.max_ai_studio_events
        if n_total > cap:
            # Stratified pick: round((i + 0.5) * n_total / cap) for i in range(cap)
            kept_idx = set()
            for i in range(cap):
                idx = int(round((i + 0.5) * n_total / cap))
                idx = min(max(idx, 0), n_total - 1)
                kept_idx.add(idx)
            kept_oids = {ai_studio_oids_in_order[i] for i in kept_idx}
            dropped = 0
            for oid in list(agent._row_app.keys()):
                if agent._row_app[oid] == "AI_Studio" and oid not in kept_oids:
                    # Re-route dropped AI_Studio rows to a sentinel app so
                    # Step 18B skips them (any non-AI_Studio label works).
                    agent._row_app[oid] = "_dropped"
                    dropped += 1
            print(f"[User {args.user_id}] AI_Studio cap: "
                  f"{n_total} → {len(kept_oids)} events kept (stratified), "
                  f"{dropped} dropped from _row_app for this run.")

    if args.verbose:
        ai_studio_oids = sum(1 for v in agent._row_app.values() if v == "AI_Studio")
        print(f"[User {args.user_id}] Reconstructed state: "
              f"{len(agent.atomic_personas)} atomic personas, "
              f"{len(agent.cross_referenced_personas)} canonicals, "
              f"{ai_studio_oids} AI_Studio-routed rows.")

    # --- Step 11C: generate_ai_studio_persona ---
    if not args.skip_11c:
        print(f"[User {args.user_id}] Step 11C: generate_ai_studio_persona…")
        agent.generate_ai_studio_persona()
        asp = agent.user_profile.ai_studio_persona
        if not asp:
            print("FAIL: generate_ai_studio_persona produced no output")
            sys.exit(1)
        if args.verbose:
            print(f"  archetype: {asp.get('persona_archetype','?')}; "
                  f"character: {asp.get('character_name','?')}")
    else:
        if args.verbose:
            print(f"[User {args.user_id}] Step 11C: SKIPPED (using cached persona)")

    # --- Step 18B: generate_ai_studio_conversations ---
    print(f"[User {args.user_id}] Step 18B: generate_ai_studio_conversations…")
    agent.generate_ai_studio_conversations()
    records = getattr(agent, "_ai_studio_records", []) or []
    if args.verbose:
        print(f"  generated {len(records)} conversation events")

    # --- Step 18C: audit (drops events failing the safety floor) ---
    print(f"[User {args.user_id}] Step 18C: audit_ai_studio_conversations…")
    agent.audit_ai_studio_conversations()
    records = getattr(agent, "_ai_studio_records", []) or []
    if args.verbose:
        print(f"  post-audit: {len(records)} events kept")

    # --- Persist updates surgically (no full save_to_backend) ---

    # 1. profile.json: refresh ai_studio_persona block only.
    if not args.skip_11c:
        profile_dict["ai_studio_persona"] = asdict(agent.user_profile)["ai_studio_persona"] \
            if hasattr(agent.user_profile, "ai_studio_persona") and \
               not isinstance(agent.user_profile.ai_studio_persona, dict) \
            else agent.user_profile.ai_studio_persona
        # Some dataclass paths leave the field as the AIStudioPersona dataclass;
        # asdict() flattens any nested dataclasses too.
        if hasattr(agent.user_profile.ai_studio_persona, "__dataclass_fields__"):
            profile_dict["ai_studio_persona"] = asdict(agent.user_profile.ai_studio_persona)
        else:
            profile_dict["ai_studio_persona"] = agent.user_profile.ai_studio_persona
        profile_path.write_text(json.dumps(profile_dict, indent=2, ensure_ascii=False))
        print(f"  wrote profile.json (ai_studio_persona refreshed)")

    # 2. ai_studio.json: overlay regenerated conversation fields onto
    #    existing per-event metadata; drop events with no fresh conversation.
    existing_by_oid = _reload_existing_ai_studio_events(user_dir)
    records_by_oid = {r.get("source_object_id", ""): r for r in records}
    kept_events: list[dict] = []
    for oid, base in existing_by_oid.items():
        rec = records_by_oid.get(oid)
        if not rec or not rec.get("conversation"):
            continue  # mirror save_to_backend: drop events with no convo
        merged = dict(base)
        merged["conversation"] = rec["conversation"]
        merged["conversation_type"] = rec.get("conversation_type", "")
        merged["prior_session_refs"] = rec.get("prior_session_refs", [])
        merged["memory_used_summary"] = rec.get("memory_used_summary", "")
        merged["oblique_reference_to_hidden_personas"] = rec.get(
            "oblique_reference_to_hidden_personas", []
        )
        merged["ai_studio_metadata"] = rec.get("ai_studio_metadata", {})
        if rec.get("audit_status"):
            merged["audit_status"] = rec["audit_status"]
        if rec.get("audit_axes"):
            merged["audit_axes"] = rec["audit_axes"]
        kept_events.append(merged)
    # Re-sort by source_timestamp for stability.
    kept_events.sort(key=lambda e: (int(e.get("source_timestamp") or 0),
                                     e.get("source_object_id", "")))
    ai_studio_path = user_dir / "ai_studio.json"
    ai_studio_path.write_text(json.dumps(kept_events, indent=2, ensure_ascii=False))
    print(f"  wrote {ai_studio_path} ({len(kept_events)} events kept)")

    # 3. ai_studio_memory.json: fresh from `_ai_studio_memory_state`.
    mem = getattr(agent, "_ai_studio_memory_state", None)
    if mem is not None and (mem.episodic_memory_items
                              or mem.running_relational_state.intimacy_arc > 0):
        from data_preparation import ai_studio_memory as _aism
        mem_path = user_dir / "ai_studio_memory.json"
        mem_path.write_text(json.dumps(
            _aism.memory_state_to_dict(mem), indent=2, ensure_ascii=False
        ))
        print(f"  wrote {mem_path}")

    print(f"[User {args.user_id}] OK.")


if __name__ == "__main__":
    main()
