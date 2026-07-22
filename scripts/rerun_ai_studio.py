#!/usr/bin/env python3
"""Targeted re-run of Step 11C (AI Studio persona) + Step 18b (AI Studio
conversations) for a set of users, with a COORDINATED name blocklist so the
re-rolled AI characters don't collide with each other or with the kept users.

Why this exists: the cohort-diversity archetype router exposed a latent bug —
the LLM defaults to "Vale" as a surname (many users) and copies the prompt's
example first names ("Rowan"/"Wren"). The prompt is now fixed (forbid Vale /
Rowan / Wren / Mira + a used_names blocklist), and this script re-rolls ONLY the
AI Studio persona + conversations for the affected users — reusing all other
pipeline state from the existing backend JSONs (no full 28-step re-run).

Usage:
    python scripts/rerun_ai_studio.py \
        --input_csv data/<your_input>.csv \
        --user_ids <ids,to,re-roll> \
        --keep_uids <ids,to,preserve> \
        --backend_dir backend [--verbose]

Per-user steps:
  1. PersonaAgent.load_interactions(rows) + load_from_backend()  (restores
     user_profile/atomic state).
  2. Rehydrate user_profile.hidden_personas / user_voice / app_personas from
     profile.json; clear ai_studio_persona so 11C regenerates.
  3. Set agent.ai_studio_name_blocklist = shared blocklist; call
     generate_ai_studio_persona() (Step 11C). Validate the new name is unique
     (distinct first AND surname vs. blocklist; no Vale/Rowan/Wren/Mira); retry
     with an expanded blocklist; deterministic fallback as last resort.
  4. Rebuild ai_studio_records from the existing ai_studio.json (the same 5
     fields the agent's Step 18b feeds), delete the old ai_studio.json so the
     generator appends fresh, then call the STANDALONE
     ai_studio_conversation.generate_ai_studio_conversations(...).
  5. Persist: new ai_studio.json (written by the generator), ai_studio_memory.json
     (from the returned memory_state), and patch profile.json's
     ai_studio_persona block.
  6. Add the new character's name to the shared blocklist for the next user.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.main import load_and_group_csv
from data_preparation.persona_agent import PersonaAgent, HiddenPersona
from data_preparation import ai_studio_conversation
from data_preparation import ai_studio_memory as aism
from data_preparation import utils

BANNED_TOKENS = {"vale", "rowan", "wren", "mira"}  # prompt seeds / default surname
# Diverse, gender-mixed fallback full names (last resort only).
FALLBACK_NAMES = [
    "Mateo Reyes", "Priya Anand", "Desmond Clarke", "Yuki Tanaka",
    "Naomi Okafor", "Elias Brandt", "Sofia Marchetti", "Tariq Haddad",
    "Hana Kim", "Lucas Moreau", "Amara Diallo", "Kenji Nakamura",
    "Freya Lindgren", "Omar Nasser", "Bianca Costa", "Idris Bello",
]


def _name_parts(name: str):
    parts = [p for p in str(name or "").replace(",", " ").split() if p]
    # Drop a leading title (Captain / Auntie / Dr / etc.) for collision checks.
    titles = {"captain", "auntie", "uncle", "dr", "dr.", "mr", "mr.", "ms", "ms.",
              "mrs", "mrs.", "sir", "lady", "professor", "prof", "prof.", "coach"}
    core = [p for p in parts if p.lower().rstrip(".") not in titles]
    if not core:
        core = parts
    first = core[0].lower() if core else ""
    last = core[-1].lower() if len(core) > 1 else ""
    return first, last, " ".join(parts).lower()


def _build_taken(blocklist):
    firsts, lasts, fulls = set(), set(), set()
    for n in blocklist:
        f, l, full = _name_parts(n)
        if f:
            firsts.add(f)
        if l:
            lasts.add(l)
        if full:
            fulls.add(full)
    return firsts, lasts, fulls


def _name_ok(name: str, blocklist) -> tuple[bool, str]:
    first, last, full = _name_parts(name)
    if not first:
        return False, "empty name"
    if last in BANNED_TOKENS or first in BANNED_TOKENS:
        return False, f"banned token in {name!r}"
    firsts, lasts, fulls = _build_taken(blocklist)
    if full in fulls:
        return False, f"full-name collision {name!r}"
    if first in firsts:
        return False, f"first-name collision {first!r}"
    if last and last in lasts:
        return False, f"surname collision {last!r}"
    return True, ""


def _rehydrate_profile(agent: PersonaAgent, profile_dict: dict) -> None:
    """load_from_backend doesn't restore hidden_personas / user_voice /
    ai_studio_persona — patch them on so Step 11C sees full signal."""
    up = agent.user_profile
    hps = []
    for hp in (profile_dict.get("hidden_personas") or []):
        if not isinstance(hp, dict):
            continue
        try:
            hps.append(HiddenPersona(**{k: v for k, v in hp.items()
                                        if k in HiddenPersona.__dataclass_fields__}))
        except Exception:
            # Minimal fallback: just the fields 11C reads.
            hps.append(HiddenPersona(
                label=hp.get("label", ""), type=hp.get("type", ""),
                description=hp.get("description", ""),
                events=hp.get("events", []) or [],
            ))
    up.hidden_personas = hps
    up.user_voice = profile_dict.get("user_voice", {}) or {}
    up.app_personas = profile_dict.get("app_personas", {}) or {}
    up.proactive_trigger_candidates = profile_dict.get("proactive_trigger_candidates", []) or []
    up.friends = profile_dict.get("friends", []) or []
    up.ai_studio_persona = {}  # force Step 11C to regenerate


def rerun_user(uid: str, rows: list, llm_client, llm_client_mini,
               backend_dir: str, blocklist: list, verbose: bool) -> dict:
    user_dir = os.path.join(backend_dir, str(uid))
    profile_path = os.path.join(user_dir, "profile.json")
    ai_studio_path = os.path.join(user_dir, "ai_studio.json")
    mem_path = os.path.join(user_dir, "ai_studio_memory.json")

    with open(profile_path, "r", encoding="utf-8") as f:
        profile_dict = json.load(f)
    old_name = (profile_dict.get("ai_studio_persona") or {}).get("character_name", "")

    agent = PersonaAgent(
        user_id=str(uid), llm_client=llm_client, backend_dir=backend_dir,
        verbose=verbose, max_workers=20, llm_client_mini=llm_client_mini,
    )
    agent.load_interactions(rows)
    if not agent.load_from_backend():
        raise RuntimeError(f"load_from_backend failed for {uid}")
    _rehydrate_profile(agent, profile_dict)

    # ---- Step 11C: regenerate persona with name uniqueness loop ----
    local_block = list(blocklist)
    new_persona = None
    for attempt in range(5):
        agent.ai_studio_name_blocklist = list(local_block)
        agent.user_profile.ai_studio_persona = {}
        agent.generate_ai_studio_persona()
        cand = agent.user_profile.ai_studio_persona or {}
        cand_name = cand.get("character_name", "")
        ok, why = _name_ok(cand_name, local_block)
        if ok and cand.get("persona_archetype"):
            new_persona = cand
            break
        if verbose:
            print(f"  [User {uid}] name reroll {attempt+1}: {cand_name!r} rejected ({why})")
        if cand_name:
            local_block.append(cand_name)  # forbid the rejected name next try
    if new_persona is None:
        # Deterministic fallback: keep last persona DNA, swap in a clean name.
        cand = agent.user_profile.ai_studio_persona or {}
        for fb in FALLBACK_NAMES:
            if _name_ok(fb, local_block)[0]:
                cand["character_name"] = fb
                # address_terms shouldn't be the name; leave as-is.
                new_persona = cand
                if verbose:
                    print(f"  [User {uid}] using deterministic fallback name {fb!r}")
                break
    if not new_persona or not new_persona.get("persona_archetype"):
        raise RuntimeError(f"11C produced no usable persona for {uid}")
    new_name = new_persona.get("character_name", "")

    # ---- Rebuild ai_studio_records from existing ai_studio.json ----
    if not os.path.exists(ai_studio_path):
        raise RuntimeError(f"no existing ai_studio.json for {uid} — nothing to re-roll")
    with open(ai_studio_path, "r", encoding="utf-8") as f:
        old_events = json.load(f)
    ai_studio_records = [{
        "source_object_id": e.get("source_object_id", ""),
        "source_timestamp": e.get("source_timestamp", 0),
        "source_hashtags": list(e.get("source_hashtags", []) or []),
        "source_interaction_type": e.get("source_interaction_type", ""),
        "preferences": e.get("preferences", []) or [],
    } for e in old_events]
    n_events = len(ai_studio_records)

    # Delete the old file so the generator appends fresh (it reads this file
    # for the prev-2 verbatim slot and appends per event).
    os.remove(ai_studio_path)
    if os.path.exists(mem_path):
        os.remove(mem_path)

    # ---- Step 18b: standalone conversation generation ----
    try:
        user_seed = int(str(uid)) * 7919 + 131
    except ValueError:
        user_seed = abs(hash(str(uid))) % (2 ** 31)

    def _conv_query_fn(prompt: str):
        return agent._query_llm_with_retry(prompt, temperature=0.7)

    def _audit_query_fn(prompt: str):
        return agent._query_mini_with_retry(prompt)

    from data_preparation.persona_agent import ROGERS_CLICHE_BLOCKLIST

    final_records, mem, audit_summary = ai_studio_conversation.generate_ai_studio_conversations(
        ai_studio_records=ai_studio_records,
        user_profile=profile_dict,
        user_voice=profile_dict.get("user_voice", {}) or {},
        ai_studio_persona=new_persona,
        hidden_personas=profile_dict.get("hidden_personas", []) or [],
        llm_query_fn=_conv_query_fn,
        user_seed=user_seed,
        user_id=str(uid),
        backend_dir=backend_dir,
        memory_state=None,
        audit_query_fn=_audit_query_fn,
        rogers_cliche_baseline=ROGERS_CLICHE_BLOCKLIST,
        verbose=verbose,
    )

    # ---- Persist memory + patched profile ----
    with open(mem_path, "w", encoding="utf-8") as f:
        json.dump(aism.memory_state_to_dict(mem), f, ensure_ascii=False, indent=2)

    profile_dict["ai_studio_persona"] = new_persona
    tmp = profile_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profile_dict, f, ensure_ascii=False, indent=2)
    os.replace(tmp, profile_path)

    return {
        "user_id": str(uid), "old_name": old_name, "new_name": new_name,
        "archetype": new_persona.get("persona_archetype"),
        "n_events_in": n_events, "n_events_out": len(final_records),
        "audit": audit_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/all20_input.csv")
    ap.add_argument("--user_ids", required=True, help="comma-separated uids to re-roll")
    ap.add_argument("--keep_uids", default="", help="comma-separated uids whose names seed the blocklist")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--mini_model", default="gpt-5.4-mini")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from query_llm import QueryLLM
    llm_client = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit)
    llm_client_mini = QueryLLM({"models": {"llm_model": args.mini_model}}, rate_limit_per_min=args.rate_limit)

    grouped = load_and_group_csv(args.input_csv)
    reroll = [u.strip() for u in args.user_ids.split(",") if u.strip()]
    keep = [u.strip() for u in args.keep_uids.split(",") if u.strip()]

    # Seed the shared blocklist with the kept users' current AI character names.
    blocklist: list[str] = []
    for uid in keep:
        p = os.path.join(args.backend_dir, uid, "profile.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                nm = (json.load(f).get("ai_studio_persona") or {}).get("character_name", "")
            if nm:
                blocklist.append(nm)
    print(f"{utils.Colors.BOLD}AI Studio re-roll{utils.Colors.ENDC}  "
          f"reroll={reroll}  keep-seed-names={blocklist}")

    results = []
    for uid in reroll:
        if uid not in grouped:
            print(f"{utils.Colors.FAIL}[User {uid}] not in {args.input_csv}; skipping{utils.Colors.ENDC}")
            continue
        print(f"\n{utils.Colors.OKBLUE}=== re-rolling user {uid} ==={utils.Colors.ENDC}")
        try:
            r = rerun_user(uid, grouped[uid], llm_client, llm_client_mini,
                           args.backend_dir, blocklist, args.verbose)
            results.append(r)
            blocklist.append(r["new_name"])  # taken — next user avoids it
            print(f"{utils.Colors.OKGREEN}[User {uid}] {r['old_name']!r} -> {r['new_name']!r} "
                  f"({r['archetype']}); events {r['n_events_in']}->{r['n_events_out']}; "
                  f"audit={r['audit']}{utils.Colors.ENDC}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{utils.Colors.FAIL}[User {uid}] re-roll FAILED: {e}{utils.Colors.ENDC}")
            results.append({"user_id": uid, "error": str(e)})

    print(f"\n{utils.Colors.BOLD}=== Re-roll summary ==={utils.Colors.ENDC}")
    for r in results:
        if "error" in r:
            print(f"  {utils.Colors.FAIL}[{r['user_id']}] ERROR {r['error']}{utils.Colors.ENDC}")
        else:
            print(f"  [{r['user_id']}] {r['old_name']!r} -> {r['new_name']!r} ({r['archetype']})")
    print(f"\nfinal blocklist ({len(blocklist)}): {blocklist}")
    print(_usage(llm_client, args.model))
    print(_usage(llm_client_mini, args.mini_model))


def _usage(client, model):
    t = client.get_usage_totals()
    return (f"  {model:25s} calls={t.get('calls',0):>5} in={t.get('input_tokens',0):>9} "
            f"out={t.get('output_tokens',0):>8} errs={t.get('errors',0)}")


if __name__ == "__main__":
    main()
