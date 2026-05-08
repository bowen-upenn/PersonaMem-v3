"""[DEV-ONLY surgical tool — NOT the production path.]

Refresh just `ai_studio_persona` on backend/{user_id}/profile.json after
editing `personalize_ai_studio_persona_prompt`, the AIStudioPersona schema,
or the AI_STUDIO_ARCHETYPES catalog — without redoing the full ~20-minute
persona pipeline.

Loads the existing profile.json (user_voice + app_personas + hidden_personas
already there), re-builds the relevant slice of PersonaAgent state, calls
`generate_ai_studio_persona()` (one mini-tier LLM call), writes the
refreshed `ai_studio_persona` block back to profile.json.

For the production batch run over many users, use:

    scripts/run_persona_pipeline.py    # generates all backend/{uid}/*.json

This script duplicates one slice (Step 11C) so quick iteration on the
4-layer voice prompt doesn't cost a full regen.

Usage:

    python scripts/refresh_ai_studio_persona.py 115
    python scripts/refresh_ai_studio_persona.py 115 --input_csv data/gistbench_sample_10users.csv
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
    InteractionRow,
    PersonaAgent,
    UserProfile,
)
from query_llm import QueryLLM


def _load_user_csv_rows(input_csv: Path, user_id: str) -> list[dict]:
    """Load all rows for `user_id` from the source CSV. Used so
    `generate_ai_studio_persona` can extract the user's top hashtags."""
    if not input_csv.exists():
        return []
    out: list[dict] = []
    with input_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("user_id", "")) == str(user_id):
                out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument(
        "--input_csv",
        default="data/gistbench_sample_10users.csv",
        help="Source CSV for hashtag extraction (defaults to gistbench sample)",
    )
    ap.add_argument("--model", default="gpt-5-chat",
                    help="Flagship model (used as fallback)")
    ap.add_argument("--mini_model", default="gpt-5.4-mini",
                    help="Mini-tier model (Step 11C uses mini-tier)")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
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

    # Drop the cached ai_studio_persona so the method actually re-runs.
    profile_dict.pop("ai_studio_persona", None)

    # Build a minimal PersonaAgent and inject cached state.
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
    )

    # Reconstitute UserProfile from profile.json. We only need the fields
    # generate_ai_studio_persona() reads: user_voice, app_personas,
    # hidden_personas, name + identity fields, ai_studio_persona (cleared).
    hp_list = []
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
        ai_studio_persona={},  # cleared so the method runs
    )

    # Load source CSV rows so the method's hashtag counter has data.
    csv_rows = _load_user_csv_rows(Path(args.input_csv), args.user_id)
    agent.load_interactions(csv_rows)

    print(f"User {args.user_id}: refreshing ai_studio_persona via Step 11C "
          f"(mini-tier LLM call)…")
    agent.generate_ai_studio_persona()

    asp = agent.user_profile.ai_studio_persona
    if not asp:
        print("FAIL: generate_ai_studio_persona produced no output")
        sys.exit(1)

    # Write back to profile.json.
    profile_dict["ai_studio_persona"] = asp
    profile_path.write_text(json.dumps(profile_dict, indent=2, ensure_ascii=False))

    arch = asp.get("persona_archetype", "?")
    name = asp.get("character_name", "?")
    print(f"OK: wrote ai_studio_persona to {profile_path}")
    print(f"  archetype: {arch}")
    print(f"  character: {name}")
    sigs = asp.get("signature_phrases", [])
    if sigs:
        print(f"  signature: {sigs}")
    spine = asp.get("identity_spine", {})
    if spine:
        print(f"  identity_spine keys: {sorted(spine.keys())}")
    idio = asp.get("idiolect", {})
    if idio:
        ct = idio.get("constructional_templates", [])
        print(f"  idiolect constructional_templates: {len(ct)}")
        cr = idio.get("catchphrase_residue", [])
        print(f"  idiolect catchphrase_residue: {cr}")
    rep = asp.get("repertoire", {})
    if rep:
        print(f"  repertoire stances: {rep.get('stances', [])}")
    fp = asp.get("forbidden_phrases", [])
    print(f"  forbidden_phrases ({len(fp)}): {fp[:3]}…")
    rs = asp.get("romantic_specifier", {})
    if rs:
        nonnull = {k: v for k, v in rs.items() if v}
        if nonnull:
            print(f"  romantic_specifier (non-null axes): {nonnull}")
    niche = asp.get("niche_specifier")
    if niche:
        print(f"  niche_specifier: {niche}")


if __name__ == "__main__":
    main()
