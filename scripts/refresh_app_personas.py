"""[DEV-ONLY surgical tool — NOT the production path.]

Refresh just `user_voice` + `app_personas` in backend/{user_id}/profile.json
after editing `generate_app_personas_prompt` or the schema, without redoing
the full ~11-minute persona pipeline. Reads existing profile + top
preferences (and optionally a sample of raw source rows for grounding),
makes ONE LLM call, writes both `user_voice` and `app_personas` back to
profile.json.

For the production batch run over many users, use:

    scripts/run_persona_pipeline.py    # generates all backend/{uid}/*.json
                                       # (already calls the same updated
                                       # generate_app_personas_prompt)
    scripts/prepare_eval_data.py --all # generates all benchmark/{uid}/...

This script duplicates one slice of run_persona_pipeline so quick
iteration on prompt wording doesn't cost a full regen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation import prompts, utils
from data_preparation.persona_agent import CHATBOT_CONTEXTS, PLATFORMS, AppPersona, UserVoice
from dataclasses import asdict
from query_llm import QueryLLM


def _load_source_samples(user_id: str, max_rows: int = 10) -> list[dict]:
    """Best-effort load ~10 stratified raw source rows for voice grounding.

    Looks for the user's CSV in `data/{user_id}.csv`, then `data/sample/{user_id}.csv`.
    Stratifies by interaction_type and picks the most-textual rows. Silently
    returns [] if no source CSV is found — the prompt tolerates an empty
    samples block.
    """
    candidates = [
        Path("data") / f"{user_id}.csv",
        Path("data") / "sample" / f"{user_id}.csv",
        Path("facebook") / "gistbench" / f"{user_id}.csv",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    if src_path is None:
        return []

    rows_by_type: dict[str, list[dict]] = {}
    try:
        with src_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                it = (row.get("interaction_type") or "").strip()
                if not it:
                    continue
                rows_by_type.setdefault(it, []).append({
                    "interaction_time": int(row.get("interaction_time") or 0),
                    "interaction_type": it,
                    "object_text": (row.get("object_text") or "").strip(),
                })
    except Exception:
        return []

    # Sort each bucket by descending object_text length so we get content-rich rows
    for t in rows_by_type:
        rows_by_type[t].sort(key=lambda r: -len(r["object_text"]))

    ordered_types = sorted(
        rows_by_type.keys(),
        key=lambda t: -sum(len(r["object_text"]) for r in rows_by_type[t]),
    )
    picked: list[dict] = []
    idx_per_type = {t: 0 for t in ordered_types}
    while len(picked) < max_rows:
        progressed = False
        for t in ordered_types:
            i = idx_per_type[t]
            if i < len(rows_by_type[t]):
                row = rows_by_type[t][i]
                idx_per_type[t] = i + 1
                progressed = True
                if row["object_text"]:
                    picked.append(row)
                    if len(picked) >= max_rows:
                        break
        if not progressed:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", help="User id (subfolder in backend/)")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--top_n", type=int, default=20)
    args = ap.parse_args()

    user_dir = Path(args.backend_dir) / args.user_id
    profile_path = user_dir / "profile.json"
    if not profile_path.exists():
        print(f"FAIL: {profile_path} not found")
        sys.exit(1)
    profile = json.loads(profile_path.read_text())

    # profile.preferences in the saved JSON is a list of pre-formatted
    # strings ("HH:MM, MM/DD/YYYY : <persona_item>"), already the
    # surviving high-confidence canonicals. Strip the timestamp prefix
    # and take up to --top_n distinct persona_items.
    prefs = profile.get("preferences") or []
    seen: set[str] = set()
    top_personas: list[str] = []
    for p in prefs:
        if isinstance(p, dict):
            text = (p.get("persona_item") or "").strip()
        elif isinstance(p, str):
            text = p.split(":", 2)[-1].strip() if ":" in p else p.strip()
        else:
            continue
        if not text or text in seen:
            continue
        seen.add(text)
        top_personas.append(text)
        if len(top_personas) >= args.top_n:
            break
    if not top_personas:
        print("FAIL: no preferences in profile.json — cannot generate app_personas")
        sys.exit(1)

    profile_dict = {
        "name": profile.get("name", ""),
        "gender": profile.get("gender", ""),
        "race_ethnicity": profile.get("race_ethnicity", ""),
        "career": profile.get("career", ""),
        "education": profile.get("education", ""),
        "big_five": profile.get("big_five", {}),
        "bio": profile.get("bio", ""),
    }

    source_samples = _load_source_samples(args.user_id, max_rows=10)
    if source_samples:
        print(f"  (grounding with {len(source_samples)} raw source rows)")
    else:
        print("  (no source CSV found — generating without raw row grounding)")

    prompt = prompts.generate_app_personas_prompt(
        profile=profile_dict,
        top_personas=top_personas,
        chatbot_contexts=CHATBOT_CONTEXTS,
        source_samples=source_samples,
    )

    print(f"User {args.user_id}: calling LLM with the updated app_personas prompt...")
    llm = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=50)
    response = llm.query_llm(prompt)
    if not response:
        print("FAIL: empty LLM response")
        sys.exit(1)

    parsed = utils.extract_json_from_response(response)
    if not isinstance(parsed, dict):
        print("FAIL: unparseable LLM response")
        print("RAW:", response[:1000])
        sys.exit(1)

    # Parse the shared user_voice block
    uv_raw = parsed.get("user_voice") or {}
    if not isinstance(uv_raw, dict) or not uv_raw:
        print("FAIL: no user_voice in LLM response")
        sys.exit(1)
    user_voice = UserVoice(
        natural_register=str(uv_raw.get("natural_register", "")),
        default_capitalization=str(uv_raw.get("default_capitalization", "")),
        punctuation_habits=str(uv_raw.get("punctuation_habits", "")),
        humor_tone=str(uv_raw.get("humor_tone", "")),
        emoji_palette=list(uv_raw.get("emoji_palette", []) or []),
        emoji_intensity_default=str(uv_raw.get("emoji_intensity_default", "medium")),
        personal_phrases=list(uv_raw.get("personal_phrases", []) or []),
        formality_baseline=float(uv_raw.get("formality_baseline", 0.3) or 0.3),
        voice_avoid=str(uv_raw.get("voice_avoid", "")),
        phrases_to_avoid=list(uv_raw.get("phrases_to_avoid", []) or []),
    )

    app_personas_block = parsed.get("app_personas") or {}
    if not isinstance(app_personas_block, dict):
        print("FAIL: no app_personas dict in LLM response")
        sys.exit(1)

    new_app_personas: dict = {}
    for app_name in PLATFORMS:
        entry = app_personas_block.get(app_name)
        if not isinstance(entry, dict):
            print(f"WARN: {app_name} missing in LLM response — skipping")
            continue
        expression = entry.get("expression") or {}
        if not isinstance(expression, dict):
            expression = {}
        overrides = entry.get("overrides") or {}
        if not isinstance(overrides, dict):
            overrides = {}
        ap = AppPersona(
            app_name=app_name,
            use_purposes=list(entry.get("use_purposes", [])),
            friend_zones=list(entry.get("friend_zones", [])),
            audience_type=entry.get("audience_type", "mixed"),
            audience_lens=str(entry.get("audience_lens", "")),
            style_description=str(entry.get("style_description", "")),
            topical_focus=list(entry.get("topical_focus", [])),
            chatbot_contexts=list(entry.get("chatbot_contexts", [])) if app_name == "Chatbot" else [],
            expression=expression,
            overrides=overrides,
            app_avoid=str(entry.get("app_avoid", "")),
        )
        new_app_personas[app_name] = asdict(ap)

    if not new_app_personas:
        print("FAIL: no app_personas parsed from LLM response")
        sys.exit(1)

    profile["user_voice"] = asdict(user_voice)
    profile["app_personas"] = new_app_personas
    # Drop legacy voice_signature if a previous version left it inside app_personas
    for ap_dict in profile["app_personas"].values():
        ap_dict.pop("voice_signature", None)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))

    n_overrides = sum(1 for ap in new_app_personas.values() if ap.get("overrides"))
    print(f"OK: wrote user_voice + {len(new_app_personas)} app_personas to {profile_path}")
    print(f"  shared voice: caps={user_voice.default_capitalization} | "
          f"palette={' '.join(user_voice.emoji_palette)} | "
          f"phrases={user_voice.personal_phrases}")
    if user_voice.voice_avoid:
        print(f"  voice avoid: {user_voice.voice_avoid}")
    if user_voice.phrases_to_avoid:
        print(f"  phrases to avoid: {user_voice.phrases_to_avoid}")
    print(f"  overrides populated: {n_overrides}/{len(new_app_personas)} apps "
          f"(target: most users ≤ 1)")
    for app, ap in new_app_personas.items():
        ov = ap.get("overrides") or {}
        ov_str = f" overrides={list(ov.keys())}" if ov else ""
        expr = ap.get("expression") or {}
        avoid = ap.get("app_avoid") or ""
        avoid_str = f" | app_avoid=\"{avoid[:60]}{'…' if len(avoid) > 60 else ''}\"" if avoid else ""
        print(f"  {app}: audience={ap.get('audience_type','?')} | "
              f"length={expr.get('length_band','?')} | "
              f"effort={expr.get('effort_level','?')}{ov_str}{avoid_str}")


if __name__ == "__main__":
    main()
