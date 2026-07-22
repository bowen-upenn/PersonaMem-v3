#!/usr/bin/env python3
"""Backfill empty @ai `user_message` fields in the shipped 20 eval personas.

The R18 regen left ~28 @ai comment events (out of 220) with an empty
`interaction_format.user_message` — the mini-LLM message generator failed under
heavy parallel load and there was no fallback (fixed permanently in
persona_agent.generate_interaction_formats). This targeted backfill regenerates
ONLY the empty ones via the same mini prompt, falling back to the deterministic
voiced template when the LLM still fails. History-only; does not touch test.json.

Usage:
    python scripts/backfill_atai.py            # dry-run (report only)
    python scripts/backfill_atai.py --apply    # regenerate + write back
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation import prompts, utils  # noqa: E402
from data_preparation.persona_agent import AT_AI_ACTIONS, PLATFORM_INTERACTION_FORMATS  # noqa: E402

USERS = [1, 2, 3, 5, 6, 8, 9, 10, 13, 14, 26, 105, 115, 209, 229, 282, 461, 655, 760, 835]
SOCIAL_APPS = ["instagram", "facebook", "threads"]


class _Ctx:
    """Minimal carrier so the shared fallback helper can be reused verbatim."""

    def __init__(self, persona_item, category):
        self.persona_item = persona_item
        self.category = category


def _topic_phrase(persona_item, category):
    text = (persona_item or "").strip()
    low = text.lower()
    for v in ("is passionate about", "is a fan of", "is interested in",
              "identifies with", "responds positively to", "responds to",
              "engages with", "interested in", "passionate about",
              "enjoys", "likes", "loves", "follows", "prefers", "values",
              "seeks", "supports"):
        if low.startswith(v):
            text = text[len(v):].strip()
            break
    text = text.rstrip(".").strip()
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text or (category or "this").replace("_", " ")


def _template(action_id, persona_item, category):
    topic = _topic_phrase(persona_item, category)
    return {
        "at_ai_recommend_more": f"@ai recommend more like this — more {topic}.",
        "at_ai_focus_topic": f"@ai focus on {topic}.",
        "at_ai_not_interested": f"@ai not interested in {topic}.",
        "at_ai_stop_recommending": f"@ai stop recommending {topic}.",
        "at_ai_feels_off": f"@ai this feels off — less {topic}.",
    }.get(action_id, f"@ai focus on {topic}.")


def _event_context(event):
    """Recover the (persona_item, category, interaction_type) a message needs."""
    prefs = event.get("preferences") or []
    persona_item = ""
    category = ""
    if prefs:
        persona_item = prefs[0].get("persona_item") or ""
        category = prefs[0].get("category") or ""
    if not persona_item:
        content = event.get("content") or {}
        persona_item = content.get("title") or content.get("caption") or ""
        tags = event.get("source_hashtags") or []
        if not persona_item and tags:
            persona_item = str(tags[0])
    return persona_item, category, event.get("source_interaction_type", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes back (default: dry-run)")
    ap.add_argument("--mini_model", default="gpt-5.4-mini")
    ap.add_argument("--rate_limit", type=int, default=50)
    args = ap.parse_args()

    client = None
    if args.apply:
        from query_llm import QueryLLM
        client = QueryLLM({"models": {"llm_model": args.mini_model}}, rate_limit_per_min=args.rate_limit)

    repo = Path(__file__).resolve().parents[1]
    total_empty = 0
    filled_llm = 0
    filled_tmpl = 0

    for uid in USERS:
        prof_path = repo / "backend" / str(uid) / "profile.json"
        try:
            profile = json.loads(prof_path.read_text())
        except Exception:
            profile = {}
        user_voice = profile.get("user_voice") or {}
        app_personas = profile.get("app_personas") or {}
        # app_personas may be a list of dicts keyed by 'app'
        if isinstance(app_personas, list):
            app_personas = {d.get("app"): d for d in app_personas if isinstance(d, dict)}

        for app in SOCIAL_APPS:
            path = repo / "backend" / str(uid) / f"{app}.json"
            try:
                events = json.loads(path.read_text())
            except Exception:
                continue
            changed = False
            for e in events:
                fmt = e.get("interaction_format") or {}
                action_id = fmt.get("action", "")
                if not (isinstance(action_id, str) and action_id in AT_AI_ACTIONS):
                    continue
                existing = (fmt.get("user_message") or "").strip()
                if existing:
                    # Deterministic prefix normalization (no LLM): the @ai
                    # prefix is occasionally dropped by the generator.
                    if not existing.lower().startswith("@ai"):
                        if args.apply:
                            fmt["user_message"] = f"@ai {existing}"
                            e["interaction_format"] = fmt
                            changed = True
                    continue
                total_empty += 1
                persona_item, category, itype = _event_context(e)
                canonical_label = fmt.get("action_label") or "@ai comment"
                new_msg = None
                if args.apply and client is not None:
                    app_key = fmt.get("app") or app.capitalize()
                    app_persona = app_personas.get(app_key) or app_personas.get(app.capitalize()) or {}
                    prompt = prompts.generate_interaction_format_prompt(
                        persona_item=persona_item,
                        category=category,
                        interaction_type=itype,
                        assigned_app=app_key,
                        app_persona=app_persona,
                        action_catalog=[{"action": action_id, "label": canonical_label}],
                        requires_user_message=True,
                        user_voice=user_voice,
                    )
                    try:
                        resp = client.query_llm(prompt, verbose=False)
                    except Exception:
                        resp = None
                    if resp:
                        parsed = utils.extract_json_from_response(resp)
                        if isinstance(parsed, dict):
                            cand = parsed.get("user_message")
                            if cand and str(cand).strip():
                                new_msg = str(cand).strip()
                if args.apply:
                    if new_msg:
                        filled_llm += 1
                    else:
                        new_msg = _template(action_id, persona_item, category)
                        filled_tmpl += 1
                    new_msg = new_msg.strip()
                    if not new_msg.lower().startswith("@ai"):
                        new_msg = f"@ai {new_msg}"
                    fmt["user_message"] = new_msg
                    e["interaction_format"] = fmt
                    changed = True
            if changed and args.apply:
                path.write_text(json.dumps(events, ensure_ascii=False, indent=2))

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] empty @ai messages found={total_empty}  "
          f"filled_by_llm={filled_llm}  filled_by_template={filled_tmpl}")


if __name__ == "__main__":
    main()
