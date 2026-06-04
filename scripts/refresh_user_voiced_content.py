"""[DEV-ONLY surgical tool — NOT the production path.]

Refresh just the user-voiced content for a backend user — self-posts, DM
threads, and chatbot conversations — without redoing Steps 1–10 of the
persona pipeline.

For the production batch run on many (e.g. 1000) users, just rerun the full
pipeline end-to-end:

    scripts/run_persona_pipeline.py --input_csv data/big.csv  # 24 steps incl.
                                                              # Step 24 Ext B
    scripts/prepare_eval_data.py --all                        # benchmarks +
                                                              # voice-evidence
                                                              # smoke gate

The full pipeline already produces the new schema (voice_avoid,
phrases_to_avoid, app_avoid) and the new behaviors (per-app length bands,
voice-evidence smoke test, bolded gold spans) natively in a single run.
No patches required. This script is the *iteration* tool when you've edited
voice-related prompts and want to refresh the voiced text on existing
backends without paying for hashtag-inference / cross-ref / hidden-persona
regen again.

Use case: after editing the writing-voice schema or the prompts that generate
voiced text (`prompts._render_user_voice_block`,
`extension_b/self_posts.py::DM/POST templates`, the four chatbot conversation
prompts in `prompts.py`), regenerate only the artifacts that depend on
`profile.json`'s `user_voice` + `app_personas` — leave hashtag inference,
cross-ref, hidden personas, sensitive events, etc. untouched.

Steps performed:
  1. Strip events with `is_self_authored=True` OR `is_dm=True` from each
     social app JSON (Extension B appends rather than replaces, so without
     this we'd get duplicates).
  2. Run `extension_b.run_extension_b()` to regen friends, self-posts, DMs,
     trending — all reading the updated `profile.json`.
  3. Regen `conversation` field on every chatbot event via
     `chatbot_conversation.generate_chatbot_conversations()` with the new
     `user_voice`.
  4. Regen `user_message` field on every social event whose
     `interaction_format.action` starts with `at_ai_` (skipped silently when
     this user has no @ai preferences — common case).

For the production batch run on many users, just rerun the full pipeline.
This script is the surgical iteration tool when only voice-dependent text
needs refreshing.

CLI:
    python scripts/refresh_user_voiced_content.py 115 --backend_dir backend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preparation import prompts
from data_preparation.chatbot_conversation import generate_chatbot_conversations
from data_preparation.extension_b.main import run_extension_b
from query_llm import QueryLLM


SOCIAL_APPS = ("instagram", "facebook", "threads")


def _strip_voiced_events(base: Path) -> dict[str, int]:
    """Remove self-posts + DMs from each social app JSON. Returns counts of
    events stripped per app so the operator can sanity-check."""
    stripped: dict[str, int] = {}
    for app in SOCIAL_APPS:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        events = json.loads(path.read_text())
        before = len(events)
        kept = [e for e in events if not (e.get("is_self_authored") or e.get("is_dm"))]
        path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
        stripped[app] = before - len(kept)
    return stripped


def _refresh_chatbot_conversations(base: Path, profile: dict, llm_client: QueryLLM,
                                   user_id: str, parallel: int) -> int:
    """Re-generate `conversation` for every chatbot event. Returns the count
    of events updated."""
    chatbot_path = base / "chatbot.json"
    if not chatbot_path.exists():
        return 0
    events = json.loads(chatbot_path.read_text())

    chatbot_persona = (profile.get("app_personas") or {}).get("Chatbot") or {}
    user_voice = profile.get("user_voice") or {}
    user_seed = abs(hash(user_id)) % (2**31)

    # generate_chatbot_conversations mutates the records in place AND skips
    # any event whose preferences list is empty — so we can pass the full
    # events list and only those with prefs get a refresh.
    generate_chatbot_conversations(
        chatbot_records=events,
        user_profile=profile,
        chatbot_persona=chatbot_persona,
        llm_query_fn=llm_client.query_llm,
        user_seed=user_seed,
        max_workers=parallel,
        user_voice=user_voice,
    )
    chatbot_path.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    return sum(1 for e in events if e.get("conversation"))


def _refresh_at_ai_user_messages(base: Path, profile: dict, llm_client: QueryLLM) -> int:
    """Regen `user_message` on every preference whose `interaction_format.action`
    starts with `at_ai_`. Iterates social-app events. No-op when the user has
    no @ai preferences (common case)."""
    n_updated = 0
    user_voice = profile.get("user_voice") or {}
    app_personas = profile.get("app_personas") or {}
    for app in SOCIAL_APPS:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        events = json.loads(path.read_text())
        ap_pretty = app.capitalize()
        app_persona = app_personas.get(ap_pretty) or {}
        if not app_persona:
            continue
        changed = False
        for e in events:
            for pref in (e.get("preferences") or []):
                ifmt = pref.get("interaction_format") or {}
                action = ifmt.get("action") or ""
                if not action.startswith("at_ai_"):
                    continue
                # Single-action catalog so the LLM regenerates the message
                # for THIS specific action without re-routing.
                catalog = {action: ifmt.get("action_label", action)}
                prompt = prompts.generate_interaction_format_prompt(
                    app=app,
                    interaction_type=e.get("source_interaction_type", ""),
                    persona_item=pref.get("persona_item", ""),
                    category=pref.get("category", ""),
                    app_persona=app_persona,
                    action_catalog=catalog,
                    user_voice=user_voice,
                    requires_user_message=True,
                )
                resp = llm_client.query_llm(prompt) or ""
                # Extract user_message from JSON response. The prompt asks
                # the LLM to emit a JSON with action + action_label +
                # user_message; we only need user_message here.
                from data_preparation.utils import extract_json_from_response
                parsed = extract_json_from_response(resp) or {}
                msg = (parsed.get("user_message") or "").strip()
                if msg:
                    pref["interaction_format"]["user_message"] = msg
                    changed = True
                    n_updated += 1
        if changed:
            path.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    return n_updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", help="User id (subfolder under --backend_dir)")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=20,
                    help="Parallel LLM calls for chatbot regen (default 20)")
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--skip_extension_b", action="store_true")
    ap.add_argument("--skip_chatbot", action="store_true")
    ap.add_argument("--skip_at_ai", action="store_true")
    args = ap.parse_args()

    base = Path(args.backend_dir) / args.user_id
    if not base.exists():
        print(f"FAIL: {base} does not exist")
        sys.exit(1)

    profile_path = base / "profile.json"
    if not profile_path.exists():
        print(f"FAIL: {profile_path} missing")
        sys.exit(1)
    profile = json.loads(profile_path.read_text())

    llm_client = QueryLLM(
        {"models": {"llm_model": args.model}},
        rate_limit_per_min=args.rate_limit,
    )

    print(f"=== refresh_user_voiced_content for user {args.user_id} ===")
    print(f"  backend: {base}")
    print(f"  model:   {args.model}")
    print(f"  parallel chatbot calls: {args.parallel}")

    # 1) Extension B (self-posts + DMs)
    if not args.skip_extension_b:
        stripped = _strip_voiced_events(base)
        print(f"  stripped from app JSONs: {stripped}")
        report = run_extension_b(
            user_id=args.user_id,
            backend_dir=args.backend_dir,
            llm_client=llm_client,
            rng_seed=args.rng_seed,
            verbose=True,
        )
        print(f"  extension_b report: {report}")
    else:
        print("  (skipping Extension B)")

    # 2) Chatbot conversations
    if not args.skip_chatbot:
        # Reload profile in case Extension B touched it (it adds friends[]).
        profile = json.loads(profile_path.read_text())
        n = _refresh_chatbot_conversations(base, profile, llm_client, args.user_id, args.parallel)
        print(f"  refreshed chatbot conversations: {n}")
    else:
        print("  (skipping chatbot conversations)")

    # 3) @ai user_messages
    if not args.skip_at_ai:
        profile = json.loads(profile_path.read_text())
        n = _refresh_at_ai_user_messages(base, profile, llm_client)
        print(f"  refreshed @ai user_messages: {n}")
    else:
        print("  (skipping @ai messages)")

    # 4) Re-render persona.html so the operator can immediately spot-check.
    try:
        from data_preparation.visualize import generate_persona_html
        generate_persona_html(args.user_id, args.backend_dir)
        print(f"  re-rendered backend/{args.user_id}/persona.html")
    except Exception as exc:
        print(f"  WARN: persona.html re-render failed: {exc}")

    print("DONE.")


if __name__ == "__main__":
    main()
