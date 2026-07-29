#!/usr/bin/env python3
"""Backfill incomplete synthetic content payloads in shipped backend personas.

A minority of organic engagement events shipped with prose-only content
(title/caption/description) because the Step-19 mini-LLM occasionally dropped
the typed payload (image `parts`+`metadata`, short_video `key_frames`+
`metadata`) and the pipeline accepted the partial JSON. The pipeline now
validates + retries (see `content_payload_complete` in persona_agent.py);
this script repairs already-generated data in place.

Surgical: touches ONLY organic image/short_video events whose content is
incomplete — skips ads, DMs, self-authored posts, trending feed items, and
planted sensitive-event rows (their schemas differ by design). Every replaced
content payload is preserved in a rollback JSONL before the file is rewritten.

Usage:
  python scripts/backfill_content_fields.py                    # dry-run: count only
  python scripts/backfill_content_fields.py --apply            # repair all personas
  PERSONAS="3 8" python scripts/backfill_content_fields.py --apply --limit 2  # smoke
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preparation import prompts, utils  # noqa: E402
from data_preparation.persona_agent import content_payload_complete  # noqa: E402

APPS = {"instagram": "Instagram", "facebook": "Facebook", "threads": "Threads"}


def _organic_incomplete(e: dict) -> bool:
    c, t = e.get("content"), e.get("content_type")
    if not isinstance(c, dict) or t not in ("image", "short_video"):
        return False
    if content_payload_complete(t, c):
        return False
    if e.get("is_ad") or e.get("is_dm") or e.get("is_self_authored") \
       or e.get("is_trending") or e.get("feed_visible"):
        return False
    if any("planted" in str(k).lower() for k in list(e.keys()) + list(c.keys())):
        return False
    return True


def _frame_lookup(profile: dict):
    tag2frame: dict[str, tuple[str, int]] = {}
    for hp in profile.get("hidden_personas") or []:
        if not isinstance(hp, dict) or hp.get("is_synthetic"):
            continue
        frame = prompts.cluster_dominant_frame(SimpleNamespace(**hp))
        if not frame or frame == "none":
            continue
        rows = int(hp.get("evidence_rows") or 0)
        for tag in hp.get("evidence_hashtags") or []:
            tag_n = (tag or "").lower().lstrip("#").strip()
            if tag_n and (tag_n not in tag2frame or rows > tag2frame[tag_n][1]):
                tag2frame[tag_n] = (frame, rows)

    def frame_for(hashtags):
        best, score = "", 0
        for tag in hashtags or []:
            hit = tag2frame.get((tag or "").lower().lstrip("#").strip())
            if hit and hit[1] > score:
                best, score = hit
        return (best, prompts.FRAME_DESCRIPTIONS.get(best, "")) if best else ("", "")
    return frame_for


def repair_user(uid: str, client, args, rollback_fh) -> tuple[int, int]:
    udir = Path(args.backend_dir) / uid
    try:
        profile = json.load(open(udir / "profile.json"))
    except (OSError, ValueError):
        return 0, 0
    user_profile_dict = {k: profile.get(k) for k in
                         ("name", "gender", "race_ethnicity", "career", "education", "bio")}
    app_personas = profile.get("app_personas") or {}
    frame_for = _frame_lookup(profile)

    fixed = failed = 0
    for fname, app in APPS.items():
        path = udir / f"{fname}.json"
        if not path.exists():
            continue
        events = json.load(open(path))
        targets = [(i, e) for i, e in enumerate(events) if _organic_incomplete(e)]
        if not targets:
            continue
        if args.limit:
            targets = targets[: args.limit]

        def _one(item):
            i, e = item
            iface = e.get("interaction_format") or {}
            prefs = [{"persona_item": p.get("persona_item"), "category": p.get("category")}
                     for p in (e.get("preferences") or [])]
            frame, frame_desc = frame_for(e.get("source_hashtags") or [])
            prompt = prompts.generate_synthetic_content_prompt(
                content_type=e["content_type"], app=app,
                app_persona=app_personas.get(app) or {},
                user_profile=user_profile_dict,
                hashtags=e.get("source_hashtags") or [],
                preferences=prefs,
                action=iface.get("action") or "unknown",
                action_label=iface.get("action_label") or "",
                motivation_frame=frame or None,
                motivation_frame_description=frame_desc or None,
            )
            for _ in range(2):
                try:
                    parsed = utils.extract_json_from_response(
                        client.query_llm(prompt, verbose=False))
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("content"), dict):
                    parsed = parsed["content"]
                if isinstance(parsed, dict) and content_payload_complete(e["content_type"], parsed):
                    return i, parsed
            return i, None

        changed = False
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed({ex.submit(_one, t) for t in targets}):
                i, new_content = fut.result()
                if new_content is None:
                    failed += 1
                    continue
                rollback_fh.write(json.dumps({
                    "uid": uid, "app": fname, "index": i,
                    "source_object_id": events[i].get("source_object_id"),
                    "old_content": events[i].get("content"),
                }) + "\n")
                events[i]["content"] = new_content
                fixed += 1
                changed = True
        if changed and args.apply:
            tmp = str(path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(events, f, indent=1, ensure_ascii=False)
            os.replace(tmp, path)
    return fixed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write repairs (default: dry-run count)")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--mini_model", default="gpt-5.4-mini")
    ap.add_argument("--rate_limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="max events per app file (smoke runs)")
    args = ap.parse_args()

    users = os.environ.get("PERSONAS", "").split() or sorted(
        (d for d in os.listdir(args.backend_dir) if d.isdigit()), key=int)

    if not args.apply:
        n = 0
        for uid in users:
            for fname in APPS:
                p = Path(args.backend_dir) / uid / f"{fname}.json"
                if p.exists():
                    n += sum(1 for e in json.load(open(p)) if _organic_incomplete(e))
        print(f"[dry-run] {n} incomplete organic events across {len(users)} personas "
              f"(rerun with --apply to repair)")
        return

    from query_llm import QueryLLM
    client = QueryLLM({"models": {"llm_model": args.mini_model}},
                      rate_limit_per_min=args.rate_limit)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    rb_path = Path("backups") / f"content_backfill_rollback_{stamp}.jsonl"
    rb_path.parent.mkdir(exist_ok=True)
    tot_fixed = tot_failed = 0
    with open(rb_path, "w") as rb:
        for uid in users:
            fixed, failed = repair_user(uid, client, args, rb)
            if fixed or failed:
                print(f"[user {uid}] repaired {fixed}, unrecoverable {failed}", flush=True)
            tot_fixed += fixed
            tot_failed += failed
    print(f"DONE: repaired {tot_fixed}, unrecoverable {tot_failed}; rollback at {rb_path}")


if __name__ == "__main__":
    main()
