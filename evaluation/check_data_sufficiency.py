"""Pre-build data-sufficiency gate.

Checks backend/{user_id}/ against the plan's minimum-volume assertions for
Tasks T6–T19. Green = benchmark can be built; red = block with an actionable
error message.

Assertions (from plan Extension D₀):
- ≥ 10 self-authored posts per social app (ceiling → floor 5 if posting_frequency=rarely)
- ≥ 25 inbound DMs across the four apps combined
- ≥ 15 outbound DMs (sent by the user)
- ≥ 3 group-DM threads (each ≥ 3 participants, ≥ 3 messages)
- ≥ 8 named friends
- ≥ 3 hidden_personas with privacy_ratio > 0.7
- ≥ 2 topics (hashtags) appearing on events from ≥ 2 apps
- trending.json exists with ≥ 20 hashtags

CLI:
    python -m evaluation.check_data_sufficiency --user_id 115
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


APPS = ("instagram", "facebook", "threads", "chatbot")
SOCIAL_APPS = ("instagram", "facebook", "threads")


def _load(path: Path) -> list | dict:
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def check(user_id: str, backend_dir: str | Path) -> list[tuple[str, bool, str]]:
    """Return a list of (check_name, passed, note) tuples."""
    base = Path(backend_dir) / user_id
    profile = _load(base / "profile.json") or {}

    results: list[tuple[str, bool, str]] = []

    # Self-posts per social app.
    for app in SOCIAL_APPS:
        events = _load(base / f"{app}.json") or []
        self_posts = [e for e in events if e.get("is_self_authored")]
        posting_freq = ((profile.get("app_personas", {}) or {}).get(app.capitalize(), {}) or {}).get("posting_frequency", "weekly")
        floor = 5 if posting_freq == "rarely" else 10
        ok = len(self_posts) >= floor
        results.append((
            f"self_posts_{app}",
            ok,
            f"{len(self_posts)} self-posts ({posting_freq}, floor={floor})",
        ))

    # DM counts across apps. DM threads now live inline in {app}.json as
    # entries with is_dm=true + full messages[]; count messages per sender
    # (each thread carries 1–4 messages).
    inbound = 0
    outbound = 0
    group_threads = 0
    for app in SOCIAL_APPS:
        events = _load(base / f"{app}.json") or []
        for e in events:
            if not e.get("is_dm"):
                continue
            msgs = e.get("messages") or []
            for m in msgs:
                if m.get("sender") == "self":
                    outbound += 1
                else:
                    inbound += 1
            if (e.get("is_group") or e.get("is_group_dm")) \
                    and len(e.get("participants") or []) >= 3 \
                    and len(msgs) >= 3:
                group_threads += 1
    results.append(("inbound_dms_total", inbound >= 25, f"{inbound} inbound DMs (need ≥25)"))
    results.append(("outbound_dms_total", outbound >= 15, f"{outbound} outbound DMs (need ≥15)"))
    results.append(("group_dm_threads", group_threads >= 3, f"{group_threads} group threads (need ≥3)"))

    # Friends graph.
    friends = profile.get("friends") or []
    results.append(("named_friends", len(friends) >= 8, f"{len(friends)} friends (need ≥8)"))

    # Privacy-sensitive hidden personas.
    hidden = profile.get("hidden_personas") or []
    sensitive = [h for h in hidden if (h.get("privacy_ratio") or 0) > 0.7]
    results.append(("sensitive_hidden_personas", len(sensitive) >= 3, f"{len(sensitive)} with privacy_ratio>0.7 (need ≥3)"))

    # Multi-app topics.
    app_hashtag_union: dict[str, set[str]] = {}
    for app in APPS:
        events = _load(base / f"{app}.json") or []
        tags = set()
        for e in events:
            for h in (e.get("source_hashtags") or []):
                tags.add(h.lower())
        app_hashtag_union[app] = tags
    shared: Counter = Counter()
    for a in APPS:
        for b in APPS:
            if a >= b:
                continue
            shared.update(app_hashtag_union[a] & app_hashtag_union[b])
    multi_app_topics = len({h for h, _ in shared.most_common() if shared[h] >= 1})
    results.append(("multi_app_topics", multi_app_topics >= 2, f"{multi_app_topics} hashtags on ≥2 apps (need ≥2)"))

    # Trending.
    trending = _load(base / "trending.json") or {}
    trending_hashtags = trending.get("hashtags", []) if isinstance(trending, dict) else []
    results.append(("trending_hashtags", len(trending_hashtags) >= 20, f"{len(trending_hashtags)} trending tags (need ≥20)"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Check Extension B data sufficiency for benchmark build.")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails")
    args = parser.parse_args()

    results = check(args.user_id, args.backend_dir)
    green = "\033[92m"; red = "\033[91m"; end = "\033[0m"
    failed = []
    for name, ok, note in results:
        tag = f"{green}✓{end}" if ok else f"{red}✗{end}"
        print(f"  {tag} {name:30s}  {note}")
        if not ok:
            failed.append(name)
    print()
    if failed:
        print(f"{red}{len(failed)} check(s) failed:{end} {', '.join(failed)}")
        print("Run Extension B to close the gap:")
        print(f"  python -m data_preparation.extension_b --user_id {args.user_id}")
        if args.strict:
            sys.exit(1)
    else:
        print(f"{green}all data-sufficiency checks passed.{end}")


if __name__ == "__main__":
    main()
