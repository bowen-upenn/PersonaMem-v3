#!/usr/bin/env python3
"""Walk backend/{uid}/{instagram,facebook,threads}.json and re-apply the
DM interaction-type rule from data_preparation/extension_b/dm_threads.py
to every is_dm event. Deterministic — no LLM calls, no new content.

Reason: the old labeler keyed off last_sender only, producing
mislabels (e.g., friend-replied threads marked implicit_positive even
when the user reacted positively). The new rule classifies by
initiator + user-response presence + user-response polarity.

Usage:
  python scripts/relabel_dm_interaction_types.py --user_id 115
  python scripts/relabel_dm_interaction_types.py --all
  python scripts/relabel_dm_interaction_types.py --user_id 115 --dry_run
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

from data_preparation.extension_b.dm_threads import _self_responded_positively


def _classify(messages: list[dict], friend_ids: set[str]) -> str:
    if not messages:
        return "implicit_positive"
    non_self = [m for m in messages if m.get("sender") != "self"]
    self_msgs = [m for m in messages if m.get("sender") == "self"]
    first_sender = messages[0].get("sender")
    # User-initiated share (first message from self) is explicit_positive
    # regardless of whether the friend reacts back.
    if first_sender == "self" or not non_self:
        return "explicit_positive"
    initiator = non_self[0].get("sender")
    initiator_is_friend = initiator in friend_ids
    if not self_msgs:
        return "implicit_positive" if initiator_is_friend else "implicit_negative"
    return (
        "explicit_positive" if _self_responded_positively(self_msgs)
        else "implicit_positive"
    )


def _load_friend_ids(user_dir: Path) -> set[str]:
    profile_path = user_dir / "profile.json"
    if not profile_path.exists():
        return set()
    profile = json.loads(profile_path.read_text())
    return {
        f.get("friend_id") for f in (profile.get("friends") or [])
        if f.get("friend_id")
    }


def relabel_user(user_id: str, backend_dir: str, dry_run: bool) -> dict:
    user_dir = Path(backend_dir) / str(user_id)
    friend_ids = _load_friend_ids(user_dir)
    stats = {"user_id": user_id, "n_dms": 0, "n_changed": 0, "by_app": {}}
    for app in ("instagram", "facebook", "threads"):
        path = user_dir / f"{app}.json"
        if not path.exists():
            continue
        evs = json.loads(path.read_text())
        n_changed = 0
        for e in evs:
            if not e.get("is_dm"):
                continue
            stats["n_dms"] += 1
            messages = e.get("messages") or []
            if not messages:
                continue
            old = e.get("source_interaction_type")
            new = _classify(messages, friend_ids)
            if new != old:
                n_changed += 1
                e["source_interaction_type"] = new
        stats["by_app"][app] = n_changed
        stats["n_changed"] += n_changed
        if not dry_run and n_changed > 0:
            path.write_text(json.dumps(evs, indent=2, ensure_ascii=False))
    return stats


def _resolve_user_ids(args: argparse.Namespace) -> list[str]:
    if args.user_id:
        return [args.user_id]
    if args.all:
        backend = Path(args.backend_dir)
        return sorted(p.name for p in backend.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--user_id")
    grp.add_argument("--all", action="store_true")
    p.add_argument("--backend_dir", default="backend")
    p.add_argument("--dry_run", action="store_true",
                   help="report what would change without writing")
    args = p.parse_args()

    user_ids = _resolve_user_ids(args)
    if not user_ids:
        print("no users resolved", file=sys.stderr)
        return 2

    total_changed = 0
    for uid in user_ids:
        stats = relabel_user(uid, args.backend_dir, args.dry_run)
        action = "would relabel" if args.dry_run else "relabeled"
        print(f"[{uid}] {action} {stats['n_changed']}/{stats['n_dms']} DMs   "
              f"by_app={stats['by_app']}")
        total_changed += stats["n_changed"]
    print(f"\n{'(dry run) ' if args.dry_run else ''}total: {total_changed} relabeled "
          f"across {len(user_ids)} user(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
