#!/usr/bin/env python3
"""Verify the output of run_feed_posts_for_user.py is correctly shaped.

Reports:
  - feed_visible event counts per app, split by author_id kind (friend vs public)
  - proactive_trigger_candidates keys and counts in profile.json
  - any obvious schema problems (missing fields, mismatched relevance)

Usage: python scripts/check_feed_posts_output.py 115
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        print("Usage: check_feed_posts_output.py <user_id>")
        sys.exit(2)
    uid = sys.argv[1]
    base = REPO_ROOT / "backend" / uid
    if not base.exists():
        print(f"backend/{uid} missing")
        sys.exit(1)

    print(f"\n=== feed_visible events in app JSONs (user {uid}) ===\n")
    for app in ("instagram", "facebook", "threads"):
        p = base / f"{app}.json"
        if not p.exists():
            print(f"  {app}.json: missing")
            continue
        events = json.loads(p.read_text())
        feed_vis = [e for e in events if e.get("source_interaction_type") == "feed_visible"]
        by_author = Counter()
        relevance_counts = Counter()
        for e in feed_vis:
            aid = e.get("author_id", "?")
            kind = "friend" if aid.startswith("friend_") else aid
            by_author[kind] += 1
            meta = e.get("_feed_react_meta") or {}
            relevance_counts[meta.get("relevance", "?")] += 1
        print(f"  {app}.json: {len(events):>4} total events; "
              f"{len(feed_vis):>3} feed_visible "
              f"({dict(by_author)} authors; {dict(relevance_counts)} relevance)")
        # Show a couple of samples
        for kind in ("friend", "public_creator"):
            sample = next((e for e in feed_vis
                          if (e.get("author_id", "").startswith("friend_") if kind == "friend"
                              else e.get("author_id") == "public_creator")), None)
            if sample:
                print(f"    sample [{kind}]: "
                      f"ts={sample.get('formatted_timestamp')!r}, "
                      f"hashtags={sample.get('source_hashtags')!r}, "
                      f"relevance={(sample.get('_feed_react_meta') or {}).get('relevance')!r}")
                content = sample.get("content") or {}
                cap = (content.get("caption") or "")[:120]
                print(f"      caption: {cap!r}")

    print(f"\n=== profile.json proactive_trigger_candidates ===\n")
    pp = base / "profile.json"
    if not pp.exists():
        print("  profile.json missing")
        return
    p = json.loads(pp.read_text())
    cands = p.get("proactive_trigger_candidates") or {}
    if not cands:
        print("  proactive_trigger_candidates: EMPTY or missing")
        return
    for k, v in cands.items():
        rel_counts = Counter(c.get("relevance", "n/a") for c in v)
        print(f"  {k:>32}: {len(v):>2} candidates  ({dict(rel_counts)})")

    # Spot-check one of each new type
    print(f"\n=== Sample new-type candidate ===\n")
    for tname in ("friend_feed_react", "trending_feed_react", "overactive_check"):
        items = cands.get(tname) or []
        if items:
            print(f"\n  --- {tname} (sample) ---")
            print(json.dumps(items[0], indent=2, ensure_ascii=False)[:1200])
        else:
            print(f"\n  {tname}: zero candidates (might be normal if user lacks the precondition)")


if __name__ == "__main__":
    main()
