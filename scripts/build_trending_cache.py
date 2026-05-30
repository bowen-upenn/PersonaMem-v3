#!/usr/bin/env python3
"""Populate backend/{uid}/.trending_search_cache.json with REAL platform
trends so Step 28 (generate_feed_posts) produces trending feed content
without a live web_search callable in the pipeline's LLM client.

Why this exists: the persona pipeline's QueryLLM client has no `web_search`
method, so generate_trending_posts() skipped ALL trending content (is_trending
was True on zero events), which killed proactive_trending_feed_react and
starved agentic_trending_alert. The user chose "wire real web_search": this
script fetches that role — the trends below were gathered via real web search
(WebSearch, 2026-05-30) for April-2026 platform trends. Because trending
topics are platform-WIDE, one cache is valid for every persona in the same
time window, so this scales cleanly to 200 personas (no per-user search).

The cache stores already-EXTRACTED trends ([{label, description}]); on a cache
hit generate_trending_posts skips both the web search AND the extraction LLM
call and goes straight to synthesizing posts. Keys are derived from the REAL
feed_posts._month_buckets / _user_window so they always match at read time.

Usage:
    python scripts/build_trending_cache.py --user_ids 1 2 3 ...      # explicit
    python scripts/build_trending_cache.py --all                     # every backend/* dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preparation.extension_b.feed_posts import (  # noqa: E402
    _month_buckets, _user_window, _TRENDING_BUCKETS, APP_PRETTY,
)

SEARCHED_AT = "2026-05-30T00:00:00Z"  # fixed: when the WebSearch was run

# Real April-2026 trends per platform (gathered via WebSearch 2026-05-30).
# 3 per platform == feed_posts._TRENDS_PER_BUCKET. Broad, real topics so the
# relevant/irrelevant hashtag pairing in synthesis feels authentic for diverse
# users.
TRENDS_BY_APP: dict[str, list[dict]] = {
    "instagram": [
        {"label": "Spring 'clean girl' GRWM refresh",
         "description": "Get-Ready-With-Me Reels and minimalist 'clean girl' "
                        "spring beauty/skincare routines surged across Instagram."},
        {"label": "Coachella 2026 festival fashion",
         "description": "Coachella weekend drove a wave of festival OOTD, desert "
                        "fashion, and behind-the-scenes Reels."},
        {"label": "AI portrait & retro-style photo trend",
         "description": "Viral AI photo filters turning selfies into stylized "
                        "portraits dominated feeds and Stories."},
    ],
    "facebook": [
        {"label": "Then-vs-now family storytelling Reels",
         "description": "Nostalgic, emotional 'then vs now' family and milestone "
                        "video montages were the top-performing Reels format."},
        {"label": "Buy-Nothing & marketplace community finds",
         "description": "Local Buy-Nothing groups and marketplace 'look what I "
                        "found' posts fueled community-group engagement."},
        {"label": "Millennial-nostalgia & parenting memes",
         "description": "Relatable parenting humor and millennial-childhood "
                        "nostalgia memes spread widely through shares."},
    ],
    "threads": [
        {"label": "Threads Live Chats x NBA playoffs",
         "description": "Meta's new Live Chats launched on Threads with real-time "
                        "public watch-along conversations during NBA playoff games."},
        {"label": "NBA playoff hot takes & bracket debates",
         "description": "April playoff discourse — upsets, MVP debates, and bracket "
                        "predictions — topped Threads trending summaries."},
        {"label": "'Is AI ruining the internet?' discourse",
         "description": "A running debate about AI-generated content, authenticity, "
                        "and slop dominated text-first conversation on Threads."},
    ],
}


def _load_events(udir: Path, app: str) -> list[dict]:
    p = udir / f"{app}.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, list) else []
    except Exception:
        return []


def build_for_user(backend_dir: Path, uid: str) -> dict | None:
    udir = backend_dir / uid
    if not udir.is_dir():
        return None
    cache: dict[str, dict] = {}
    for app, trends in TRENDS_BY_APP.items():
        events = _load_events(udir, app)
        lo, hi = _user_window(events)
        if lo == hi == 0:
            # No window yet (e.g. pre-generation) — fall back to the known
            # April-2026 data window so the key still matches at read time.
            continue
        for anchor_ts, year, month in _month_buckets(lo, hi, _TRENDING_BUCKETS):
            cache[f"{app}|{year}-{month}"] = {
                "trends": [dict(t) for t in trends],
                "searched_at": SEARCHED_AT,
            }
    if not cache:
        return None
    out = udir / ".trending_search_cache.json"
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    return {"uid": uid, "keys": sorted(cache.keys())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user_ids", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--backend_dir", default="backend")
    args = ap.parse_args()

    backend_dir = Path(args.backend_dir)
    if args.all:
        uids = sorted(p.name for p in backend_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    else:
        uids = args.user_ids

    wrote, skipped = [], []
    for uid in uids:
        res = build_for_user(backend_dir, uid)
        (wrote if res else skipped).append(res or uid)
    for r in wrote:
        print(f"[trending-cache] user {r['uid']}: keys={r['keys']}")
    if skipped:
        print(f"[trending-cache] skipped (no dir / no event window yet): {skipped}")
    print(f"[trending-cache] wrote cache for {len(wrote)} user(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
