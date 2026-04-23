"""Deterministically generate trending.json — no LLM call needed.

Composition:
- Top 15 hashtags ranked by user's positive engagement counts in the backend.
- 5 additional plausible off-user hashtags drawn from the user's negative
  events (topics they're aware of even if they dislike them) + a few
  demographically-plausible general ones.

trending.json shape:
    {"built_at": "...", "hashtags": [
       {"hashtag": "#NFL", "rank": 1, "post_ids": [...], "user_aligned": true},
       ...
     ]}

MCP `*_search(type="hashtag", query="__trending__")` returns this list.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def _count_hashtags(events: list[dict], polarity: str) -> dict[str, list[str]]:
    """Returns hashtag → [event_ids] for events with the given polarity."""
    out: dict[str, list[str]] = {}
    for e in events:
        it = e.get("source_interaction_type", "")
        if polarity == "positive" and "positive" not in it:
            continue
        if polarity == "negative" and "negative" not in it:
            continue
        eid = str(e.get("source_object_id", ""))
        for h in e.get("source_hashtags", []) or []:
            out.setdefault(h, []).append(eid)
    return out


def generate_trending(user_id: str, backend_dir: Path) -> dict:
    base = Path(backend_dir) / user_id
    all_events: list[dict] = []
    for app in ("instagram", "facebook", "threads", "chatbot"):
        p = base / f"{app}.json"
        if p.exists():
            with p.open() as f:
                all_events.extend(json.load(f))

    pos = _count_hashtags(all_events, "positive")
    neg = _count_hashtags(all_events, "negative")

    # Aligned: user's top 15 positive hashtags.
    aligned = sorted(pos.items(), key=lambda kv: len(kv[1]), reverse=True)[:15]
    aligned_out = [
        {"hashtag": h, "rank": i + 1, "post_ids": eids[:10], "user_aligned": True}
        for i, (h, eids) in enumerate(aligned)
    ]

    # Off-user: top 5 negative hashtags the user is aware of but dislikes.
    aligned_tags = {h.lower() for h in pos.keys()}
    off_user = [
        (h, eids) for h, eids in sorted(neg.items(), key=lambda kv: len(kv[1]), reverse=True)
        if h.lower() not in aligned_tags
    ][:5]
    off_user_out = [
        {"hashtag": h, "rank": len(aligned_out) + i + 1, "post_ids": eids[:10], "user_aligned": False}
        for i, (h, eids) in enumerate(off_user)
    ]

    return {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hashtags": aligned_out + off_user_out,
    }
