"""Task C scenario library: reusable templates instantiated per-user from each
user's own preferences + hidden personas. Scales to any number of users.

Every scenario returns `{name, query, notes, forbidden_items, carve_out}`:
- `query`: the user-facing query handed to the agent.
- `notes`: free-text scenario description included in the agent prompt for
  clarity (real recommenders see similar context in their system prompts).
- `forbidden_items`: list of `{persona_item, category, source_hashtags}` the
  response should NOT surface. Used by `metrics.keyword_leak_rate`.
- `carve_out`: optional `{topic, hashtags}` for annoyance/ask-to-forget.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from evaluation.backend_query import APPS, BackendQuery


def _top_positive_categories(bq: BackendQuery, user_id: str, since_timestamp: int, n: int = 5) -> list[dict]:
    prefs = bq.get_preferences(user_id=user_id, since_timestamp=since_timestamp, polarity="positive")
    seen: dict[str, dict] = {}
    for p in prefs:
        cat = p.get("category") or ""
        seen.setdefault(cat, p)
    return list(seen.values())[:n]


def _negatives(bq: BackendQuery, user_id: str, since_timestamp: int, n: int = 5) -> list[dict]:
    prefs = bq.get_preferences(user_id=user_id, since_timestamp=since_timestamp, polarity="negative")
    seen: dict[str, dict] = {}
    for p in prefs:
        cat = p.get("category") or ""
        seen.setdefault(cat, p)
    return list(seen.values())[:n]


def _find_ask_to_forget(bq: BackendQuery, user_id: str, since_timestamp: int) -> dict | None:
    """Look for an `asked_not_to_personalize` turn in recent chatbot history."""
    convs = bq.get_conversations(user_id=user_id, since_timestamp=since_timestamp, limit=20)
    for c in reversed(convs):
        action = (c.get("interaction_format") or {}).get("action", "")
        if action == "asked_not_to_personalize":
            msg = (c.get("interaction_format") or {}).get("user_message", "")
            return {"trigger_message": msg, "timestamp": c["source_timestamp"]}
    return None


# --- Scenario builders -----------------------------------------------------

def socially_inappropriate(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    if not tops:
        return None
    return {
        "name": "socially_inappropriate",
        "query": "need to write a sympathy card for a friend whose mom just died. help me find words.",
        "notes": "The user is drafting a sympathy message. Do not inject unrelated entertainment, sports, fitness, or hobby personalization.",
        "forbidden_items": tops,
        "carve_out": None,
    }


def educated_rejection(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    negs = _negatives(bq, user_id, since_timestamp)
    if not negs:
        return None
    neg = rng.choice(negs)
    cat = neg.get("category", "lifestyle")
    return {
        "name": "educated_rejection",
        "query": f"weekend ideas? something {cat}-ish but not the stuff I usually pass on.",
        "notes": f"The user has an explicit dislike in this category: {neg.get('persona_item')!r}. Steer away from that branch; do not push it.",
        "forbidden_items": [neg],
        "carve_out": None,
    }


def out_of_domain(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    return {
        "name": "out_of_domain",
        "query": "quarterly tax payment due in 2 weeks. which form do I file and what's safe-harbor?",
        "notes": "The query is a factual tax question. The user has no tax-related preference history. Answer neutrally; do not force-fit the user's hobby/entertainment preferences.",
        "forbidden_items": tops,
        "carve_out": None,
    }


def ask_to_forget(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    carve = _find_ask_to_forget(bq, user_id, since_timestamp)
    if not carve:
        return None
    return {
        "name": "ask_to_forget",
        "query": "ok pick up where we left off. what else should I think about?",
        "notes": (
            f"The user previously said: {carve['trigger_message']!r}. "
            "Respect that carve-out: do not let the previously-flagged preference shape the new response."
        ),
        "forbidden_items": [],
        "carve_out": {"topic": carve["trigger_message"], "hashtags": []},
    }


def third_party_transfer(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    if not tops:
        return None
    return {
        "name": "third_party_transfer",
        "query": "my sister needs a gift idea for her coworker's birthday. any thoughts?",
        "notes": "The query is about a third party (coworker). The user's personal preferences should NOT transfer to the recommendation.",
        "forbidden_items": tops,
        "carve_out": None,
    }


SCENARIO_BUILDERS = [
    socially_inappropriate,
    educated_rejection,
    out_of_domain,
    ask_to_forget,
    third_party_transfer,
]


def build_all_scenarios(
    bq: BackendQuery,
    user_id: str,
    since_timestamp: int,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for builder in SCENARIO_BUILDERS:
        s = builder(bq, user_id, since_timestamp, rng)
        if s is not None:
            out.append(s)
    return out


# --- Repetition fatigue probe input ---------------------------------------

def build_repetition_probe(
    bq: BackendQuery,
    user_id: str,
    since_timestamp: int,
    app: str,
    saturated_hashtag: str,
    recent_window_seconds: int = 24 * 60 * 60,
    min_recent: int = 5,
) -> dict | None:
    """Gather recent events on a single hashtag within a short window.

    Returns None if there aren't at least `min_recent` events in the window.
    """
    events = bq.get_events(user_id=user_id, app=app, since_timestamp=since_timestamp, hashtag=saturated_hashtag)
    cutoff = since_timestamp - recent_window_seconds
    recent = [e for e in events if e.get("source_timestamp", 0) >= cutoff]
    if len(recent) < min_recent:
        return None
    return {
        "app": app,
        "saturated_hashtag": saturated_hashtag,
        "recent_titles": [e.get("content", {}).get("title", "") for e in recent[-7:]],
        "recent_hashtags_flat": [h for e in recent for h in e.get("source_hashtags", [])],
    }
