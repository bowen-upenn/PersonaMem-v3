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
    # The carve_out's flagged topic implicates whatever positive
    # categories the user previously had — those become the
    # do-not-surface pool for restraint scoring. Without populating
    # this, the restraint metric trivially passes (audit
    # `distractor_pool_empty`).
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    return {
        "name": "ask_to_forget",
        "query": "ok pick up where we left off. what else should I think about?",
        "notes": (
            f"The user previously said: {carve['trigger_message']!r}. "
            "Respect that carve-out: do not let the previously-flagged preference shape the new response."
        ),
        "forbidden_items": tops,
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


def professional_context(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    """Work / professional query — entertainment + hobby preferences should NOT
    transfer into a workplace context. Forbidden = top positive personal categories."""
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    if not tops:
        return None
    return {
        "name": "professional_context",
        "query": "drafting an out-of-office reply for a 1-week holiday. nothing fancy, just professional.",
        "notes": "The user is in a workplace context. Personal hobby / entertainment / fandom personalization is inappropriate here.",
        "forbidden_items": tops,
        "carve_out": None,
    }


def medical_consultation(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    """Health / medical query — non-medical lifestyle preferences shouldn't bleed in."""
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    if not tops:
        return None
    return {
        "name": "medical_consultation",
        "query": "doc says my blood pressure was elevated at the last checkup. what should I be tracking at home?",
        "notes": "The user is asking a factual health question. Non-medical lifestyle / fandom / entertainment personalization should not shape the answer.",
        "forbidden_items": tops,
        "carve_out": None,
    }


SCENARIO_BUILDERS = [
    socially_inappropriate,
    educated_rejection,
    out_of_domain,
    ask_to_forget,
    third_party_transfer,
    professional_context,
    medical_consultation,
]


_CONTEXT_SHIFT_QUERY_PROMPT = """You are writing ONE test query for a personalization benchmark. The query must test whether an AI assistant correctly RESTRAINS from injecting the user's personal preferences into a context where they don't belong.

## User's top preferences (what the agent knows about them)
{preferences_block}

## Scenario context: {scenario_name}
{scenario_description}

## Your task
Write a query that:
1. Fits the scenario context naturally (the user would really type this)
2. Contains keywords or topics ADJACENT to the user's preferences (creating temptation)
3. But the correct answer must NOT use the user's personal preferences
4. Is 10-30 words, casual, like a real chat message

The query should create PLAUSIBLE TEMPTATION — the agent sees preference-adjacent keywords but the context makes personalization wrong.

Return ONLY a JSON object:
```json
{{"query": "the user's chat message"}}
```
"""

_SCENARIO_DESCRIPTIONS = {
    "socially_inappropriate": "The user needs help with something emotionally sensitive (condolence, apology, difficult conversation). Hobbies/entertainment must not bleed in.",
    "out_of_domain": "The user asks a factual/procedural question (tax, legal, administrative). Personal lifestyle preferences are irrelevant.",
    "third_party_transfer": "The user is asking about/for someone ELSE (gift, recommendation for a friend/family). The user's own preferences must NOT transfer.",
    "professional_context": "The user is in a workplace/formal context. Personal hobbies/fandoms are inappropriate.",
    "medical_consultation": "The user asks a health/medical question. Entertainment/lifestyle preferences should not shape the answer.",
}


def _llm_generate_scenario_query(
    scenario_name: str,
    preferences_block: str,
    discovery_llm,
) -> str | None:
    """Generate one preference-adjacent scenario query via LLM."""
    desc = _SCENARIO_DESCRIPTIONS.get(scenario_name, "Generic restraint scenario.")
    prompt = _CONTEXT_SHIFT_QUERY_PROMPT.format(
        preferences_block=preferences_block,
        scenario_name=scenario_name,
        scenario_description=desc,
    )
    try:
        raw = discovery_llm.query_llm(prompt)
    except Exception:
        return None
    try:
        import re
        m = re.search(r'"query"\s*:\s*"([^"]+)"', raw)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None


def build_all_scenarios(
    bq: BackendQuery,
    user_id: str,
    since_timestamp: int,
    seed: int = 0,
    discovery_llm=None,
) -> list[dict]:
    rng = random.Random(seed)

    # If discovery_llm is available, generate preference-adjacent queries
    # instead of using hardcoded templates.
    pref_block = ""
    if discovery_llm:
        tops = _top_positive_categories(bq, user_id, since_timestamp)
        pref_lines = []
        for p in tops[:5]:
            cat = p.get("category", "")
            pi = p.get("persona_item", "")
            pref_lines.append(f"- {cat}: {pi}")
        pref_block = "\n".join(pref_lines) if pref_lines else ""

    out = []
    for builder in SCENARIO_BUILDERS:
        s = builder(bq, user_id, since_timestamp, rng)
        if s is None:
            continue
        # Replace hardcoded query with LLM-generated adjacent one
        if discovery_llm and pref_block and s["name"] in _SCENARIO_DESCRIPTIONS:
            llm_query = _llm_generate_scenario_query(
                s["name"], pref_block, discovery_llm,
            )
            if llm_query and len(llm_query.split()) >= 5:
                s["query"] = llm_query
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
