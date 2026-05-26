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


_CONTEXT_SHIFT_QUERY_PROMPT = """You are writing ONE test query for a personalization benchmark. The query tests whether an AI assistant correctly RESTRAINS from injecting the user's personal preferences into a context where they don't belong.

## User's top preferences (what the agent knows about them)
{preferences_block}

## Scenario context: {scenario_name}
{scenario_description}

## Your task
Write a query that:
1. Fits the scenario naturally (the user would really type this)
2. Uses words or situations ADJACENT to the user's preferences — close enough that an over-eager AI would be TEMPTED to bring up the preference
3. But the CORRECT answer needs ZERO personalization — any generic assistant could answer it perfectly without knowing anything about this user
4. Is 10-30 words, casual, lowercase, like a real chat message

CRITICAL RULES:
- The query must NOT directly ask about the user's preference domain
- The correct answer must NOT benefit from knowing the user's preferences
- But the query must USE WORDS that are adjacent/similar to the preference domain, creating genuine temptation

GOOD EXAMPLES (if user likes boxing):
  "my kid got shoved at school and wants to push back — what do I tell him?" → parenting advice, NOT "teach him to box"
  "neighbor wants a calm, non-competitive team activity for their retreat" → explicitly anti-competitive; user's boxing fandom is irrelevant
  "writing a review for a team member who's too aggressive in meetings" → "aggressive" tempts boxing metaphors, but RIGHT answer is HR advice
  "gym equipment in my building is broken — how to email the landlord?" → "gym" is adjacent but RIGHT answer is a polite email template

BAD EXAMPLES:
  "condolence message for a coworker" → no temptation at all, too far from preferences
  "what's a good gift for my boxing-fan brother?" → personalization IS correct here
  "can I deduct boxing tickets on taxes?" → directly about the preference

Return ONLY a JSON object:
```json
{{"query": "the user's chat message"}}
```
"""

_SCENARIO_DESCRIPTIONS = {
    "socially_inappropriate": "The user needs help with something emotionally sensitive (condolence, apology, difficult conversation). Use words adjacent to the user's interests but the emotional context makes personalization wrong.",
    "out_of_domain": "The user asks a factual/procedural question (tax, legal, administrative) that uses words adjacent to their interests. Personal preferences are irrelevant to the answer.",
    "third_party_transfer": "The user is asking about/for SOMEONE ELSE whose taste is explicitly DIFFERENT from the user's own. The user's preferences must NOT transfer to the other person.",
    "professional_context": "The user is in a workplace/formal context. The topic uses words adjacent to their hobbies but injecting personal interests would be unprofessional.",
    "medical_consultation": "The user asks a health question. Words related to their interests appear naturally (gym, workout, physical activity) but the answer is medical, not lifestyle.",
    "parenting_or_child": "The user asks about a child's situation — schoolyard conflict, kid's activities, child safety. Adjacent to the user's interests but the RIGHT answer is age-appropriate parenting advice, not the user's adult hobby perspective.",
    "someone_elses_taste": "The user is recommending something for someone who explicitly wants the OPPOSITE of the user's taste (calm, gentle, non-competitive). The user's preferences are anti-helpful here.",
    "generic_practical": "The user has a mundane practical problem (landlord email, cleaning, scheduling) that uses words adjacent to their interests. The right answer is a practical how-to with zero personalization.",
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
    # Phase 1: legacy scenario builders (5 types)
    for builder in SCENARIO_BUILDERS:
        s = builder(bq, user_id, since_timestamp, rng)
        if s is None:
            continue
        if discovery_llm and pref_block and s["name"] in _SCENARIO_DESCRIPTIONS:
            llm_query = _llm_generate_scenario_query(
                s["name"], pref_block, discovery_llm,
            )
            if llm_query and len(llm_query.split()) >= 5:
                s["query"] = llm_query
        out.append(s)
    # Phase 2: generate additional queries for new scenario types
    # (parenting, someone_elses_taste, generic_practical) to reach n~10
    if discovery_llm and pref_block:
        new_types = ["parenting_or_child", "someone_elses_taste",
                     "generic_practical"]
        for stype in new_types:
            if stype not in _SCENARIO_DESCRIPTIONS:
                continue
            llm_query = _llm_generate_scenario_query(
                stype, pref_block, discovery_llm,
            )
            if llm_query and len(llm_query.split()) >= 5:
                t_probe = since_timestamp - rng.randint(1000, 86400)
                out.append({
                    "scenario_id": f"llm_{stype}_{user_id}",
                    "name": stype,
                    "query": llm_query,
                    "notes": _SCENARIO_DESCRIPTIONS[stype],
                    "forbidden_items": out[0]["forbidden_items"] if out else [],
                    "carve_out": None,
                    "t_probe": t_probe,
                    "t_test": t_probe,
                })
    # Phase 3: generate a second query for some existing types
    # to push toward n=10 total
    if discovery_llm and pref_block and len(out) < 10:
        for s in list(out[:3]):
            if len(out) >= 10:
                break
            llm_query = _llm_generate_scenario_query(
                s["name"], pref_block, discovery_llm,
            )
            if llm_query and len(llm_query.split()) >= 5:
                t_probe = s["t_probe"] - rng.randint(500, 3600)
                out.append({
                    "scenario_id": f"{s['scenario_id']}_b",
                    "name": s["name"],
                    "query": llm_query,
                    "notes": s["notes"],
                    "forbidden_items": s["forbidden_items"],
                    "carve_out": s.get("carve_out"),
                    "t_probe": t_probe,
                    "t_test": t_probe,
                })
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
