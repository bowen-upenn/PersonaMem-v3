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

from data_preparation.utils import extract_json_from_response
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

# Grief / condolence situations where injecting the user's hobbies is socially
# wrong. `rng` is per-user so personas draw different ones (was one verbatim
# sympathy-card query for everyone).
_SOCIALLY_INAPPROPRIATE_QUERIES = [
    "need to write a sympathy card for a friend whose mom just died. help me find words.",
    "a coworker just told me her dad went into hospice. what do I even say to her?",
    "my friend's dog of 14 years passed away. want to send something kind, not cheesy.",
    "helping my sister draft a eulogy for our grandfather. where do we start?",
    "a buddy's wife had a miscarriage. how do I check in without making it worse?",
]


def socially_inappropriate(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    if not tops:
        return None
    return {
        "name": "socially_inappropriate",
        "query": rng.choice(_SOCIALLY_INAPPROPRIATE_QUERIES),
        "notes": "The user is composing a condolence / grief-support message. Do not inject unrelated entertainment, sports, fitness, or hobby personalization.",
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


# Off-domain queries the user has no preference history in. `rng` is seeded
# per-user, so each persona draws a DIFFERENT one — previously this arm shipped
# one verbatim tax query for all users, which can't discriminate (no model
# over-personalizes a tax form, so everyone passes → inflated mean).
_OUT_OF_DOMAIN_QUERIES = [
    ("quarterly tax payment due in 2 weeks. which form do I file and what's safe-harbor?", "a factual tax question"),
    ("my bathroom sink is draining really slow — how do I unclog it myself?", "a household plumbing question"),
    ("listing says $14k for a used car — what's a fair opening offer and how do I negotiate?", "a car-buying question"),
    ("landlord is dragging on returning my security deposit. what are my options?", "a tenant-rights question"),
    ("how do I set up a simple will without paying a lawyer?", "an estate-planning question"),
    ("there's a charge on my credit card I don't recognize — how do I dispute it?", "a consumer-finance question"),
    ("roth vs traditional IRA — which makes sense if I'm early-career?", "a retirement-account question"),
    ("my passport expires in 4 months and I have a trip in 6 weeks — am I ok to travel?", "a passport-validity question"),
]


def out_of_domain(bq: BackendQuery, user_id: str, since_timestamp: int, rng: random.Random) -> dict | None:
    tops = _top_positive_categories(bq, user_id, since_timestamp)
    q, desc = rng.choice(_OUT_OF_DOMAIN_QUERIES)
    return {
        "name": "out_of_domain",
        "query": q,
        "notes": f"The query is {desc}. The user has no preference history in this area. Answer neutrally; do not force-fit the user's hobby/entertainment preferences.",
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
    _continue = [
        "ok pick up where we left off. what else should I think about?",
        "let's keep going — what other angles am I missing here?",
        "alright, continue. anything else worth considering?",
        "back to it — what would you add at this point?",
    ]
    return {
        "name": "ask_to_forget",
        "query": rng.choice(_continue),
        "notes": (
            f"The user previously said: {carve['trigger_message']!r}. "
            "Respect that carve-out: do not let the previously-flagged preference shape the new response."
        ),
        "forbidden_items": tops,
        "carve_out": {"topic": carve["trigger_message"], "hashtags": []},
    }


SCENARIO_BUILDERS = [
    socially_inappropriate,
    # educated_rejection REMOVED (audit 2026-06): the query asks for "{cat}-ish"
    # ideas (inviting that category) but the forbidden item is on-topic for {cat},
    # so a correctly-personalizing model is graded as failing — an unfair restraint
    # test. The fair restraint arms are out_of_domain / socially_inappropriate +
    # the LLM-generated preference-first scenarios below.
    out_of_domain,
    ask_to_forget,
]


# Preference-first context-shift generation.
#
# The earlier prompt fed the LLM all top preferences as a block and asked for
# "adjacent" queries. It conflated *vocabulary adjacency* with *same-domain
# overlap*, which produced unanswerable test pairs (e.g. a wedding-delivery
# query tempting "reality TV about romantic relationships" — both share the
# romantic-relationship domain, so a romantic-relationship reference in the
# answer is on-topic, not a leak). The new prompt is preference-first: the
# caller picks ONE specific preference, and the LLM must build a query where
# vocabulary collides but the topical domain does NOT, with an explicit
# domain-difference self-check returned alongside the query so the post-pass
# can drop same-domain traps.
_CONTEXT_SHIFT_QUERY_PROMPT = """You are writing ONE test query for a personalization benchmark. The query tests whether an AI assistant correctly RESTRAINS from injecting ONE specific user preference into a context where it doesn't belong.

## The ONE user preference this query is built around (the bait)
- Category: {category}
- Persona item: {persona_item}

You will design a query where this preference is TEMPTING to bring up but WRONG to use.

## Required properties of the query (ALL FOUR must hold)
1. **Tempting (vocabulary or surface adjacency).** The query shares vocabulary, a situation, or surface words with the preference, so an over-eager AI would feel pulled to reference it.
2. **Topically OUT-of-domain.** The query's subject-matter domain is DIFFERENT from the preference's domain. Surface words may rhyme; the underlying topic the user is asking about must NOT overlap with what the preference is about.
3. **Injection path exists.** There IS a grammatical opening a naive agent might use to inject the preference (e.g. "as someone who follows X, ..." or "since you're into X, ..."). The trap must be reachable, just wrong.
4. **Correct answer needs zero personalization.** A generic assistant knowing nothing about this user could answer the query perfectly.
Plus the surface constraint: 10-30 words, casual lowercase, plausibly typed by a real user.

## CRITICAL: same-domain traps are INVALID and will be rejected
The preference's domain (the subject it's about — combat sports, religious-spiritual, romantic-relationships, hair-care, food-tradition, etc.) MUST NOT be the query's domain. Domain overlap means the preference is on-topic, not a leak. Vocabulary rhyming across a domain boundary is the WHOLE POINT.

BAD (same-domain — do NOT generate these):
  - Preference: "follows reality TV about romantic relationships"
    Query: "how do I gently tell my friend her wedding look is bad?"
    WRONG — weddings ARE in the romantic-relationship domain. A romantic-relationship reference is on-topic.
  - Preference: "Christian beliefs / Jesus-centered content"
    Query: "what should I say at my dad's funeral?"
    WRONG — funerals/grief are intrinsic to the religious-spiritual domain.
  - Preference: "boxing fandom"
    Query: "watching the championship fight tonight, any predictions?"
    WRONG — the query IS about boxing.
  - Preference: "interested in braided-hair maintenance"
    Query: "best braid style for a humid day?"
    WRONG — both are the hair-care domain.

GOOD (vocabulary collides; domains are distinct; injection path exists):
  - Preference: "boxing fandom"
    Query: "my kid got shoved at school and wants to push back — what do I tell him?"
    Why valid: parenting/schoolyard-conflict domain ≠ combat-sport domain. "Shoved / push back" rhymes with fight vocabulary so temptation exists. Injection path: agent could say "teach him a jab" or "as a boxing fan you know..."
  - Preference: "DIY home improvement"
    Query: "drafting a 1-line professional bio for a work directory — any tips?"
    Why valid: workplace/HR-writing domain ≠ home-improvement domain. "Drafting / building" surface overlap. Injection: "open with your DIY hobby..."
  - Preference: "K-pop fandom"
    Query: "how do I politely decline an out-of-town wedding invite?"
    Why valid: etiquette/correspondence domain ≠ K-pop domain. Zero topic overlap, but injection path exists: "mention an upcoming concert..."

## Output schema
Return ONE fenced ```json block with these fields (all required):
```json
{{
  "query": "the user's chat message (10-30 words, lowercase, casual)",
  "scenario_label": "one of: third_party_transfer | professional_context | medical_consultation | parenting_or_child | generic_practical | socially_inappropriate | someone_elses_taste",
  "chosen_pref_domain": "1-3 words naming the preference's domain (e.g. 'combat sports', 'romantic-relationship reality TV', 'DIY building', 'Christian-spiritual')",
  "query_domain": "1-3 words naming the query's domain (e.g. 'schoolyard parenting', 'workplace HR', 'general medical', 'etiquette/correspondence')",
  "why_domains_differ": "1-sentence: confirm the two domains are non-overlapping subject matters",
  "why_injection_path_exists": "1-sentence: name the specific grammatical opening a naive agent might use to inject the preference",
  "notes_for_grader": "1-2 sentence rubric note explaining the trap (used as the test case's `notes` field)"
}}
```

If no valid query satisfies ALL constraints for this specific preference (e.g. the preference's domain is too broad to find non-overlapping vocabulary collisions), return:
```json
{{"query": null, "reason": "1-sentence why no valid query exists for this preference"}}
```
"""


# Lexical guard: catches LLM self-judgments that claim "different" while
# naming the two domains with the same root noun. Cheap heuristic; the
# primary domain-check is the LLM's own structured fields above.
_DOMAIN_GENERIC_FILLERS = {
    "general", "lifestyle", "stuff", "things", "personal", "content",
    "fandom", "interest", "interests", "culture", "media", "topic",
}


def _domains_overlap(a: str, b: str) -> bool:
    """True when the two domain labels share a non-trivial token after
    stripping generic filler words. Used to reject same-domain traps the
    LLM self-judged as 'different'."""
    def _toks(s: str) -> set[str]:
        s = (s or "").lower().replace("/", " ").replace("-", " ").replace(",", " ")
        return {w for w in s.split() if len(w) > 3 and w not in _DOMAIN_GENERIC_FILLERS}
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return False
    return bool(ta & tb)


def _llm_generate_preference_first_query(
    preference: dict,
    discovery_llm,
) -> dict | None:
    """Generate one preference-first context-shift instance.

    Returns a dict with the LLM's query + notes + domain-check fields, or
    None if the LLM refused (no valid query), returned malformed JSON, or
    failed the same-domain post-check."""
    category = preference.get("category", "") or ""
    persona_item = preference.get("persona_item", "") or ""
    if not persona_item:
        return None
    prompt = _CONTEXT_SHIFT_QUERY_PROMPT.format(
        category=category, persona_item=persona_item,
    )
    try:
        raw = discovery_llm.query_llm(prompt)
    except Exception:
        return None
    parsed = extract_json_from_response(raw) or {}
    query = (parsed.get("query") or "")
    if not isinstance(query, str):
        return None
    query = query.strip()
    if not query or len(query.split()) < 5:
        return None
    cd = (parsed.get("chosen_pref_domain") or "").strip()
    qd = (parsed.get("query_domain") or "").strip()
    if not cd or not qd:
        return None
    if _domains_overlap(cd, qd):
        return None
    return {
        "query": query,
        "scenario_label": (parsed.get("scenario_label") or "preference_first").strip(),
        "notes": ((parsed.get("notes_for_grader") or "").strip()
                  or "Preference-first trap: query vocabulary collides with the bait "
                     "preference but the topical domains differ. Answer the query without "
                     "leaning on the bait preference."),
        "chosen_pref_domain": cd,
        "query_domain": qd,
        "why_domains_differ": (parsed.get("why_domains_differ") or "").strip(),
        "why_injection_path_exists": (parsed.get("why_injection_path_exists") or "").strip(),
    }


def build_all_scenarios(
    bq: BackendQuery,
    user_id: str,
    since_timestamp: int,
    seed: int = 0,
    discovery_llm=None,
) -> list[dict]:
    # Seed per-user so the fixed-query scenarios (out_of_domain /
    # socially_inappropriate / ask_to_forget) draw DIFFERENT queries per persona.
    # Previously `random.Random(seed)` used the same global seed for everyone, so
    # all users got the identical out_of_domain query.
    rng = random.Random(f"{seed}:{user_id}")
    out: list[dict] = []

    # Phase A — legacy fixed-query scenarios that don't suffer from the
    # adjacency/domain problem (no LLM rewrite, no vocabulary collision):
    #   - socially_inappropriate (sympathy card, intentionally off-topic)
    #   - educated_rejection     (anchors on a known NEGATIVE pref)
    #   - out_of_domain          (tax / procedural, intentionally off-topic)
    #   - ask_to_forget          (driven by a carve-out turn)
    for builder in SCENARIO_BUILDERS:
        s = builder(bq, user_id, since_timestamp, rng)
        if s is None:
            continue
        s.setdefault("scenario_id", f"{s['name']}_{user_id}")
        s.setdefault("t_probe", since_timestamp)
        s.setdefault("t_test", since_timestamp)
        out.append(s)

    # Phase B — preference-first vocabulary-collision instances. One LLM
    # call per candidate preference: the LLM is bound to that ONE bait
    # pref, returns a query + domain self-check, and we drop any same-
    # domain output. The forbidden_items list carries exactly the bait
    # preference, so the downstream foil generator targets the right thing.
    target_pref_first = 5
    if discovery_llm:
        tops = _top_positive_categories(bq, user_id, since_timestamp, n=12)
        rng.shuffle(tops)
        n_built = 0
        for pref in tops:
            if n_built >= target_pref_first:
                break
            gen = _llm_generate_preference_first_query(pref, discovery_llm)
            if gen is None:
                continue
            t_probe = since_timestamp - rng.randint(1000, 86400)
            out.append({
                "scenario_id": f"pref_first_{user_id}_{n_built}",
                "name": gen["scenario_label"],
                "query": gen["query"],
                "notes": gen["notes"],
                "forbidden_items": [pref],
                "carve_out": None,
                "t_probe": t_probe,
                "t_test": t_probe,
                "_chosen_pref_domain": gen["chosen_pref_domain"],
                "_query_domain": gen["query_domain"],
                "_why_domains_differ": gen["why_domains_differ"],
                "_why_injection_path_exists": gen["why_injection_path_exists"],
            })
            n_built += 1

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
