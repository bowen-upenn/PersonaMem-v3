"""Task `local_recommendation_geo_shift` — silent geo-shift awareness probe.

When a user has shifted geo at least twice (e.g. home → travel → home, or
home → travel1 → travel2) and asks a city-agnostic local-recommendation
question, the chatbot should ground its recommendations in the user's
*current* city (most recent `event_location.city` in their pre-T_test
history) while still respecting the user's general persona profile —
without the user ever naming the new city.

This is **NOT** an over-personalization test. The correct behavior is to
personalize *more* by reading the latest geo signal; the inferior behavior
is to under-personalize by anchoring on stale geo context. Sits in the
Task E family alongside `e5_horizon_lifecycle` (both push the agent to pick
up an out-of-band signal — geo, expiry timestamp, calendar — without being
prompted).

Headline metric: `geo_shift_correctness ∈ {0.0, 0.5, 1.0}` —
  1.0 if the response names the current city/region and does NOT name the prior city,
  0.5 if the response is geo-neutral (neither city named) so the agent at
      least didn't pin to stale context,
  0.0 if the response names the prior city (stale anchor — hard fail).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from data_preparation.utils import extract_json_from_response
from evaluation import metrics, prompts
from evaluation import personalization_rubric as pr
from evaluation.backend_query import BackendQuery


_HOUR = 3600
_DAY = 24 * _HOUR

# Δ post-transition: how long after the user's first event in the new city
# we place T_test. 6h gives the agent at least one cluster of in-new-city
# events to read without waiting so long that the user's session pattern
# turns into "they live here now". Multiple Δ values would multiply
# instance count without adding signal — pick one good anchor.
_DELTA_POST_TRANSITION = 6 * _HOUR

# Cap eligible transitions per user.
_MAX_TRANSITIONS_PER_USER = 3

# Categories per transition. The full bank is 9; pick 3 deterministically
# per (user, transition) to keep the per-user instance count modest.
_CATEGORIES_PER_TRANSITION = 3

CATEGORIES: list[str] = [
    "restaurant",
    "coffee",
    "activity",
    "sports",
    "entertainment",
    "bar",
    "market",
    "coworking",
    "gas",
]

# City-agnostic query templates. Critical invariant: NO template names a
# specific city/country, AND no template signals "I just arrived" / "in the
# new city" / "since I'm here" — the whole point is the agent has to infer
# the geo shift from the user's history, not from the query text. Phrases
# like "tonight" / "this weekend" / "around here" / "right now" are fine
# because they don't reveal *which* place.
_QUERY_BANK: dict[str, list[str]] = {
    "restaurant": [
        "where should I grab dinner tonight?",
        "any good restaurants you can recommend?",
        "looking for a solid place to eat — what would you suggest?",
    ],
    "coffee": [
        "looking for a solid coffee spot — got any rec?",
        "where's a good cafe to set up with a laptop?",
        "any coffee shop recs for the morning?",
    ],
    "activity": [
        "what should I do this weekend?",
        "any cool activities you'd suggest?",
        "thinking of something fun for saturday — ideas?",
    ],
    "sports": [
        "where can I find a gym or a spot to run?",
        "any rec for a place to play pickup or hit a class?",
        "looking for a workout option — what's around?",
    ],
    "entertainment": [
        "anywhere fun to spend friday night?",
        "what's good for a night out?",
        "movie / show / something to do tonight — got ideas?",
    ],
    "bar": [
        "any good bars to grab drinks?",
        "where would you send me for a chill drink?",
        "wanna grab a drink — recommend somewhere?",
    ],
    "market": [
        "where can I pick up groceries?",
        "looking for a good market — any rec?",
        "need to do a grocery run, where would you suggest?",
    ],
    "coworking": [
        "looking for a coworking spot to take a call from",
        "where can I find a quiet place to work for a few hours?",
        "need a workspace for the afternoon — any rec?",
    ],
    "gas": [
        "need to fill up — anywhere reliable around?",
        "where's a decent gas station to swing by?",
        "looking for a gas stop, what would you recommend?",
    ],
}


# ---------------------------------------------------------------------------
# Eligibility + transition detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CityTransition:
    prior_city: str
    prior_region: str
    prior_country: str
    new_city: str
    new_region: str
    new_country: str
    first_ts_in_new_city: int


def _norm(s) -> str:
    return (s or "").strip()


def _load_events_with_location(bq: BackendQuery, user_id: str) -> list[dict]:
    """Concat all 4 apps' events with non-empty event_location, sorted by ts.

    Mirrors `e5_horizon_lifecycle._load_events` patterns and uses the same
    private accessor (`bq._load_events`) to read events with their
    `event_location` block intact (the public `get_events` snapshot path
    strips fields, but the unstripped path is the right one for build-time
    decisions that never reach the agent).
    """
    out: list[dict] = []
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(user_id, app):
            loc = e.get("event_location") or {}
            if not isinstance(loc, dict):
                continue
            city = _norm(loc.get("city"))
            if not city:
                continue
            ts = e.get("source_timestamp") or 0
            if not ts:
                continue
            out.append(e)
    out.sort(key=lambda e: e.get("source_timestamp") or 0)
    return out


def _detect_city_transitions(events: list[dict]) -> list[CityTransition]:
    """Walk events sorted ascending; emit transitions when city changes.

    A transition's `first_ts_in_new_city` is the timestamp of the first
    event whose city differs from the running city. The very first event
    is not a transition — there's no "prior" city.
    """
    transitions: list[CityTransition] = []
    running_city = None
    running_region = ""
    running_country = ""
    for e in events:
        loc = e.get("event_location") or {}
        city = _norm(loc.get("city"))
        if not city:
            continue
        if running_city is None:
            running_city = city
            running_region = _norm(loc.get("region"))
            running_country = _norm(loc.get("country"))
            continue
        if city != running_city:
            transitions.append(CityTransition(
                prior_city=running_city,
                prior_region=running_region,
                prior_country=running_country,
                new_city=city,
                new_region=_norm(loc.get("region")),
                new_country=_norm(loc.get("country")),
                first_ts_in_new_city=int(e.get("source_timestamp") or 0),
            ))
            running_city = city
            running_region = _norm(loc.get("region"))
            running_country = _norm(loc.get("country"))
    return transitions


def _is_eligible(profile: dict, transitions: list[CityTransition]) -> bool:
    """Eligible iff non-homebody AND the user's data shows a multi-shift
    pattern.

    "Multi-shift" is satisfied by either:
      - >= 2 visible transitions in the event stream (clear shift + return
        or shift + onward leg), OR
      - >= 1 visible transition AND >= 1 entry in `geo_trip_arcs` (the
        trip arc represents an additional implicit transition that the
        observation window may have started in the middle of — common for
        users who were already mid-trip on day 1).

    A single visible transition with NO trip arc is treated as a permanent
    relocation and excluded — it doesn't fit the user's "shifts again/back"
    pattern.
    """
    mobility = (profile.get("mobility_class") or "").lower()
    if mobility == "homebody":
        return False
    n_transitions = len(transitions)
    if n_transitions >= 2:
        return True
    n_arcs = len(profile.get("geo_trip_arcs") or [])
    return n_transitions >= 1 and n_arcs >= 1


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _pick_query(rng: random.Random, category: str) -> str:
    bank = _QUERY_BANK.get(category) or []
    if not bank:
        return ""
    return rng.choice(bank)


def _pick_categories(rng: random.Random, n: int) -> list[str]:
    pool = list(CATEGORIES)
    rng.shuffle(pool)
    return pool[:n]


def build_local_recommendation_geo_shift(
    bq: BackendQuery,
    user_id: str,
    rng_seed: int = 0,
) -> list[dict]:
    """Emit per-(transition, category) instances for eligible users."""
    profile = bq.get_full_profile(user_id) or {}
    events = _load_events_with_location(bq, user_id)
    transitions = _detect_city_transitions(events)
    if not _is_eligible(profile, transitions):
        return []

    instances: list[dict] = []
    # Use 1-based transition index for human readability in instance ids.
    # Every visible transition is "the user is now somewhere different"
    # for the purposes of this probe — eligibility (above) already
    # confirms the user has multi-shift evidence (either >= 2 visible
    # transitions, or 1 visible + a trip arc that implies the missing
    # leg). Cap at _MAX_TRANSITIONS_PER_USER so a heavy-traveler doesn't
    # dominate the benchmark.
    eligible = list(enumerate(transitions, start=1))[:_MAX_TRANSITIONS_PER_USER]

    for transition_idx, tr in eligible:
        t_test = tr.first_ts_in_new_city + _DELTA_POST_TRANSITION
        rng_cat = random.Random(f"{rng_seed}:geo_shift_cats:{user_id}:{transition_idx}")
        chosen_cats = _pick_categories(rng_cat, _CATEGORIES_PER_TRANSITION)
        for category in chosen_cats:
            rng_q = random.Random(f"{rng_seed}:geo_shift_q:{user_id}:{transition_idx}:{category}")
            query_text = _pick_query(rng_q, category)
            if not query_text:
                continue
            instance_id = f"geo_shift_{user_id}_{transition_idx}_{category}"
            instances.append({
                "instance_id": instance_id,
                "task_id": "local_recommendation_geo_shift",
                "task_type": "local_recommendation_geo_shift",
                "t_test": t_test,
                "user_query": query_text,
                "query_text": query_text,
                "app_context": "chatbot",
                "category": category,
                "current_city": tr.new_city,
                "current_region": tr.new_region,
                "current_country": tr.new_country,
                "prior_city": tr.prior_city,
                "prior_region": tr.prior_region,
                "prior_country": tr.prior_country,
                "transition_first_ts": tr.first_ts_in_new_city,
                "transition_idx": transition_idx,
                "groundtruth": {
                    "expected_city_in_response": tr.new_city,
                    "stale_anchor_city": tr.prior_city,
                    "must_align_with_persona_profile": True,
                },
            })
    return instances


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _word_boundary_match(text: str, term: str) -> bool:
    """Case-insensitive whole-word match on a multi-word term.

    Empty terms never match. Special-regex chars in the term are escaped so
    a city like "St. John's" can't blow up the matcher.
    """
    term = (term or "").strip()
    if not term or not text:
        return False
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return bool(pattern.search(text))


def _any_match(text: str, *terms: str) -> bool:
    return any(_word_boundary_match(text, t) for t in terms)


def compute_geo_shift_metrics(
    response_text: str,
    current_city: str,
    prior_city: str,
    current_region: str = "",
) -> dict:
    """Score one geo-shift response.

    - `current_city_grounded`: response names current city or its region.
    - `stale_geo_anchor`: response names prior city — hard fail.
    - `geo_neutral_response`: neither named — partial pass.
    - `geo_shift_correctness ∈ {0.0, 0.5, 1.0}` — composite headline.
    """
    grounded = _any_match(response_text, current_city, current_region)
    stale = _word_boundary_match(response_text, prior_city)
    neutral = (not grounded) and (not stale)

    if stale:
        correctness = 0.0
    elif grounded:
        correctness = 1.0
    elif neutral:
        correctness = 0.5
    else:
        # Defensive — shouldn't happen given the branches above.
        correctness = 0.0

    return {
        "current_city_grounded": int(grounded),
        "stale_geo_anchor": int(stale),
        "geo_neutral_response": int(neutral),
        "geo_shift_correctness": correctness,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def geo_shift_prompt(user_query: str, history_block: str | None = None) -> str:
    """Use the standard chatbot prompt — no special framing.

    The user's instructions are explicit: the user query MUST NOT signal
    that geo has shifted, and the agent has to infer it from history. So
    the prompt mirrors the regular `chatbot_response_prompt` (no mention
    of geo, no hint that this is a geo-shift probe).
    """
    return prompts.chatbot_response_prompt(user_query, prior_conversation=[], history_block=history_block)


def run_local_recommendation_geo_shift(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    from evaluation.inference_utils import dispatch_agent_run, merge_token_metrics

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    for inst in instances:
        t = int(inst.get("t_test") or 0)
        user_query = inst.get("user_query") or inst.get("query_text") or ""
        current_city = inst.get("current_city") or ""
        current_region = inst.get("current_region") or ""
        prior_city = inst.get("prior_city") or ""

        history_block = None
        history_tokens = 0
        if mode in ("llm_longctx", "llm_memory", "mem0"):
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget,
            )
            history_tokens = stats["total_tokens"]

        prompt = geo_shift_prompt(user_query, history_block)

        if dry_run:
            results.append({
                "task": "local_recommendation_geo_shift",
                "user_id": user_id,
                "instance_id": inst.get("instance_id"),
                "category": inst.get("category"),
                "transition_idx": inst.get("transition_idx"),
                "current_city": current_city,
                "prior_city": prior_city,
                "t_test": t,
                "mode": mode,
                "history_tokens": history_tokens,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response) or {}
        response_text = parsed.get("response") or raw_response or ""

        scored = compute_geo_shift_metrics(
            response_text, current_city, prior_city, current_region=current_region,
        )

        # Universal personalization rubric — preference_alignment + hard
        # rule dims. The user's request explicitly asks that recommendations
        # still align with the general persona profile, so we plug into the
        # rubric to get that scored automatically.
        ground_truth = pr.build_source_a(
            bq, user_id, t,
            query_text=user_query,
            query_hashtags=[],
        )
        pers_rubric = pr.score(
            task_id="local_recommendation_geo_shift",
            agent_output=response_text,
            ground_truth=ground_truth,
            source_b=None,
            judge_client=(judge_client if enable_llm_judge else None),
        )

        result_metrics = {
            **scored,
            **{f"pr_{k}": v for k, v in pers_rubric.items() if isinstance(v, (int, float, str))},
        }
        merge_token_metrics(
            result_metrics, prompt=prompt, response=raw_response or "",
            stats=subagent_stats, model=model_name,
        )
        results.append({
            "task": "local_recommendation_geo_shift",
            "user_id": user_id,
            "instance_id": inst.get("instance_id"),
            "category": inst.get("category"),
            "transition_idx": inst.get("transition_idx"),
            "current_city": current_city,
            "prior_city": prior_city,
            "t_test": t,
            "mode": mode,
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "agent_response": response_text,
            "metrics": result_metrics,
        })
    return results
