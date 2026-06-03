"""Task `local_recommendation_geo_shift` — silent geo-shift awareness probe.

When a user has shifted geo at least twice (e.g. home → travel → home, or
home → travel1 → travel2) and asks a city-agnostic local-recommendation
question, the chatbot should ground its recommendations in the user's
*current* city (most recent `event_location.city` in their pre-T_test
history) while still respecting the user's general persona profile —
without the user ever naming the new city.

**Round trip, not a single hop.** Each scenario pairs TWO consecutive
transitions: a recommendation query asked AFTER the shift to the away city
(`leg="after_shift"`, current city = away) and a SIMILAR query (same
category) asked AFTER shifting BACK home / onward to a different city
(`leg="after_return"`, current city = the second hop's destination). Both
legs share a `scenario_id`. The correct answer for each leg reflects the
city current AT THAT leg's timestamp — so the agent must track location on
every turn rather than latching onto the most-recent city once.

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
    """Emit ROUND-TRIP instances for eligible users.

    The probe is a *round trip*, not a single hop: a model that merely
    latches onto the most-recent city once would pass a single post-shift
    query, so each scenario pairs TWO consecutive transitions —

      - leg ``after_shift``: a recommendation query asked just after the
        user shifts to the away city (current city = away), and
      - leg ``after_return``: a SIMILAR query (same category) asked just
        after the user shifts BACK home / onward to a different city
        (current city = the second transition's destination).

    Both legs share one ``scenario_id`` so downstream readers can see they
    are a pair, and each leg's ``t_test`` / ``current_city`` is set to the
    city current AT THAT timestamp — so the grader expects away-city recs
    for leg #1 and home-city recs for leg #2. This tests location tracking
    on each turn.

    Per-(scenario × category) diversity is preserved: each scenario picks
    ``_CATEGORIES_PER_TRANSITION`` categories, and each category yields the
    two-leg pair.
    """
    profile = bq.get_full_profile(user_id) or {}
    events = _load_events_with_location(bq, user_id)
    transitions = _detect_city_transitions(events)
    if not _is_eligible(profile, transitions):
        return []

    instances: list[dict] = []
    # Pair consecutive transitions into round-trip scenarios. Transition i
    # is the "outbound" hop (home -> away); transition i+1 is the "return"
    # hop (away -> home, or away -> a different city). A scenario needs both
    # legs, so we walk consecutive pairs (tr_out, tr_back). The 1-based
    # scenario index is used for human-readable instance ids. Cap at
    # _MAX_TRANSITIONS_PER_USER scenarios so a heavy-traveler doesn't
    # dominate the benchmark.
    #
    # If eligibility was satisfied by a single visible transition + a trip
    # arc (i.e. the observation window started mid-trip), there is no second
    # visible transition to anchor a return leg on, so we fall back to a
    # single-leg scenario for that lone transition — still labeled with a
    # scenario id, just without an after_return leg.
    pairs: list[tuple[CityTransition, CityTransition | None]] = []
    if len(transitions) >= 2:
        for i in range(len(transitions) - 1):
            pairs.append((transitions[i], transitions[i + 1]))
    elif len(transitions) == 1:
        pairs.append((transitions[0], None))
    pairs = pairs[:_MAX_TRANSITIONS_PER_USER]

    def _emit_leg(
        scenario_id: str,
        scenario_idx: int,
        leg: str,
        tr: CityTransition,
        category: str,
        query_text: str,
        paired_with_ts: int | None,
    ) -> dict:
        t_test = tr.first_ts_in_new_city + _DELTA_POST_TRANSITION
        return {
            "instance_id": f"{scenario_id}_{leg}_{category}",
            "task_id": "local_recommendation_geo_shift",
            "task_type": "local_recommendation_geo_shift",
            "t_test": t_test,
            "user_query": query_text,
            "query_text": query_text,
            "app_context": "chatbot",
            "category": category,
            # Round-trip linkage: both legs share scenario_id; `leg` marks
            # which transition this query was asked after.
            "scenario_id": scenario_id,
            "scenario_idx": scenario_idx,
            "leg": leg,
            "paired_t_test": paired_with_ts,
            # Current city = the destination of THIS leg's transition, as of
            # this leg's t_test. The grader expects this city and rejects the
            # prior city (the stale anchor for this turn).
            "current_city": tr.new_city,
            "current_region": tr.new_region,
            "current_country": tr.new_country,
            "prior_city": tr.prior_city,
            "prior_region": tr.prior_region,
            "prior_country": tr.prior_country,
            "transition_first_ts": tr.first_ts_in_new_city,
            # transition_idx retained for backward-compat readers (the audit
            # cross-checks per-(transition × category)); equals scenario_idx.
            "transition_idx": scenario_idx,
            "groundtruth": {
                "expected_city_in_response": tr.new_city,
                "stale_anchor_city": tr.prior_city,
                "must_align_with_persona_profile": True,
            },
        }

    for scenario_idx, (tr_out, tr_back) in enumerate(pairs, start=1):
        scenario_id = f"geo_shift_{user_id}_{scenario_idx}"
        rng_cat = random.Random(f"{rng_seed}:geo_shift_cats:{user_id}:{scenario_idx}")
        chosen_cats = _pick_categories(rng_cat, _CATEGORIES_PER_TRANSITION)
        for category in chosen_cats:
            rng_q = random.Random(f"{rng_seed}:geo_shift_q:{user_id}:{scenario_idx}:{category}")
            query_text = _pick_query(rng_q, category)
            if not query_text:
                continue

            out_t = tr_out.first_ts_in_new_city + _DELTA_POST_TRANSITION
            back_t = (
                tr_back.first_ts_in_new_city + _DELTA_POST_TRANSITION
                if tr_back is not None
                else None
            )

            # Leg 1: after the outbound shift (current city = away).
            instances.append(_emit_leg(
                scenario_id, scenario_idx, "after_shift", tr_out,
                category, query_text, paired_with_ts=back_t,
            ))
            # Leg 2: after shifting back / onward (current city = home or a
            # different city). Same category → "similar" query. Only present
            # when a second transition exists.
            if tr_back is not None:
                instances.append(_emit_leg(
                    scenario_id, scenario_idx, "after_return", tr_back,
                    category, query_text, paired_with_ts=out_t,
                ))
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
                "scenario_id": inst.get("scenario_id"),
                "leg": inst.get("leg"),
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
            "scenario_id": inst.get("scenario_id"),
            "leg": inst.get("leg"),
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
