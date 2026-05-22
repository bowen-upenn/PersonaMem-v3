"""Step 4.6 — hidden_persona_implicit_qa.

Tests whether the agent has *inferred* a hidden persona well enough to
*serve it implicitly* in response to a surface query that does not name
or hint at the hidden persona. Both example and inferior responses are
naturalistic; only the example reflects deeper inference.

All 12 hidden-persona types are eligible (`personality_trait`,
`aspiration`, `emotional_pattern`, `identity_anchor`, `intimate_interest`,
`intellectual_curiosity`, `private_hobby`, `parasocial_attachment`,
`compensatory_need`, `covert_concern`, `medical_aesthetic_concern`,
`sensitive_life_event`), but every instance must be socially appropriate.
For privacy-flagged types (intimate_interest, covert_concern,
compensatory_need, medical_aesthetic_concern, sensitive_life_event), the
discovery prompt is explicitly instructed to craft a surface query +
example that serve the deeper need *without* naming or directly evoking
the sensitive topic.

Eligibility filter for a hidden persona at T_test:
  - evidence_rows ≥ 25
  - evidence_row_fraction ≥ 0.03
  - last_seen_ts within the last 30 days before T_test
Type diversity: ≥ 2 distinct hidden_persona types per user; cap 4
instances per user.

T_test selection: a moment where the persona has recent (≤ 7d) evidence,
but the surface query itself is timeless (e.g., "What should I do this
weekend?" rather than "Should I do X again?").

Instance shape uses the canonical 5-field layout (`user_query`,
`example_response`, `inferior_response`, `groundtruth_preference`,
`rubric_dimensions`). Hidden-persona metadata lives inside
`groundtruth_preference`.

Hard constraints enforced by the build-time foil check:
- example_response MUST NOT contain verbatim the persona `label`, the
  persona `type` string, or n-grams from the persona `description`.
- inferior_response MUST NOT serve the deeper motivation (judged by LLM).
- Both responses MUST be naturalistic / conversationally plausible.
- For privacy-flagged personas: example MUST NOT touch the sensitive
  topic directly; it must reach the deeper need via adjacent content.

This file ships with discovery scaffolding + a stub runner. Bringing it
to full life requires either:
  - wiring an LLM client through `build_hidden_persona_implicit_qa`
    (see the `discovery_llm` parameter), OR
  - generating user_query + example/inferior pairs offline.
"""

from __future__ import annotations

import datetime as dt
import random

from evaluation.backend_query import BackendQuery


DAY_SECONDS = 24 * 60 * 60

# Eligibility floors at T_test.
MIN_EVIDENCE_ROWS = 25
MIN_EVIDENCE_ROW_FRACTION = 0.03
MAX_DAYS_SINCE_LAST_SEEN = 30
RECENT_EVIDENCE_DAYS = 7

# Hard caps.
INSTANCES_PER_USER_CAP = 4
MIN_DISTINCT_TYPES = 2

# Privacy-flagged hidden persona types — surfacing these implicitly is
# allowed (privacy_leak only fires on direct mention), but the discovery
# prompt must craft a query/response pair that doesn't touch the
# sensitive topic head-on.
PRIVACY_FLAGGED_TYPES = {
    "intimate_interest",
    "covert_concern",
    "compensatory_need",
    "medical_aesthetic_concern",
    "sensitive_life_event",
}


def _filter_eligible_personas(
    profile: dict, t_test: int,
) -> list[dict]:
    """Return hidden_personas that meet the eligibility floors at t_test."""
    out: list[dict] = []
    for hp in profile.get("hidden_personas") or []:
        if not isinstance(hp, dict):
            continue
        if hp.get("evidence_rows", 0) < MIN_EVIDENCE_ROWS:
            continue
        if hp.get("evidence_row_fraction", 0.0) < MIN_EVIDENCE_ROW_FRACTION:
            continue
        last_seen = hp.get("last_seen_ts") or 0
        if not last_seen:
            continue
        if (t_test - last_seen) > MAX_DAYS_SINCE_LAST_SEEN * DAY_SECONDS:
            continue
        out.append(hp)
    return out


def _hp_is_privacy_flagged(hp: dict) -> bool:
    return (hp.get("type") or "").lower() in PRIVACY_FLAGGED_TYPES


def _build_instance(
    hp: dict,
    flavor: str,
    user_id: str,
    seq: int,
    t_test: int,
) -> dict:
    """Emit one instance in the canonical 5-field shape.

    `user_query`, `example_response`, `inferior_response` are placeholders
    here — production builds should fill them via an LLM discovery call.
    """
    is_pf = _hp_is_privacy_flagged(hp)
    return {
        "instance_id": f"hp_implicit_{user_id}_{seq:03d}_{flavor}",
        "task_type": "hidden_persona_implicit_qa",
        "task_id": "hidden_persona_implicit_qa",
        "flavor": flavor,
        "entry_point": "chatbot_routed" if flavor == "chatbot" else "app_native",
        "t_test": t_test,
        "user_query": "",  # to be filled by discovery LLM
        "example_response": "",  # to be filled by discovery LLM
        "inferior_response": "",  # to be filled by discovery LLM
        "groundtruth_preference": {
            "hidden_persona": {
                "label": hp.get("label", ""),
                "type": hp.get("type", ""),
                "is_privacy_flagged": is_pf,
                "description": hp.get("description", ""),
                # Top hashtags surface here for the judge prompt only;
                # they're stripped from the agent's snapshot via the
                # standard hidden_persona_labels firewall.
                "evidence_hashtags_sample": (hp.get("evidence_hashtags") or [])[:6],
            },
            # The discovery LLM should fill these by characterising why
            # the example response reflects deeper inference vs. why the
            # inferior takes the surface query at face value.
            "implicit_signal": "",
            "surface_only_signal": "",
        },
        # rubric_tags set at CSV emission from TASK_TYPE_META.
    }


def _t_test_anchor(profile: dict, t_now: int) -> int:
    """Pick a T_test ~7 days before t_now so 'recent' evidence is fresh
    but the surface query is timeless. Falls back to t_now if there's
    not enough headroom.
    """
    candidate = t_now - RECENT_EVIDENCE_DAYS * DAY_SECONDS
    return max(candidate, 0) or t_now


def build_hidden_persona_implicit_qa(
    bq: BackendQuery,
    user_id: str,
    t_now: int,
    discovery_llm=None,
    rng_seed: int = 0,
) -> list[dict]:
    """Build hidden_persona_implicit_qa instances for one user.

    Scaffolding only: emits instances with empty user_query / example /
    inferior fields. A discovery LLM call (TODO) should populate those.
    The audit step drops empty-query rows so this is safe to ship.
    """
    profile = bq.get_full_profile(user_id) or {}
    if not profile:
        return []
    t_test = _t_test_anchor(profile, t_now)

    eligible = _filter_eligible_personas(profile, t_test)
    if not eligible:
        return []

    rng = random.Random(rng_seed)
    rng.shuffle(eligible)

    # Type diversity: try to pick across distinct types.
    seen_types: set[str] = set()
    picked: list[dict] = []
    for hp in eligible:
        ptype = hp.get("type", "")
        if ptype in seen_types and len(picked) >= 2:
            continue
        picked.append(hp)
        seen_types.add(ptype)
        if len(picked) >= INSTANCES_PER_USER_CAP:
            break

    # If diversity gate not met, still emit (advisory).
    if len(seen_types) < MIN_DISTINCT_TYPES:
        pass

    out: list[dict] = []
    flavor_cycle = iter(["chatbot", "recsys", "chatbot", "recsys"])
    for i, hp in enumerate(picked):
        flavor = next(flavor_cycle, "chatbot")
        inst = _build_instance(hp, flavor, user_id, i + 1, t_test)
        inst["_scaffolding_stub"] = True
        out.append(inst)

    if discovery_llm is None:
        print(f"[hidden_persona_implicit_qa] user {user_id}: "
              f"emitted {len(out)} scaffolded instance(s) — "
              f"discovery_llm not wired, user_query/example/inferior empty.")
    return out


# ---------------------------------------------------------------------------
# Runner — stub. Dispatches to chatbot_response.run_task_b once instances
# carry non-empty user_query + example_response + inferior_response.
# ---------------------------------------------------------------------------


def run_hidden_persona_implicit_qa(
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
    """Stub runner. Returns one dry_run-style result per scaffolded instance
    so the pipeline doesn't crash. Replace with a chatbot_response dispatch
    once the discovery LLM fills user_query / example / inferior.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        results.append({
            "task": "hidden_persona_implicit_qa",
            "user_id": user_id,
            "instance_id": inst.get("instance_id", ""),
            "flavor": inst.get("flavor", ""),
            "mode": mode,
            "metrics": {},
            "status": "scaffolding_stub" if inst.get("_scaffolding_stub") else "ok",
        })
    return results
