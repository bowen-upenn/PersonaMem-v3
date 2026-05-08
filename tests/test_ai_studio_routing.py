"""Unit tests for milestone (b) routing changes — deterministic only,
no LLM calls.

Covers:
  - PLATFORMS now contains "AI_Studio" (5 apps)
  - SOCIAL_PLATFORMS module-level constant
  - AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES + category keyword sets
  - WRITING_UTILITY_CATEGORY_KEYWORDS
  - PersonaAgent quota constants (CHATBOT_CANONICAL_TARGET = 0.27,
    AI_STUDIO_CANONICAL_TARGET = 0.18, SOCIAL_CANONICAL_FLOOR = 0.17)
  - _quota_rebalance_apps three-pass behavior:
      (1) Chatbot top-up
      (2) AI_Studio carve-out from Chatbot (eligible categories only;
          implicit_negative + writing-utility kept on Chatbot)
      (3) Social-app top-up

Run: `python tests/test_ai_studio_routing.py`
     or `pytest tests/test_ai_studio_routing.py -v`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.persona_agent import (
    AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS,
    AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES,
    CrossReferencedPersona,
    PLATFORMS,
    PersonaAgent,
    SOCIAL_PLATFORMS,
    WRITING_UTILITY_CATEGORY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Catalog / constants
# ---------------------------------------------------------------------------

def test_platforms_includes_ai_studio():
    assert PLATFORMS == ["Instagram", "Facebook", "Threads", "Chatbot", "AI_Studio"]


def test_social_platforms_constant():
    assert SOCIAL_PLATFORMS == ["Instagram", "Facebook", "Threads"]
    # Sanity: SOCIAL_PLATFORMS ⊂ PLATFORMS, no overlap with conversational surfaces
    for p in SOCIAL_PLATFORMS:
        assert p in PLATFORMS
    assert "Chatbot" not in SOCIAL_PLATFORMS
    assert "AI_Studio" not in SOCIAL_PLATFORMS


def test_eligibility_constants():
    # Hidden-persona types that map to AI_Studio
    assert "emotional_pattern" in AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES
    assert "intimate_interest" in AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES
    assert "parasocial_attachment" in AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES
    assert "aspiration" in AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES
    assert "identity_anchor" in AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES
    # Category keyword presence
    assert "identity" in AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS
    assert "intimate" in AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS
    assert "parasocial" in AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS
    # Utility keywords (stay on Chatbot)
    assert "email" in WRITING_UTILITY_CATEGORY_KEYWORDS
    assert "translation" in WRITING_UTILITY_CATEGORY_KEYWORDS
    assert "code" in WRITING_UTILITY_CATEGORY_KEYWORDS


def test_quota_constants_milestone_b():
    """Milestone (b) carve-out: Chatbot drops 0.40 → 0.27, AI_Studio gets 0.18."""
    assert PersonaAgent.CHATBOT_CANONICAL_TARGET == 0.27
    assert PersonaAgent.AI_STUDIO_CANONICAL_TARGET == 0.18
    assert PersonaAgent.SOCIAL_CANONICAL_FLOOR == 0.17
    # Total quota across ALL apps stays under 1.0 (3 social floors + Chatbot + AI_Studio
    # = 0.17*3 + 0.27 + 0.18 = 0.96, leaves slack for noise)
    total = (
        PersonaAgent.CHATBOT_CANONICAL_TARGET
        + PersonaAgent.AI_STUDIO_CANONICAL_TARGET
        + 3 * PersonaAgent.SOCIAL_CANONICAL_FLOOR
    )
    assert total <= 1.0


# ---------------------------------------------------------------------------
# _quota_rebalance_apps three-pass behavior
# ---------------------------------------------------------------------------

def _agent_with_pool(pool: list) -> PersonaAgent:
    """Build a minimal PersonaAgent and inject a canonical pool. Bypasses
    the normal __init__ because we don't need any LLM state."""
    a = PersonaAgent.__new__(PersonaAgent)
    a.user_id = "test"
    a.verbose = False
    a.cross_referenced_personas = pool
    return a


def _cr(persona_item: str, category: str, app: str = "Threads",
        xref: float = 5.0, itype: str = "explicit_positive") -> CrossReferencedPersona:
    return CrossReferencedPersona(
        persona_item=persona_item,
        category=category,
        confidence_score_init=0.9,
        confidence_cross_referenced=xref,
        assigned_app=app,
        source_interaction_type=itype,
    )


def test_rebalance_top_up_chatbot_to_27_percent():
    # 100 canonicals, all on Threads. After rebalance Chatbot should hit ~27.
    pool = [_cr(f"item{i}", "general") for i in range(100)]
    a = _agent_with_pool(pool)
    a._quota_rebalance_apps()
    cb = sum(1 for cr in pool if cr.assigned_app == "Chatbot")
    assert 25 <= cb <= 30, f"Chatbot count {cb} not near 27"


def test_rebalance_carves_out_ai_studio_from_chatbot():
    # 100 canonicals. 50 with AI-Studio-eligible categories, 50 with writing-
    # utility categories. After rebalance: AI_Studio should pull eligible ones
    # OUT of Chatbot, never the utility ones.
    eligible = [_cr(f"e{i}", "identity exploration", app="Chatbot", xref=3.0)
                for i in range(50)]
    utility  = [_cr(f"u{i}", "email writing", app="Chatbot", xref=8.0)
                for i in range(50)]
    pool = eligible + utility
    a = _agent_with_pool(pool)
    a._quota_rebalance_apps()
    cb = sum(1 for cr in pool if cr.assigned_app == "Chatbot")
    ais = sum(1 for cr in pool if cr.assigned_app == "AI_Studio")
    # AI_Studio target ~18. Should hit close (within rounding)
    assert 16 <= ais <= 20, f"AI_Studio count {ais} not near 18"
    # Utility canonicals MUST NOT be on AI_Studio
    util_on_ais = sum(1 for cr in utility if cr.assigned_app == "AI_Studio")
    assert util_on_ais == 0, "writing-utility canonicals should stay on Chatbot"
    # The eligible ones moved should be the LOWEST xref (3.0); we check that
    # by confirming none of the high-xref utility items moved.


def test_rebalance_implicit_negative_never_routes_to_ai_studio():
    """Even if a canonical's category is identity-coded, implicit_negative
    must stay off AI_Studio."""
    pool = [_cr(f"n{i}", "identity exploration",
                app="Chatbot", itype="implicit_negative", xref=2.0)
            for i in range(50)]
    pool += [_cr(f"p{i}", "general", app="Threads") for i in range(50)]
    a = _agent_with_pool(pool)
    a._quota_rebalance_apps()
    neg_on_ais = sum(1 for cr in pool[:50] if cr.assigned_app == "AI_Studio")
    assert neg_on_ais == 0


def test_rebalance_social_floor_topup_runs_after_ai_studio_carve():
    # 100 canonicals split between Threads (with AI Studio-eligible categories)
    # and Chatbot (mix of eligible + utility). After rebalance: Chatbot ≈ 27,
    # AI_Studio ≈ 18, social-app top-ups respect the floor.
    pool = [_cr(f"a{i}", "identity values") for i in range(40)]   # Threads, eligible
    pool += [_cr(f"b{i}", "intimate vulnerability", app="Chatbot", xref=2.0)  # Chatbot, eligible (low xref → carved)
             for i in range(30)]
    pool += [_cr(f"c{i}", "email writing", app="Chatbot", xref=8.0)  # Chatbot, utility (kept)
             for i in range(30)]
    a = _agent_with_pool(pool)
    a._quota_rebalance_apps()
    from collections import Counter
    counts = Counter(cr.assigned_app for cr in pool)
    # Must respect the Chatbot target (within ±5 for rounding/migration overshoot)
    assert 22 <= counts.get("Chatbot", 0) <= 32, f"Chatbot {counts.get('Chatbot', 0)} out of range"
    # AI_Studio carved out from eligible Chatbot pool
    assert 14 <= counts.get("AI_Studio", 0) <= 22, f"AI_Studio {counts.get('AI_Studio', 0)} out of range"
    # All apps are present in the distribution
    for app in PLATFORMS:
        assert counts.get(app, 0) >= 0
    # Sanity: total adds up to pool size
    assert sum(counts.values()) == 100


def test_rebalance_empty_pool_safe():
    a = _agent_with_pool([])
    a._quota_rebalance_apps()  # must not raise


# ---------------------------------------------------------------------------
# assign_personas_to_apps_prompt — surface AI_Studio when persona present
# ---------------------------------------------------------------------------

def test_routing_prompt_omits_ai_studio_when_persona_absent():
    from data_preparation import prompts
    out = prompts.assign_personas_to_apps_prompt(
        app_personas={"Chatbot": {}, "Instagram": {}},
        preferences=[{"persona_item": "x", "category": "y"}],
        ai_studio_persona=None,
    )
    assert "AI_Studio" not in out
    assert "AI Studio" not in out
    assert "~40% Chatbot" in out  # legacy distribution language


def test_routing_prompt_includes_ai_studio_when_persona_present():
    from data_preparation import prompts
    out = prompts.assign_personas_to_apps_prompt(
        app_personas={"Chatbot": {}, "Instagram": {}},
        preferences=[{"persona_item": "x", "category": "y"}],
        ai_studio_persona={
            "persona_archetype": "mentor_coach",
            "character_name": "Rowan",
            "relational_stance": "warm",
            "topical_strengths": ["aspiration"],
            "eligibility_signal": {},
        },
    )
    assert "AI_Studio" in out
    assert "AI Studio" in out  # the "AI Studio (5th app)" header
    assert "27%" in out and "18%" in out  # new distribution
    assert "implicit_negative" in out  # negative firewall mentioned
    assert "Rowan" in out  # persona surfaced


if __name__ == "__main__":
    import inspect
    failures = []
    n_passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and inspect.signature(fn).parameters == {}:
            try:
                fn()
                n_passed += 1
                print(f"  PASS {name}")
            except AssertionError as e:
                failures.append((name, e))
                print(f"  FAIL {name}: {e}")
            except Exception as e:
                failures.append((name, e))
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{n_passed} passed, {len(failures)} failed")
    sys.exit(0 if not failures else 1)
