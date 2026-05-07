"""Unit tests for the AI Studio persona scaffold (milestone (a)):
  - `AIStudioPersona` + `RomanticSpecifier` dataclasses
  - `AI_STUDIO_ARCHETYPES` catalog (10 archetypes)
  - Closed sub-typing vocabularies (gender / sexuality / aesthetic / body /
    relational / explicitness)
  - `ROGERS_CLICHE_BLOCKLIST`
  - `personalize_ai_studio_persona_prompt` builder
  - `PersonaAgent.generate_ai_studio_persona` validation + auto-disable
    behaviour, exercised with a stubbed LLM client (no network calls)

These tests do NOT make any LLM API calls. They use a stub PersonaAgent
that returns canned LLM responses, so the full validation/sanitization
pipeline can be exercised in isolation.

Run: `pytest tests/test_ai_studio_persona.py -v`
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation import prompts
from data_preparation.persona_agent import (
    AIStudioPersona,
    AI_STUDIO_ARCHETYPES,
    InteractionRow,
    PersonaAgent,
    RomanticSpecifier,
    ROGERS_CLICHE_BLOCKLIST,
    ROMANTIC_AESTHETIC_VIBES,
    ROMANTIC_BODY_ROLE_CODINGS,
    ROMANTIC_EXPLICITNESS_BANDS,
    ROMANTIC_GENDER_PRESENTATIONS,
    ROMANTIC_RELATIONAL_DYNAMICS,
    ROMANTIC_SEXUALITY_ORIENTATIONS,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Catalog + vocabulary tests
# ---------------------------------------------------------------------------

def test_archetype_catalog_has_ten_entries():
    assert len(AI_STUDIO_ARCHETYPES) == 10
    expected = {
        "anime_or_fandom_character",
        "late_night_best_friend",
        "romantic_partner",
        "older_sibling_figure",
        "therapist_companion_reflective",
        "mentor_coach",
        "wise_elder_grandparent",
        "niche_expert_creator_ai",
        "hype_affirmation_friend",
        "historical_or_philosophical_voice",
    }
    assert set(AI_STUDIO_ARCHETYPES) == expected


def test_archetype_catalog_required_fields():
    """Every archetype carries voice_template, allowed_topical_depths,
    forbidden_phrases, auto_disable_on_high_acuity_sensitive_event,
    requires_niche_specifier, requires_romantic_specifier, inspiration."""
    required = {
        "voice_template",
        "allowed_topical_depths",
        "forbidden_phrases",
        "auto_disable_on_high_acuity_sensitive_event",
        "requires_niche_specifier",
        "requires_romantic_specifier",
        "inspiration",
    }
    for name, meta in AI_STUDIO_ARCHETYPES.items():
        missing = required - set(meta.keys())
        assert not missing, f"{name} missing fields: {missing}"
        assert isinstance(meta["allowed_topical_depths"], (set, frozenset))
        assert meta["allowed_topical_depths"] <= {"S1", "S2", "S3", "S4"}


def test_only_romantic_partner_auto_disables_on_high_acuity_sle():
    """Generation guard: only `romantic_partner` flips off on high-acuity
    active sensitive_life_event."""
    auto = {
        k for k, v in AI_STUDIO_ARCHETYPES.items()
        if v.get("auto_disable_on_high_acuity_sensitive_event")
    }
    assert auto == {"romantic_partner"}


def test_only_niche_expert_requires_niche_specifier():
    req = {
        k for k, v in AI_STUDIO_ARCHETYPES.items()
        if v.get("requires_niche_specifier")
    }
    assert req == {"niche_expert_creator_ai"}


def test_only_romantic_partner_requires_romantic_specifier():
    req = {
        k for k, v in AI_STUDIO_ARCHETYPES.items()
        if v.get("requires_romantic_specifier")
    }
    assert req == {"romantic_partner"}


def test_rogers_cliche_blocklist_covers_canonical_phrases():
    """Smoke check that the canonical fake-therapist clichés are in the
    blocklist."""
    canon = [
        "I hear you",
        "You're not alone",
        "It's okay to feel that way",
        "Thank you for sharing that",
    ]
    bl = " | ".join(ROGERS_CLICHE_BLOCKLIST).lower()
    for phrase in canon:
        assert phrase.lower() in bl, f"{phrase!r} missing from baseline"


def test_aesthetic_vibe_vocab_includes_user_requested():
    """User explicitly asked for hot_nerd, dark_academia, e_girl, e_boy,
    cottagecore, etc."""
    for v in ("hot_nerd", "dark_academia", "e_girl", "e_boy",
              "goth", "cottagecore", "y2k"):
        assert v in ROMANTIC_AESTHETIC_VIBES, f"{v!r} missing"


def test_body_role_coding_vocab_includes_femboy_and_subcultural_terms():
    for v in ("femboy", "twink", "butch", "femme", "bear", "otter", "bara"):
        assert v in ROMANTIC_BODY_ROLE_CODINGS, f"{v!r} missing"


def test_relational_dynamic_vocab_includes_user_requested():
    """User explicitly asked for elder sis (romantic), mommy, pet, bottom."""
    for v in ("elder_sis_romantic", "mommy", "pet", "bottom",
              "dom_gentle", "sub_eager", "switch", "vers"):
        assert v in ROMANTIC_RELATIONAL_DYNAMICS, f"{v!r} missing"


def test_explicitness_band_vocab():
    assert ROMANTIC_EXPLICITNESS_BANDS == frozenset(
        {"soft_affection", "sensual", "erotic_explicit"}
    )


def test_gender_and_sexuality_vocab_inclusive():
    for v in ("male", "female", "nonbinary", "trans_fem", "trans_masc",
              "genderfluid", "agender"):
        assert v in ROMANTIC_GENDER_PRESENTATIONS
    for v in ("straight", "gay_mm", "lesbian_ff", "bi", "pan",
              "ace_romantic", "queer_unspecified"):
        assert v in ROMANTIC_SEXUALITY_ORIENTATIONS


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

def test_aistudiopersona_default_construction():
    p = AIStudioPersona()
    d = asdict(p)
    # Has every advertised key
    expected_keys = {
        "persona_archetype", "character_name", "backstory_brief",
        "relational_stance", "address_terms", "self_reference_style",
        "voice_traits", "communication_style", "default_register",
        "formality", "humor_tone", "length_band",
        "emoji_palette", "emoji_intensity_default",
        "signature_phrases", "forbidden_phrases",
        "topical_strengths", "topical_avoid",
        "generation_guardrails", "eligibility_signal", "fit_rationale",
        "niche_specifier", "romantic_specifier",
    }
    assert set(d.keys()) == expected_keys


def test_romanticspecifier_default_explicitness_is_sensual():
    rs = RomanticSpecifier()
    assert rs.explicitness_band == "sensual"
    assert rs.gender_presentation is None
    assert rs.relational_dynamic is None


def test_userprofile_has_ai_studio_persona_field():
    up = UserProfile()
    assert hasattr(up, "ai_studio_persona")
    assert up.ai_studio_persona == {}


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

def _build_test_prompt(**kw):
    """Build the prompt with sensible defaults; kw lets tests override."""
    menu = [{"name": k, **v} for k, v in AI_STUDIO_ARCHETYPES.items()]
    defaults = dict(
        profile={"gender": "female", "race_ethnicity": "white",
                 "career": "engineer", "education": "BS",
                 "bio": "loves climbing"},
        user_voice={"natural_register": "casual", "humor_tone": "dry",
                    "formality_baseline": 0.3, "emoji_intensity_default": "low"},
        app_personas={},
        hidden_personas_brief=[
            {"persona_type": "aspiration", "label": "Career-driven",
             "description": "wants growth"},
        ],
        sensitive_event_topics=[],
        sensitive_event_acuity={},
        top_hashtags=["python", "climbing", "coffee"],
        archetypes_menu=menu,
        rogers_cliche_baseline=ROGERS_CLICHE_BLOCKLIST,
        locale_country="US",
    )
    defaults.update(kw)
    return prompts.personalize_ai_studio_persona_prompt(**defaults)


def test_prompt_includes_all_archetypes():
    out = _build_test_prompt()
    for name in AI_STUDIO_ARCHETYPES:
        assert f"**{name}**" in out, f"{name!r} missing from menu"


def test_prompt_includes_full_sub_typing_vocabularies():
    out = _build_test_prompt()
    for v in ("hot_nerd", "dark_academia", "femboy", "elder_sis_romantic",
              "mommy", "pet", "soft_affection", "erotic_explicit"):
        assert v in out, f"{v!r} missing from prompt body"


def test_prompt_includes_rogers_baseline():
    out = _build_test_prompt()
    for phrase in ROGERS_CLICHE_BLOCKLIST:
        assert phrase in out, f"{phrase!r} missing from baseline section"


def test_prompt_includes_user_specific_grounding():
    out = _build_test_prompt(top_hashtags=["unique_test_hashtag_xyz"])
    assert "unique_test_hashtag_xyz" in out


# ---------------------------------------------------------------------------
# generate_ai_studio_persona — stubbed-LLM end-to-end tests
# ---------------------------------------------------------------------------

class _StubAgent(PersonaAgent):
    """PersonaAgent subclass that stubs LLM calls. Constructed without the
    normal __init__ so we can hand-build a minimal user_profile + interactions
    state."""

    def __init__(self, user_id="test", canned_response="{}", verbose=False):
        # Bypass real __init__ — set the attributes the method actually reads.
        self.user_id = user_id
        self.verbose = verbose
        self.interactions = []
        self.user_profile = UserProfile(name="Test", gender="female",
                                        race_ethnicity="white",
                                        career="engineer",
                                        education="BS",
                                        bio="loves climbing")
        self.user_profile.user_voice = {"natural_register": "casual"}
        self.user_profile.app_personas = {}
        self.user_profile.hidden_personas = []
        self._canned_response = canned_response
        self.llm_client = object()        # truthy, never called
        self.llm_client_mini = object()   # truthy, never called

    # Override the only LLM call the method makes.
    def _query_mini_with_retry(self, prompt: str) -> str:
        return self._canned_response

    # Stub hashtag extractor — _infer_locale_country uses interactions only.
    def _extract_hashtags(self, text: str):
        return []


def _canned(persona_dict: dict) -> str:
    """Wrap a persona dict in a fenced JSON code block (matches what
    `extract_json_from_response` expects)."""
    return f"```json\n{json.dumps(persona_dict)}\n```"


def test_generate_persona_happy_path_late_night_best_friend():
    canned = _canned({
        "persona_archetype": "late_night_best_friend",
        "character_name": "Jules",
        "backstory_brief": "Jules is your unflappable midnight friend.",
        "relational_stance": "warm-confidant",
        "address_terms": ["you"],
        "self_reference_style": "first_person",
        "voice_traits": ["dry", "warm", "patient", "real"],
        "communication_style": "Talks like a tired best friend on a porch.",
        "default_register": "warm_casual",
        "formality": 0.2,
        "humor_tone": "deadpan-affectionate",
        "length_band": "medium",
        "emoji_palette": [],
        "emoji_intensity_default": "low",
        "signature_phrases": ["okay so"],
        "forbidden_phrases": ["I hear you"],   # only 1 — must be back-filled
        "topical_strengths": ["aspiration", "venting"],
        "topical_avoid": [],
        "generation_guardrails": {},
        "eligibility_signal": {},
        "fit_rationale": "Default for unspecified profile.",
        "niche_specifier": None,
        "romantic_specifier": {},
    })
    agent = _StubAgent(canned_response=canned, verbose=False)
    agent.generate_ai_studio_persona()

    p = agent.user_profile.ai_studio_persona
    assert p["persona_archetype"] == "late_night_best_friend"
    assert p["character_name"] == "Jules"
    # Forbidden phrases back-filled with the full Rogers baseline + archetype-specific
    fp = " | ".join(p["forbidden_phrases"]).lower()
    for phrase in ROGERS_CLICHE_BLOCKLIST:
        assert phrase.lower() in fp, f"{phrase!r} not back-filled"
    # The archetype-specific forbidden_phrases for late_night_best_friend
    assert any("as your friend, i think" in s.lower() for s in p["forbidden_phrases"])


def test_generate_persona_invalid_archetype_falls_back():
    canned = _canned({
        "persona_archetype": "made_up_archetype_xyz",
        "character_name": "Wren",
        "fit_rationale": "test",
    })
    agent = _StubAgent(canned_response=canned)
    agent.generate_ai_studio_persona()
    assert agent.user_profile.ai_studio_persona["persona_archetype"] == "late_night_best_friend"


def test_generate_persona_signature_phrases_capped_at_three():
    canned = _canned({
        "persona_archetype": "mentor_coach",
        "signature_phrases": ["a", "b", "c", "d", "e", "f"],
        "fit_rationale": "test",
    })
    agent = _StubAgent(canned_response=canned)
    agent.generate_ai_studio_persona()
    assert len(agent.user_profile.ai_studio_persona["signature_phrases"]) == 3


def test_generate_persona_romantic_specifier_validates_vocabularies():
    canned = _canned({
        "persona_archetype": "romantic_partner",
        "character_name": "Cass",
        "fit_rationale": "test",
        "romantic_specifier": {
            "gender_presentation": "nonbinary",
            "sexuality_orientation": "queer_unspecified",
            "aesthetic_vibe": "hot_nerd",          # valid
            "body_role_coding": "femboy",          # valid
            "relational_dynamic": "mommy",         # valid
            "explicitness_band": "erotic_explicit",
            # Inject an invalid value to confirm it gets nulled
            "gender_presentation_invalid": "alien",
        },
        # No active high-acuity sensitive_life_event => should NOT auto-disable
    })
    agent = _StubAgent(canned_response=canned)
    agent.generate_ai_studio_persona()
    p = agent.user_profile.ai_studio_persona
    assert p["persona_archetype"] == "romantic_partner"
    rs = p["romantic_specifier"]
    assert rs["gender_presentation"] == "nonbinary"
    assert rs["aesthetic_vibe"] == "hot_nerd"
    assert rs["body_role_coding"] == "femboy"
    assert rs["relational_dynamic"] == "mommy"
    assert rs["explicitness_band"] == "erotic_explicit"


def test_generate_persona_romantic_specifier_invalid_value_becomes_none():
    canned = _canned({
        "persona_archetype": "romantic_partner",
        "fit_rationale": "test",
        "romantic_specifier": {
            "gender_presentation": "fictional_gender_xyz",  # invalid
            "aesthetic_vibe": "not_a_real_vibe",            # invalid
            "explicitness_band": "ultra_extreme",           # invalid -> default
        },
    })
    agent = _StubAgent(canned_response=canned)
    agent.generate_ai_studio_persona()
    rs = agent.user_profile.ai_studio_persona["romantic_specifier"]
    assert rs["gender_presentation"] is None
    assert rs["aesthetic_vibe"] is None
    assert rs["explicitness_band"] == "sensual"  # default


def test_generate_persona_romantic_auto_disables_on_high_acuity_sle():
    """Generation guard: high-acuity active sensitive_life_event must
    flip a romantic_partner pick off to a non-romantic fallback."""
    canned = _canned({
        "persona_archetype": "romantic_partner",
        "fit_rationale": "test",
        "romantic_specifier": {"explicitness_band": "erotic_explicit"},
    })
    agent = _StubAgent(canned_response=canned)
    # Inject a high-acuity active sensitive_life_event hidden persona.
    from data_preparation.persona_agent import HiddenPersona
    hp = HiddenPersona(
        label="Acute mental health diagnosis",
        type="sensitive_life_event",
        events=[{
            "topic": "mental_health_diagnosis",
            "active_window_end": 10**12,  # far future => still active
        }],
    )
    agent.user_profile.hidden_personas = [hp]
    agent.generate_ai_studio_persona()
    p = agent.user_profile.ai_studio_persona
    assert p["persona_archetype"] == "late_night_best_friend"
    assert p["romantic_specifier"] == {}


def test_generate_persona_niche_expert_backfills_specifier():
    canned = _canned({
        "persona_archetype": "niche_expert_creator_ai",
        "fit_rationale": "test",
        "niche_specifier": None,  # missing -> should be back-filled
    })
    agent = _StubAgent(canned_response=canned)
    # Stub _extract_hashtags to return something so the back-fill picks a niche.
    agent._extract_hashtags = lambda text: ["climbing"] if text else []  # type: ignore
    # Need at least one InteractionRow with object_text so the counter sees 'climbing'.
    agent.interactions = [InteractionRow(
        interaction_type="explicit_positive", user_id="test",
        object_id="o1", interaction_time=0, object_text="#climbing",
    )]
    agent.generate_ai_studio_persona()
    p = agent.user_profile.ai_studio_persona
    assert p["persona_archetype"] == "niche_expert_creator_ai"
    assert p["niche_specifier"] is not None
    assert p["niche_specifier"]   # non-empty


def test_generate_persona_unparseable_response_leaves_empty():
    agent = _StubAgent(canned_response="not valid json at all")
    agent.generate_ai_studio_persona()
    assert agent.user_profile.ai_studio_persona == {}


def test_generate_persona_skipped_if_already_cached():
    """Re-running Step 11C should be a no-op when ai_studio_persona is set."""
    agent = _StubAgent(canned_response=_canned({
        "persona_archetype": "mentor_coach",
        "fit_rationale": "should not run",
    }))
    agent.user_profile.ai_studio_persona = {"persona_archetype": "wise_elder_grandparent"}
    agent.generate_ai_studio_persona()
    assert agent.user_profile.ai_studio_persona["persona_archetype"] == "wise_elder_grandparent"


def test_generate_persona_skipped_when_no_llm_client():
    agent = _StubAgent(canned_response=_canned({
        "persona_archetype": "mentor_coach",
        "fit_rationale": "no client",
    }))
    agent.llm_client = None
    agent.llm_client_mini = None
    agent.generate_ai_studio_persona()
    assert agent.user_profile.ai_studio_persona == {}


if __name__ == "__main__":
    # Standalone runner — useful when pytest isn't available.
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
