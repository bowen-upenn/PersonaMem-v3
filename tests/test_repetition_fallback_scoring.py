"""Regression test for the judge-disabled fallback in the C1c/C1d repetition
runners (`evaluation/tasks/over_personalization.py`).

Bug (2026-06): with `enable_llm_judge=False` (or `judge_client=None`) the
fallback passed each agent-response **dict** straight into
`metrics.is_substantive_response`, whose `tokenize` then raised
`TypeError: expected string or bytes-like object, got 'dict'`
(evaluation/metrics.py:64). This errored 50/100 repetition rows in
`llm_longctx_gpt5.5` and all 50 in `llm_memory_gemini3.5flash`.

Fix: the fallback now flattens the dict to the same text the judge-enabled
branch scores (c1c: title+caption; c1d: the "response" field).
`tokenize` keeps its str-only contract — coercing dicts inside it would
count dict-repr noise (keys like "title"/"caption") as tokens.

No LLM spend: `_dispatch_agent` is monkeypatched with a canned-response stub.

Run: `python tests/test_repetition_fallback_scoring.py` (or pytest -q).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import metrics
from evaluation.tasks import over_personalization as op

SUBSTANTIVE_CAPTION = (
    "Here is a genuinely detailed recommendation covering several distinct "
    "angles: weekend market visits, beginner pottery classes, neighborhood "
    "walking routes, and a documentary series about urban gardening projects."
)
SUBSTANTIVE_CHAT = (
    "Great question — for a first marathon you want three quality sessions "
    "weekly: one long slow run, one tempo workout, one interval day, plus "
    "easy recovery jogs, strength work, and consistent sleep and fueling."
)


def _mk_queries(n: int) -> list[dict]:
    return [
        {"ts": 1700000000 + i * 3600, "user_query": f"query number {i}", "is_target": True}
        for i in range(n)
    ]


def _stub_dispatch(canned_raw: str):
    """Return a `_dispatch_agent`-shaped stub: (raw, tool_calls, stats)."""
    def _dispatch(mode, prompt, **kwargs):
        return canned_raw, 0, {}
    return _dispatch


def _run_c1c(cluster: dict) -> list[dict]:
    return op.run_task_c1c(
        [cluster], user_id="testU", bq=None, llm_client=None,
        judge_client=None, mode="llm", snapshot_cache=None,
        model_name=None, claude_model="", context_budget=None,
        enable_llm_judge=False, dry_run=False,
    )


def _run_c1d(cluster: dict) -> list[dict]:
    return op.run_task_c1d(
        [cluster], user_id="testU", bq=None, llm_client=None,
        judge_client=None, mode="llm", snapshot_cache=None,
        model_name=None, claude_model="", context_budget=None,
        enable_llm_judge=False, dry_run=False,
    )


def test_tokenize_str_contract():
    # tokenize keeps a str-only contract; callers must flatten dicts.
    assert metrics.tokenize("Hello WORLD #foo ab") == {"hello", "world", "#foo"}
    assert metrics.is_substantive_response(SUBSTANTIVE_CAPTION)
    assert not metrics.is_substantive_response("")
    try:
        metrics.tokenize({"title": "x"})  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("tokenize silently accepted a dict — callers now "
                             "rely on the str contract; do not coerce here")
    print("  ✓ tokenize_str_contract")


def test_c1c_judge_off_fallback_scores_dict_responses():
    # Previously raised TypeError at metrics.py tokenize (dict input).
    orig = op._dispatch_agent
    op._C1C_CLUSTER_CACHE.clear()
    op._dispatch_agent = _stub_dispatch(json.dumps({
        "title": "Five fresh ideas",
        "caption": SUBSTANTIVE_CAPTION,
        "hashtags": ["#pottery", "#markets"],
    }))
    try:
        out = _run_c1c({
            "cluster_id": "regress-c1c-substantive",
            "queries": _mk_queries(5),
            "target_pref": "loves indie board games",
            "primary_category": "gaming",
            "persona_hint": {"top_categories": ["gaming"], "top_hashtags": ["#boardgames"]},
            "off_persona_distractor_hashtags": [],
            "cluster_hashtags": ["#boardgames"],
            "n_allowed_repetitions": 2,
        })
    finally:
        op._dispatch_agent = orig
    assert len(out) == 1, f"expected 1 cluster result, got {len(out)}"
    res = out[0]
    assert res["pref_invoked_per_response"] == [10.0] * 5, res["pref_invoked_per_response"]
    m = res["metrics"]
    assert m["n_total"] == 5 and m["fatigue_passed"] is True, m
    assert m["query_score_0_10"] == 10.0, m
    print("  ✓ c1c_judge_off_fallback_scores_dict_responses")


def test_c1c_fallback_gates_empty_responses():
    # Empty/refusal turns must still gate to 0 (silence is not restraint).
    orig = op._dispatch_agent
    op._C1C_CLUSTER_CACHE.clear()
    op._dispatch_agent = _stub_dispatch(json.dumps({
        "title": "", "caption": "no.", "hashtags": [],
    }))
    try:
        out = _run_c1c({
            "cluster_id": "regress-c1c-empty",
            "queries": _mk_queries(4),
            "target_pref": "loves indie board games",
            "primary_category": "gaming",
            "persona_hint": {},
            "n_allowed_repetitions": 2,
        })
    finally:
        op._dispatch_agent = orig
    res = out[0]
    assert res["pref_invoked_per_response"] == [0.0] * 4, res["pref_invoked_per_response"]
    assert res["metrics"]["fatigue_passed"] is False, res["metrics"]
    print("  ✓ c1c_fallback_gates_empty_responses")


def test_c1d_judge_off_fallback_scores_dict_responses():
    # c1d response dicts carry the text under "response" — same crash, same fix.
    orig = op._dispatch_agent
    op._C1D_CLUSTER_CACHE.clear()
    op._dispatch_agent = _stub_dispatch(json.dumps({"response": SUBSTANTIVE_CHAT}))
    try:
        out = _run_c1d({
            "cluster_id": "regress-c1d-substantive",
            "queries": _mk_queries(5),
            "target_pref": "obsessed with marathon training",
            "primary_category": "fitness",
            "target_hashtags": ["#marathon"],
            "n_allowed_repetitions": 2,
        })
    finally:
        op._dispatch_agent = orig
    assert len(out) == 1, f"expected 1 cluster result, got {len(out)}"
    res = out[0]
    assert res["pref_invoked_per_response"] == [10.0] * 5, res["pref_invoked_per_response"]
    assert res["metrics"]["fatigue_passed"] is True, res["metrics"]
    print("  ✓ c1d_judge_off_fallback_scores_dict_responses")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} repetition-fallback regression tests (no LLM)...")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\nFAILED {failed}/{len(tests)}")
        sys.exit(1)
    print(f"\nAll {len(tests)} passed ✓")


if __name__ == "__main__":
    _run_all()
