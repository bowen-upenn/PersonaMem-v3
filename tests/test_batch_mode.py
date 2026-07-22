"""Smoke tests for Gemini batch mode in QueryLLM.

FOCUS: tokens (and $) SAVED by batch + cache — NOT response accuracy. The mock
reports realistic per-request usage (mostly cache hits) and the tests assert the
realized dollar savings from the accumulated usage, using the same cost model
the cost report uses.
  * mock layer (default, no network): batch is taken by default for Gemini;
    EVAL_GEMINI_BATCH=0 disables it; savings math holds on accumulated usage.
  * live layer (opt-in via PM3_BATCH_LIVE=1): submits a real 2-prompt batch.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.cost_model import gemini_cost

# Per-request usage the mock reports: a ~400K-token long-context prompt where
# ~92% is a cache hit (matches the measured long-context split).
PROMPT_TOK = 400_000
CACHED_TOK = 368_000     # 92% cache hits
OUTPUT_TOK = 260


def _mk_client(model="gemini-3.5-flash"):
    from query_llm import QueryLLM
    return QueryLLM({"models": {"llm_model": model}})


class _FakeResp:
    def __init__(self):
        self.text = "x"  # body irrelevant; we test tokens, not accuracy
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=PROMPT_TOK,
            candidates_token_count=OUTPUT_TOK,
            cached_content_token_count=CACHED_TOK)


class _FakeBatches:
    def __init__(self):
        self.created_src = None

    def create(self, *, model, src, config=None):
        self.created_src = src
        inlined = [types.SimpleNamespace(metadata={"idx": ir.metadata["idx"]},
                                         response=_FakeResp(), error=None) for ir in src]
        return types.SimpleNamespace(
            name="batches/fake123",
            state=types.SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            dest=types.SimpleNamespace(inlined_responses=inlined), error=None)


def test_use_batch_default_on_for_gemini():
    c = _mk_client()
    assert c.is_gemini and c.use_batch, "batch should default ON for Gemini"
    print("PASS: use_batch defaults ON for Gemini")


def test_env_disables_batch():
    os.environ["EVAL_GEMINI_BATCH"] = "0"
    try:
        c = _mk_client()
        assert c.is_gemini and not c.use_batch, "EVAL_GEMINI_BATCH=0 must disable batch"
    finally:
        del os.environ["EVAL_GEMINI_BATCH"]
    print("PASS: EVAL_GEMINI_BATCH=0 disables batch")


def test_batch_plus_cache_saves_dollars_vs_standard():
    """Run N prompts through the batch path; from the accumulated usage compute
    the realized cost and compare to the no-cache/no-batch baseline."""
    c = _mk_client()
    c.client = types.SimpleNamespace(batches=_FakeBatches())
    n = 8
    c.query_many([f"q{i}" for i in range(n)])
    u = c.get_usage_totals()
    assert u["calls"] == n, u

    # Baseline: every input token at full rate, no batch.
    baseline = gemini_cost("gemini-3.5-flash", u["input_tokens"], u["output_tokens"], cached_input_tokens=0)
    # Realized: cache hits at cache rate + batch 50% off the rest.
    realized = gemini_cost("gemini-3.5-flash", u["input_tokens"], u["output_tokens"],
                           cached_input_tokens=u["cached_input_tokens"], batch=True)
    saving = (baseline - realized) / baseline
    assert u["cached_input_tokens"] == n * CACHED_TOK, "cache hits not accumulated"
    assert realized < baseline and saving > 0.85, f"batch+cache saved only {saving:.0%}"
    print(f"PASS: {n} batched prompts — ${baseline:.2f} -> ${realized:.2f} "
          f"({saving:.0%} saved via cache+batch); cached={u['cached_input_tokens']:,} tok")


def test_batch_and_cache_each_contribute_savings():
    """Decompose: cache alone and batch alone must each beat the baseline, and
    together beat either one (savings stack, though not multiplicatively)."""
    n = 8
    tin, tout, tcached = n * PROMPT_TOK, n * OUTPUT_TOK, n * CACHED_TOK
    base = gemini_cost("gemini-3.5-flash", tin, tout, cached_input_tokens=0)
    cache_only = gemini_cost("gemini-3.5-flash", tin, tout, cached_input_tokens=tcached)
    batch_only = gemini_cost("gemini-3.5-flash", tin, tout, cached_input_tokens=0, batch=True)
    both = gemini_cost("gemini-3.5-flash", tin, tout, cached_input_tokens=tcached, batch=True)
    assert cache_only < base and batch_only < base, "each optimization must save vs baseline"
    assert both < cache_only and both < batch_only, "combining must beat either alone"
    print(f"PASS: base ${base:.2f} | cache ${cache_only:.2f} | batch ${batch_only:.2f} | both ${both:.2f}")


def test_repetition_tasks_are_not_batchable():
    """Response-feedback clusters must be flagged non-batchable; independent
    tasks must stay batchable."""
    from evaluation.inference_utils import is_batchable_task
    assert not is_batchable_task("over_personalization_repetition_recsys")
    assert not is_batchable_task("over_personalization_repetition_chatbot")
    for ok in ("slate_ranking", "chatbot_response", "e2_at_ai_followup", None):
        assert is_batchable_task(ok), f"{ok} should be batchable"
    print("PASS: repetition tasks flagged non-batchable; independent tasks batchable")


def test_allow_batch_false_runs_sequentially():
    """allow_batch=False must NOT submit a batch even for Gemini."""
    c = _mk_client()
    class _Boom:
        def create(self, **k): raise AssertionError("batch must NOT be used for a sequential-dependency task")
    c.client = types.SimpleNamespace(batches=_Boom())
    seen = []
    c.query_llm = lambda p, temperature=None: (seen.append(p) or f"resp:{p}")  # type: ignore
    out = c.query_many(["a", "b", "c"], allow_batch=False)
    assert out == ["resp:a", "resp:b", "resp:c"] and seen == ["a", "b", "c"], out
    print("PASS: allow_batch=False runs sequentially (no batch submitted)")


def test_repetition_feedback_dependency_is_real():
    """Prove the repetition prompts can't be batched: query k+1's PROMPT embeds
    the agent's response to query k, so the prompts aren't known upfront."""
    from evaluation import prompts as P
    responses = []
    built = []
    for i in range(3):
        pr = P.over_personalization_repetition_recsys_prompt(
            target_pref="running shoes", primary_category="fitness",
            user_query=f"recommend something (turn {i})",
            persona_top_categories=["fitness"], persona_top_hashtags=["#run"],
            off_persona_distractor_hashtags=["#crypto"],
            prior_responses=list(responses), n_allowed_repetitions=2, history_block=None)
        built.append(pr)
        # Simulate the agent's answer feeding the NEXT prompt.
        responses.append({"title": f"UNIQUE_TITLE_{i}", "hashtags": [f"#tag{i}"]})
    # Turn 1's prompt must reflect turn 0's response; turn 0's must not.
    assert "UNIQUE_TITLE_0" not in built[0], "turn 0 cannot know any response yet"
    assert "UNIQUE_TITLE_0" in built[1], "turn 1 must embed turn 0's response (feedback dep)"
    assert "UNIQUE_TITLE_1" in built[2], "turn 2 must embed turn 1's response (feedback dep)"
    print("PASS: repetition prompts depend on prior responses → correctly run sequentially")


def test_live_batch_submission():
    if os.getenv("PM3_BATCH_LIVE") != "1":
        print("SKIP: live batch (set PM3_BATCH_LIVE=1 to run; costs a few cents)")
        return
    c = _mk_client()
    out = c.query_llm_batch(["Reply with the single word OK.", "What is 2+2? Reply with only the number."],
                            poll_interval=15, max_wait=1800)
    print(f"LIVE batch outputs: {out!r}")
    assert len(out) == 2 and any(o for o in out), "live batch returned no usable output"
    print("PASS: live batch submission completed")


if __name__ == "__main__":
    test_use_batch_default_on_for_gemini()
    test_env_disables_batch()
    test_batch_plus_cache_saves_dollars_vs_standard()
    test_batch_and_cache_each_contribute_savings()
    test_repetition_tasks_are_not_batchable()
    test_allow_batch_false_runs_sequentially()
    test_repetition_feedback_dependency_is_real()
    test_live_batch_submission()
    print("\nALL BATCH-MODE SMOKE TESTS PASSED")
