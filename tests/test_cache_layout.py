"""Smoke tests for the long-context cache layout.

FOCUS: tokens (and $) SAVED by the history-first hoist + chronological
serialization — NOT response accuracy. Each test quantifies how much of the
history becomes a reusable cache prefix and what that saves on gemini-3.5-flash.
No API calls.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import prompts
from evaluation.cost_model import gemini_cost
from evaluation.inference_utils import (
    _hoist_history_prefix,
    _wrap_history_block,
    count_tokens,
    HIST_SENTINEL_START,
    HIST_SENTINEL_END,
)


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return a[:n]


def test_layout_makes_history_a_cacheable_prefix():
    """The hoist must put the whole history ahead of the (varying) user query,
    so the history tokens are reusable. Quantify the cacheable token count."""
    hist_text = "EVENT-A\nEVENT-B\nEVENT-C\n" + ("token " * 5000)
    hist = _wrap_history_block(hist_text)
    p1 = _hoist_history_prefix(prompts.chatbot_response_prompt("Question ONE about shoes?", [], hist))
    p2 = _hoist_history_prefix(prompts.chatbot_response_prompt("A different question TWO?", [], hist))
    assert HIST_SENTINEL_START not in p1 and HIST_SENTINEL_END not in p1, "sentinels leaked to model"

    shared = _common_prefix(p1, p2)
    cacheable_tokens = count_tokens(shared)
    hist_tokens = count_tokens(hist_text)
    # ≥95% of the history must sit in the shared (cacheable) prefix.
    assert cacheable_tokens >= 0.95 * hist_tokens, (
        f"only {cacheable_tokens}/{hist_tokens} history tokens are cacheable")
    print(f"PASS: {cacheable_tokens}/{hist_tokens} history tokens become a reusable cache prefix "
          f"({100*cacheable_tokens/hist_tokens:.0f}%)")


def test_repeated_query_saves_tokens_and_dollars():
    """On the 2nd+ query at the same (user,T) the history bills at the cache
    rate, not full. Assert the realized $ saving on gemini-3.5-flash."""
    hist_text = "token " * 80_000          # ~80K-token history
    hist = _wrap_history_block(hist_text)
    q1 = _hoist_history_prefix(prompts.chatbot_response_prompt("first question?", [], hist))
    q2 = _hoist_history_prefix(prompts.chatbot_response_prompt("second different question?", [], hist))
    hist_tokens = count_tokens(_common_prefix(q1, q2))

    in_q2 = count_tokens(q2)
    out_tokens = 200
    # Without cache: full input rate every call.
    no_cache = gemini_cost("gemini-3.5-flash", in_q2, out_tokens, cached_input_tokens=0)
    # With cache: the shared history bills at the cache rate on the repeat.
    with_cache = gemini_cost("gemini-3.5-flash", in_q2, out_tokens, cached_input_tokens=hist_tokens)
    saving = (no_cache - with_cache) / no_cache
    assert with_cache < no_cache and saving > 0.7, f"cache saved only {saving:.0%}"
    print(f"PASS: repeat query input ${no_cache:.4f} -> ${with_cache:.4f} "
          f"({saving:.0%} cheaper via cache on {hist_tokens} reused tokens)")


def test_chronological_enables_cross_T_prefix_reuse():
    """A later-T history must be a prefix-EXTENSION of an earlier-T one, so
    cross-T queries reuse the earlier tokens. Quantify reused tokens."""
    t1_body = "\n".join(f'{{"app":"instagram","t":{i}}}' for i in range(2000))
    t2_body = t1_body + "\n" + "\n".join(f'{{"app":"instagram","t":{i}}}' for i in range(2000, 2600))
    h1 = _hoist_history_prefix(prompts.chatbot_response_prompt("q", [], _wrap_history_block(t1_body)))
    h2 = _hoist_history_prefix(prompts.chatbot_response_prompt("q", [], _wrap_history_block(t2_body)))
    reused = count_tokens(_common_prefix(h1, h2))
    t1_tokens = count_tokens(t1_body)
    assert reused >= 0.95 * t1_tokens, f"cross-T reuse only {reused}/{t1_tokens}"
    print(f"PASS: later-T query reuses {reused} earlier-T tokens "
          f"({100*reused/t1_tokens:.0f}% of the earlier cut billed at cache rate)")


def test_no_sentinel_is_noop():
    plain = "no history here\n## Current user query\nhello"
    assert _hoist_history_prefix(plain) == plain, "hoist mutated a sentinel-free prompt"
    print("PASS: hoist is a no-op without sentinels (safe for tool/agent modes)")


if __name__ == "__main__":
    test_layout_makes_history_a_cacheable_prefix()
    test_repeated_query_saves_tokens_and_dollars()
    test_chronological_enables_cross_T_prefix_reuse()
    test_no_sentinel_is_noop()
    print("\nALL CACHE-LAYOUT (SAVINGS) SMOKE TESTS PASSED")
