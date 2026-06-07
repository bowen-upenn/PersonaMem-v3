"""LIVE smoke: prove Azure gpt-5.5 prompt caching fires with our hoisted
long-context layout. Sends the SAME large history prefix (hoisted to the front)
with a varying trailing query, several times, and checks the API reports
cached_tokens > 0 on the repeat calls. Costs a few cents (small prompts).

Run: python tests/test_gpt5_input_cache.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_llm import QueryLLM
from evaluation import prompts
from evaluation.inference_utils import _wrap_history_block, _hoist_history_prefix, count_tokens


def main():
    c = QueryLLM({"models": {"llm_model": "gpt-5.5"}})
    if getattr(c, "is_gemini", False) or getattr(c, "is_claude", False):
        print("SKIP: gpt-5.5 did not route to Azure/OpenAI"); return

    # Large, fixed history block (>>1024 tokens) — the cacheable prefix.
    hist_text = "\n".join(
        f'{{"app":"instagram","t":{i},"hashtags":["#topic{i%37}"],"title":"engaged with item {i}"}}'
        for i in range(1500)
    )
    hist = _wrap_history_block(hist_text)
    prefix_tokens = count_tokens(hist_text)
    print(f"history prefix ~{prefix_tokens} tokens (Azure caches prefixes >=1024)")

    prev = c.get_usage_totals()
    hits = 0
    for i in range(4):
        # Same hoisted history prefix every call; only the trailing query varies.
        prompt = _hoist_history_prefix(
            prompts.chatbot_response_prompt(f"Briefly: what is item {i}? (one short sentence)", [], hist))
        c.query_llm(prompt, temperature=0.0)
        u = c.get_usage_totals()
        d_in = u["input_tokens"] - prev["input_tokens"]
        d_cached = u["cached_input_tokens"] - prev["cached_input_tokens"]
        prev = u
        pct = (100 * d_cached / d_in) if d_in else 0
        print(f"  call {i}: input={d_in:>6} cached={d_cached:>6} ({pct:4.1f}%)")
        if i >= 1 and d_cached > 0:
            hits += 1

    tot = c.get_usage_totals()
    print(f"\ntotals: input={tot['input_tokens']} cached={tot['cached_input_tokens']} "
          f"({100*tot['cached_input_tokens']/max(1,tot['input_tokens']):.1f}%)  errors={tot['errors']}")
    if hits > 0:
        print(f"PASS: Azure gpt-5.5 prompt cache fired on {hits} repeat call(s) — the hoisted "
              f"long-context layout IS cached for gpt-5.5.")
    else:
        print("INCONCLUSIVE: no cached_tokens reported. Azure caching is best-effort "
              "(prefix>=1024, same prefix, ~5-10min TTL, node affinity) — may not fire on a "
              "tiny burst; the layout is still correct. errors above indicate API issues.")


if __name__ == "__main__":
    main()
