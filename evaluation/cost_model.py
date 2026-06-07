"""Token-cost model for the LLM-baseline eval modes.

Single source of truth for provider rates + the cache/batch discount math, so
the cost report and the savings smoke tests agree. Rates are $ per 1M tokens
(June 2026, Google AI for Developers pricing).

Discount stacking (Gemini, per Google docs):
  * Context-cache hits bill at `cached_in` (~10x cheaper than `in`).
  * Batch mode bills NON-cached input + output at 50%.
  * They do NOT multiply: a cache hit bills at the cache rate with NO further
    batch discount; batch's 50% applies only to the remaining non-cached tokens.
Implicit caching carries no storage fee; explicit caching adds `cache_storage`
per 1M tokens per hour (set to None where not modeled).
"""
from __future__ import annotations

# $ per 1M tokens. `in_hi`/`out_hi` are the >200K long-context tier (Pro only).
# in/out/cached_in for gemini-3.5-flash and gemini-3-flash-preview are
# provider-confirmed (ai.google.dev pricing, June 2026). The flash-lite and pro
# `cached_in` values are estimated (10x-off heuristic) and not load-bearing for
# the gemini-3.5-flash re-estimate; confirm before relying on them.
RATES = {
    "gemini-3.5-flash":        {"in": 1.50, "out": 9.00, "cached_in": 0.15, "cache_storage": 1.00},
    "gemini-3-flash-preview":  {"in": 0.50, "out": 3.00, "cached_in": 0.05, "cache_storage": 1.00},
    "gemini-3.1-flash-lite":   {"in": 0.25, "out": 1.50, "cached_in": 0.025, "cache_storage": 1.00},
    "gemini-3.1-pro": {"in": 2.00, "out": 12.00, "in_hi": 4.00, "out_hi": 18.00,
                       "cached_in": 0.20, "tier_threshold": 200_000},
}

PER_M = 1_000_000.0


def gemini_cost(model, input_tokens, output_tokens, cached_input_tokens=0, *, batch=False):
    """Return USD cost for one workload on a single-tier Gemini model.

    cached_input_tokens are the subset of input_tokens served from a context
    cache (billed at `cached_in`; batch does NOT further discount them).
    """
    r = RATES[model]
    if "in_hi" in r:
        raise ValueError(f"{model} is tiered; use gemini_cost_tiered")
    cached = max(0, min(cached_input_tokens, input_tokens))
    uncached = input_tokens - cached
    bf = 0.5 if batch else 1.0
    cost = (uncached * r["in"] * bf
            + cached * r["cached_in"]
            + output_tokens * r["out"] * bf) / PER_M
    return cost


def gemini_cost_tiered(model, uncached_lo, uncached_hi, cached, out_lo, out_hi, *, batch=False):
    """Cost for a tiered model (e.g. gemini-3.1-pro) where prompts >200K bill at
    the high tier. Inputs are pre-split token counts. cached tokens bill at the
    single cache rate regardless of tier; batch halves non-cached only."""
    r = RATES[model]
    bf = 0.5 if batch else 1.0
    cost = (uncached_lo * r["in"] * bf + uncached_hi * r["in_hi"] * bf
            + cached * r["cached_in"]
            + out_lo * r["out"] * bf + out_hi * r["out_hi"] * bf) / PER_M
    return cost
