"""Task: personalized_recommendation

Models a *proactive recsys feed push*: at each picked t_test, the agent
is shown a slate of candidates (1 held-out + 7 hard negatives + fillers)
and asked to rank them as if it were the recsys deciding what to surface
next in the user's feed. There is no user-typed query — the user_query
field is left empty so the runner knows to skip the chat preamble and
just rank the slate.

Slate construction:
  - held_out: a real positive engagement the user has AFTER t_test inside
    the anchor window. The agent's job is to rank this at position 1.
  - hard_negatives: real items the user negatively engaged with (or items
    whose hashtags overlap held_out's but the user didn't engage). These
    are surface-similar — testing whether the agent picks up on the
    user's actual preference signal vs. just hashtag co-occurrence.
  - fillers: random pre-t_test events with NO hashtag overlap (noise).

Time-masking: t_test cuts history. The agent must rank candidates
without seeing the user's actual post-t_test engagement.

Multi-anchor fan-out: per-day quotas were too restrictive (≤8 days × strict
hard-negative gate left most users with ~5 instances). The builder now
fans out 3–5 anchors per eligible day (morning / midday / afternoon /
evening / late-evening), so a single active user-day can yield several
slates with disjoint held-out + hard-negative pools.

Metrics: recall@k, ndcg@k, mrr, hit@k. Deterministic — no LLM judge.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections import Counter

from evaluation.backend_query import BackendQuery


# Targeted anchor count per user. Per-day fan-out (7 anchors/day across
# the active hours) on a typical 8-day window gives ~56 candidate anchors;
# after the hard-negatives gate drops a portion, surviving instances land
# comfortably in the 30–35 range that task_distribution.py targets.
PERSONALIZED_REC_DEFAULT_N_ANCHORS: int = 56
SLATE_SIZE: int = 16            # 1 held-out + 7 hard negatives + 8 fillers
N_HARD_NEGATIVES: int = 7

# Anchor offsets within a UTC day (hours past midnight UTC). Each yields
# a separate 3-hour slate window. 7 anchors × ~8 active days = 56 candidate
# instances upstream of floor-failure drops, comfortably above the 30/35
# task-distribution target.
_ANCHOR_HOURS: tuple[int, ...] = (5, 8, 11, 14, 17, 20, 23)
_ANCHOR_WINDOW_SECONDS: int = 3 * 3600

# Minimum hard negatives required for a slate to be ranking-worthy.
# Was 3 (per-day single-anchor design); softened to 2 for the multi-anchor
# fan-out so anchors with narrow-hashtag held-outs aren't all dropped.
# Hard negatives are the discriminative items in the slate; 2+ still
# tests the agent's ability to prefer the held-out over surface-similar
# rejected items.
_MIN_HARD_NEGATIVES: int = 2

_TASK_ID = "personalized_recommendation"


def _content_summary(e: dict) -> dict:
    """Compact projection for a candidate slate item.

    Falls back through title → caption → first hashtag → "post on
    {app}" so no candidate ever renders as an empty `<item>` placeholder
    in the user-facing query string."""
    content = e.get("content") or {}
    hashtags = list(e.get("source_hashtags") or [])[:8]
    app = e.get("_app", "")
    title = (content.get("title") or content.get("caption") or "").strip()
    if not title and hashtags:
        title = " ".join(h.lstrip("#") for h in hashtags[:3])
    if not title:
        title = f"post on {app}" if app else "post"
    return {
        "source_object_id": e.get("source_object_id", ""),
        "title": title[:120],
        "caption": (content.get("caption") or "")[:200],
        "hashtags": hashtags,
        "source_app": app,
        "source_timestamp": int(e.get("source_timestamp") or 0),
    }


def _hashtag_set(e: dict) -> set[str]:
    return {(h or "").lstrip("#").lower() for h in (e.get("source_hashtags") or []) if h}


def build_personalized_recommendation(
    bq: BackendQuery,
    user_id: str,
    t_anchor: int,
    n_anchors: int = PERSONALIZED_REC_DEFAULT_N_ANCHORS,
    rng_seed: int = 0,
) -> list[dict]:
    """Build proactive-recsys ranking instances scattered across the
    user's interaction window. Multi-anchor fan-out: each eligible day
    contributes up to len(_ANCHOR_HOURS) anchors (separated 4h+) so a
    single active day can yield 3–5 distinct slates instead of one.

    Each instance carries a 16-item slate where:
      - held_out is a real positive engagement inside the anchor's
        4-hour window (the next thing the user actually engaged with),
      - hard negatives are real items the user disliked or skipped that
        share ≥1 hashtag with held_out (drawn from history strictly
        before t_test),
      - fillers are random pre-t_test events with NO hashtag overlap.

    Time-mask discipline: only events with source_timestamp < t_test are
    visible to the agent. Held_out and hard negatives are not.
    """
    from evaluation.tasks.e3_daily_briefing_multi import (
        _collect_day_buckets,
        _events_in_window,
        _POSITIVE_INTERACTION_TYPES,
        _NEGATIVE_INTERACTION_TYPES,
    )

    pos_buckets = _collect_day_buckets(bq, user_id, positive_only=True)
    eligible = [d for d, rows in pos_buckets.items() if len(rows) >= 1]
    if not eligible:
        return []
    eligible.sort()
    if len(eligible) >= 3:
        eligible = eligible[1:-1]   # 24h guard on both ends

    # Build a flat pool of all events across the user (for filler sampling
    # and held-out / hard-negative gating).
    all_events: list[dict] = []
    for app in ("instagram", "facebook", "threads"):
        for e in bq._load_events(user_id, app):
            row = dict(e)
            row.setdefault("_app", app)
            all_events.append(row)
    all_events.sort(key=lambda e: int(e.get("source_timestamp") or 0))

    rng = random.Random(rng_seed or hash(user_id) % (2**31))

    # Build the anchor list: round-robin across days × anchor-hours so a
    # user with few active days still gets multiple anchors per day, but
    # we don't pile all anchors onto the busiest day. Capped at n_anchors.
    by_pos_volume_desc = sorted(eligible, key=lambda d: -len(pos_buckets[d]))
    candidate_anchors: list[tuple[str, int, int]] = []  # (day, anchor_idx, hour)
    for anchor_idx, hour in enumerate(_ANCHOR_HOURS):
        for day in by_pos_volume_desc:
            candidate_anchors.append((day, anchor_idx, hour))
    candidate_anchors = candidate_anchors[:n_anchors]

    instances: list[dict] = []
    used_holdouts_per_day: dict[str, set] = {}
    instance_counter = 0
    for day, anchor_idx, hour in candidate_anchors:
        dt0 = _dt.datetime.strptime(day, "%Y-%m-%d").replace(
            hour=hour, minute=0, second=0, tzinfo=_dt.timezone.utc
        )
        anchor_ts = int(dt0.timestamp())
        window_end = anchor_ts + _ANCHOR_WINDOW_SECONDS

        # Held-out: a positive engagement inside the anchor's 4-hour window.
        # Disjoint from prior anchors on the same day so multi-anchor
        # fan-out doesn't re-test the same item.
        used = used_holdouts_per_day.setdefault(day, set())
        pos_in_window = [
            e for e in _events_in_window(pos_buckets, day, anchor_ts, window_end)
            if e.get("source_object_id") not in used
        ]
        if not pos_in_window:
            continue
        held_out = pos_in_window[0]
        used.add(held_out.get("source_object_id"))
        held_hashtags = _hashtag_set(held_out)

        # Hard negatives: events in the user's history (strictly pre-t_test)
        # where the user explicitly or implicitly disliked, AND whose
        # hashtags overlap held_out's. Surface-similar but unwanted.
        neg_pool = [
            e for e in all_events
            if (e.get("source_interaction_type") or "") in _NEGATIVE_INTERACTION_TYPES
            and _hashtag_set(e) & held_hashtags
            and int(e.get("source_timestamp") or 0) < anchor_ts
        ]
        rng.shuffle(neg_pool)
        hard_negatives = neg_pool[:N_HARD_NEGATIVES]

        # If we don't have enough negative-engagement hard negatives, fall
        # back to events with overlapping hashtags but ZERO engagement
        # signal (the user didn't react either way — adjacent but
        # unwanted at this moment).
        if len(hard_negatives) < N_HARD_NEGATIVES:
            seen_ids = {e.get("source_object_id") for e in hard_negatives}
            seen_ids.add(held_out.get("source_object_id"))
            for e in all_events:
                if int(e.get("source_timestamp") or 0) >= anchor_ts:
                    continue
                if e.get("source_object_id") in seen_ids:
                    continue
                if not (_hashtag_set(e) & held_hashtags):
                    continue
                hard_negatives.append(e)
                seen_ids.add(e.get("source_object_id"))
                if len(hard_negatives) >= N_HARD_NEGATIVES:
                    break

        if len(hard_negatives) < _MIN_HARD_NEGATIVES:
            # Not enough adjacent items in the user's data to make a
            # meaningful ranking task — skip this anchor.
            continue

        # Fillers: random pre-t_test events with NO hashtag overlap (noise).
        used_ids = {held_out.get("source_object_id")} | {
            n.get("source_object_id") for n in hard_negatives
        }
        filler_pool = [
            e for e in all_events
            if e.get("source_object_id") not in used_ids
            and int(e.get("source_timestamp") or 0) < anchor_ts
            and not (_hashtag_set(e) & held_hashtags)
        ]
        rng.shuffle(filler_pool)
        n_fillers = SLATE_SIZE - 1 - len(hard_negatives)
        fillers = filler_pool[:n_fillers]

        # Assemble + shuffle the slate so held_out isn't always at idx=0.
        slate_rows: list[dict] = [held_out] + hard_negatives + fillers
        order = list(range(len(slate_rows)))
        rng.shuffle(order)
        slate = [_content_summary(slate_rows[j]) for j in order]
        held_out_idx = order.index(0)
        hard_negative_idxs = [order.index(j + 1) for j in range(len(hard_negatives))]

        # User-facing query: empty (proactive recsys feed push — there is
        # no user-typed query, the runner skips the chat preamble and just
        # ranks the slate).
        query_text = ""

        instances.append({
            "instance_id": f"recsys_{day}_a{anchor_idx}",
            "task_id": _TASK_ID,
            "entry_point": "chatbot_routed",
            "t_test": anchor_ts,
            "anchor_hour_utc": hour,
            "day_index": instance_counter,
            "day_label": day,
            "candidates": slate,
            "held_out_idx": held_out_idx,
            "hard_negative_idxs": hard_negative_idxs,
            "query_text": query_text,
        })
        instance_counter += 1
    return instances


# Backward-compat alias for any caller that still imports the old function name.
build_e4_google_search = build_personalized_recommendation


def personalized_recommendation_prompt(instance: dict, history_block: str | None) -> str:
    cands = instance.get("candidates") or []
    hour = int(instance.get("anchor_hour_utc") or 8)
    cand_lines = "\n".join(
        f"  [{i}] {c.get('title','')} (hashtags: {', '.join(c.get('hashtags', []))})"
        for i, c in enumerate(cands)
    )
    history = (
        f"\n## User history (time-masked to before {instance['day_label']} {hour:02d}:00 UTC)\n"
        f"{history_block}\n"
        if history_block else ""
    )
    # Two flavors share this prompt:
    #   1. Empty query_text → proactive recsys feed-push framing (no
    #      user-typed query, just a slate to rank).
    #   2. Real user query text → moment-aware curation framing (the user
    #      asked the agent for a curated feed at this moment). The merge
    #      from the old `agentic_moment_recommendation` task lands here:
    #      the agentic MCP-feed path is impractical without a live backend,
    #      so moment instances now ride the same deterministic ranking
    #      metric as proactive recsys but with a voiced user query.
    raw_query = (instance.get("query_text") or "").strip()
    is_recsys = not raw_query
    if is_recsys:
        framing = (
            f"It's {instance['day_label']} {hour:02d}:00 UTC. The recommendation "
            f"system has proposed {len(cands)} candidate items. Rank them in "
            f"the order the user is most likely to engage with, given their "
            f"time-masked history below."
        )
    else:
        moment = (instance.get("moment") or "").strip()
        moment_line = f" (moment: {moment})" if moment else ""
        framing = (
            f"It's {instance['day_label']} {hour:02d}:00 UTC{moment_line}. "
            f"The user just asked their agent: \"{raw_query}\"\n\n"
            f"Below are {len(cands)} candidate items already in the user's "
            f"feeds. Rank them in the order the user is most likely to want "
            f"to see right now, given the moment + their time-masked history."
        )
    return f"""# Task: personalized social-media content recommendation

{framing}

## Candidate slate
{cand_lines}
{history}
## Output
Respond with ONE fenced ```json block containing the ranked indexes:
```json
{{
  "ranked_indexes": [<idx>, <idx>, ...],
  "reasoning": "<=2 sentences"
}}
```"""


def _recall_at_k(ranked: list[int], target: int, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def _hit_at_k(ranked: list[int], target: int, k: int) -> float:
    return _recall_at_k(ranked, target, k)


def _mrr(ranked: list[int], target: int) -> float:
    for i, r in enumerate(ranked):
        if r == target:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(ranked: list[int], target: int, hard_negatives: list[int], k: int) -> float:
    """Single-target NDCG@K with hard-negatives at relevance 0 (penalty
    if ranked above the target). Held-out has relevance 1; everything
    else 0."""
    import math
    dcg = 0.0
    for i, r in enumerate(ranked[:k]):
        if r == target:
            dcg += 1.0 / math.log2(i + 2)
    return dcg  # ideal DCG = 1 / log2(2) = 1.0, so NDCG = dcg/1.0 = dcg


def compute_personalized_recommendation_metrics(parsed: dict, instance: dict) -> dict:
    ranked = parsed.get("ranked_indexes") or []
    target = instance.get("held_out_idx")
    hard_negs = instance.get("hard_negative_idxs") or []
    if not isinstance(target, int):
        return {"n_ranked": len(ranked), "valid": False}
    return {
        "n_ranked": len(ranked),
        "valid": True,
        "recall_at_1": _recall_at_k(ranked, target, 1),
        "recall_at_3": _recall_at_k(ranked, target, 3),
        "recall_at_5": _recall_at_k(ranked, target, 5),
        "ndcg_at_3":   round(_ndcg_at_k(ranked, target, hard_negs, 3), 4),
        "ndcg_at_5":   round(_ndcg_at_k(ranked, target, hard_negs, 5), 4),
        "mrr":         round(_mrr(ranked, target), 4),
        "hit_at_1":    _hit_at_k(ranked, target, 1),
        "hit_at_3":    _hit_at_k(ranked, target, 3),
    }


def run_personalized_recommendation(
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
    """Deterministic ranking-metric runner. The agent ranks the provided
    slate using only its time-masked history (no external tools)."""
    from data_preparation.utils import extract_json_from_response
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode == "llm_longctx":
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget
            )
            history_tokens = stats["total_tokens"]
        prompt = personalized_recommendation_prompt(inst, history_block)
        if dry_run:
            results.append({
                "task": _TASK_ID,
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "day_label": inst.get("day_label"),
                "mode": mode,
                "history_tokens": history_tokens,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, _ = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response) or {}
        m = compute_personalized_recommendation_metrics(parsed, inst)
        results.append({
            "task": _TASK_ID,
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "day_label": inst.get("day_label"),
            "mode": mode,
            "metrics": m,
            "agent_response": raw_response,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })
    return results


# Backward-compat aliases — kept so any caller still importing the old
# function/constant names resolves without code change. New callers should
# use the canonical names.
run_e4_google_search = run_personalized_recommendation
e4_prompt = personalized_recommendation_prompt
compute_e4_metrics = compute_personalized_recommendation_metrics
E4_DEFAULT_N_DAYS = PERSONALIZED_REC_DEFAULT_N_ANCHORS
