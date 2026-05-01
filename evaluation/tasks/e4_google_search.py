"""Task: personalized_recommendation
(was: personalized_search_ranking / e4_google_search — workstream D rename)

Renamed and rebuilt in workstream D. The task is now a *social-media
content recommendation* benchmark: at each picked t_test, the agent is
shown a slate of candidates (1 held-out + 7 hard negatives + fillers)
and asked to rank them. Held-out is a real positive engagement the user
will have AFTER t_test on that day. Hard negatives are real items the
user negatively engaged with (or items whose hashtags overlap held-out's
but the user didn't engage). Fillers are random other content.

Time-masking: t_test cuts history. The agent must rank candidates
without seeing the user's actual post-t_test engagement.

Metrics: recall@k, ndcg@k, mrr, hit@k. Deterministic — no LLM judge.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections import Counter

from evaluation.backend_query import BackendQuery


E4_DEFAULT_N_DAYS: int = 8
SLATE_SIZE: int = 16            # 1 held-out + 7 hard negatives + 8 fillers
N_HARD_NEGATIVES: int = 7

# Backward-compat alias kept so any caller that still imports the old name
# resolves to the new builder. The builder's emitted task_id is the new one.
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


def build_e4_google_search(
    bq: BackendQuery,
    user_id: str,
    t_anchor: int,
    n_days: int = E4_DEFAULT_N_DAYS,
    rng_seed: int = 0,
) -> list[dict]:
    """Workstream D: scatter ranking-style instances across the user's
    interaction window. Each instance carries a 16-item slate where the
    held-out is a real positive engagement that day after t_test, hard
    negatives are real items the user disliked or skipped that share
    hashtags with held-out, and fillers are random non-overlapping
    items.

    Time-mask discipline: only events with source_timestamp < t_test are
    visible to the agent. The held-out + hard negatives all live AFTER
    t_test or in the user's general history with no engagement.
    """
    from evaluation.tasks.e3_daily_briefing_multi import (
        _collect_day_buckets,
        _events_in_window,
        _POSITIVE_INTERACTION_TYPES,
        _NEGATIVE_INTERACTION_TYPES,
    )

    pos_buckets = _collect_day_buckets(bq, user_id, positive_only=True)
    all_buckets = _collect_day_buckets(bq, user_id, positive_only=False)
    eligible = [d for d, rows in pos_buckets.items() if len(rows) >= 1]
    if not eligible:
        return []
    eligible.sort()
    if len(eligible) >= 3:
        eligible = eligible[1:-1]   # 24h guard on both ends

    # Pick scattered days — prefer days with at least one positive event,
    # take up to n_days highest-volume.
    by_pos_volume_desc = sorted(eligible, key=lambda d: -len(pos_buckets[d]))
    picks = sorted(by_pos_volume_desc[:n_days])

    # Build a flat pool of all events across the user (for filler sampling).
    all_events: list[dict] = []
    for app in ("instagram", "facebook", "threads"):
        for e in bq._load_events(user_id, app):
            row = dict(e)
            row.setdefault("_app", app)
            all_events.append(row)
    all_events.sort(key=lambda e: int(e.get("source_timestamp") or 0))

    rng = random.Random(rng_seed or hash(user_id) % (2**31))
    DAY = 24 * 3600
    instances: list[dict] = []
    for i, day in enumerate(picks):
        dt0 = _dt.datetime.strptime(day, "%Y-%m-%d").replace(
            hour=8, minute=0, second=0, tzinfo=_dt.timezone.utc
        )
        morning_ts = int(dt0.timestamp())
        end_ts = morning_ts + DAY

        # Held-out: pick a positive engagement that day after t_test. The
        # agent's job is to rank this item at position 1 in the slate.
        pos_after = _events_in_window(pos_buckets, day, morning_ts, end_ts)
        if not pos_after:
            continue
        held_out = pos_after[0]
        held_hashtags = _hashtag_set(held_out)

        # Hard negatives: events in the user's history (any time) where
        # the user explicitly or implicitly disliked, AND whose hashtags
        # overlap with held-out's hashtags ≥ 1 token. These are
        # surface-similar but the user's behavior shows they don't want
        # them surfaced.
        neg_pool = [
            e for e in all_events
            if (e.get("source_interaction_type") or "") in _NEGATIVE_INTERACTION_TYPES
            and _hashtag_set(e) & held_hashtags
            and int(e.get("source_timestamp") or 0) < morning_ts  # only pre-t_test
        ]
        rng.shuffle(neg_pool)
        hard_negatives = neg_pool[:N_HARD_NEGATIVES]

        # If we don't have enough negative-engagement hard negatives, fall
        # back to events with overlapping hashtags but ZERO engagement
        # signal (the user didn't react to them either way — adjacent
        # but unwanted at this moment).
        if len(hard_negatives) < N_HARD_NEGATIVES:
            seen_ids = {e.get("source_object_id") for e in hard_negatives}
            seen_ids.add(held_out.get("source_object_id"))
            for e in all_events:
                if int(e.get("source_timestamp") or 0) >= morning_ts:
                    continue
                if e.get("source_object_id") in seen_ids:
                    continue
                if not (_hashtag_set(e) & held_hashtags):
                    continue
                hard_negatives.append(e)
                seen_ids.add(e.get("source_object_id"))
                if len(hard_negatives) >= N_HARD_NEGATIVES:
                    break

        if len(hard_negatives) < 3:
            # Not enough adjacent items in the user's data to make a
            # meaningful ranking task — skip this day.
            continue

        # Fillers: random pre-t_test events with NO hashtag overlap with
        # held-out (so they're not confusable adversaries — they're noise).
        used_ids = {held_out.get("source_object_id")} | {
            n.get("source_object_id") for n in hard_negatives
        }
        filler_pool = [
            e for e in all_events
            if e.get("source_object_id") not in used_ids
            and int(e.get("source_timestamp") or 0) < morning_ts
            and not (_hashtag_set(e) & held_hashtags)
        ]
        rng.shuffle(filler_pool)
        n_fillers = SLATE_SIZE - 1 - len(hard_negatives)
        fillers = filler_pool[:n_fillers]

        # Assemble the slate (held-out + hard negatives + fillers), shuffle
        # so the agent doesn't get held-out at idx=0 every time, then
        # record held_out_idx + hard_negative_idxs.
        slate_rows: list[dict] = [held_out] + hard_negatives + fillers
        order = list(range(len(slate_rows)))
        rng.shuffle(order)
        slate = [_content_summary(slate_rows[j]) for j in order]
        held_out_idx = order.index(0)
        hard_negative_idxs = [order.index(j + 1) for j in range(len(hard_negatives))]

        # User-facing query in the fixed format from workstream D.
        candidate_titles = [
            (c["title"] or c["caption"] or "<item>")[:50]
            for c in slate[:8]
        ]
        query_text = (
            "[No user query] [Recommendation system proposed candidates: "
            + "; ".join(candidate_titles) + "]"
        )

        instances.append({
            "instance_id": f"e4_day_{i}",
            "task_id": _TASK_ID,
            "entry_point": "chatbot_routed",
            "t_test": morning_ts,
            "day_index": i,
            "day_label": day,
            # Workstream D ranking instance shape:
            "candidates": slate,
            "held_out_idx": held_out_idx,
            "hard_negative_idxs": hard_negative_idxs,
            "query_text": query_text,
        })
    return instances


def e4_prompt(instance: dict, history_block: str | None) -> str:
    cands = instance.get("candidates") or []
    cand_lines = "\n".join(
        f"  [{i}] {c.get('title','')} (hashtags: {', '.join(c.get('hashtags', []))})"
        for i, c in enumerate(cands)
    )
    history = (
        f"\n## User history (time-masked to before {instance['day_label']} 08:00 UTC)\n"
        f"{history_block}\n"
        if history_block else ""
    )
    return f"""# Task: personalized social-media content recommendation

It's {instance['day_label']} 08:00 UTC. The recommendation system has
proposed {len(cands)} candidate items. Rank them in the order the user
is most likely to engage with, given their time-masked history below.

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


def compute_e4_metrics(parsed: dict, instance: dict) -> dict:
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


def run_e4_google_search(
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
    """Workstream D: deterministic ranking-metric runner. The
    google_search MCP path is gone — the agent ranks the provided
    slate using only its time-masked history."""
    from data_preparation.utils import extract_json_from_response
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget
            )
            history_tokens = stats["total_tokens"]
        prompt = e4_prompt(inst, history_block)
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
        m = compute_e4_metrics(parsed, inst)
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
