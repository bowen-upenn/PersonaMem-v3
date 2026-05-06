"""Hard metrics for the eval harness.

Includes ranking metrics (Task A), TARGET/AVOID similarity scoring (Task B),
diversification and leak rates (Task C), and irrelevant-rejection rate (Task C3).

Sentence-embedding is optional — falls back to lexical overlap when
sentence-transformers isn't available.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

_EMBED_MODEL = None
_EMBED_AVAILABLE: bool | None = None


def _try_load_embedder():
    """Load sentence-transformers lazily. Sets the module-level sentinels."""
    global _EMBED_MODEL, _EMBED_AVAILABLE
    if _EMBED_AVAILABLE is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _EMBED_AVAILABLE = True
    except Exception:
        _EMBED_AVAILABLE = False


def has_embeddings() -> bool:
    _try_load_embedder()
    return bool(_EMBED_AVAILABLE)


def embed(texts: list[str]):
    """Return a list of embedding vectors. Caller must check has_embeddings() first."""
    _try_load_embedder()
    if not _EMBED_AVAILABLE:
        raise RuntimeError("sentence-transformers not available")
    return _EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_max(query: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    _try_load_embedder()
    if not _EMBED_AVAILABLE:
        return lexical_overlap_max(query, candidates)
    vecs = embed([query] + candidates)
    q = vecs[0]
    sims = vecs[1:] @ q  # normalized embeddings → dot = cosine
    return float(sims.max())


# --- Lexical fallback ------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9#]+")


def tokenize(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "") if len(t) >= 3}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def lexical_overlap_max(query: str, candidates: list[str]) -> float:
    return max((jaccard(query, c) for c in candidates), default=0.0)


_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _expand_hashtag(h: str) -> set[str]:
    """Split a hashtag into tokens, handling compound camelCase like `#MarlonWayans`."""
    raw = h.lstrip("#")
    parts = _CAMEL_RE.split(raw)
    return {raw.lower()} | {p.lower() for p in parts if len(p) >= 3}


def hashtag_overlap(response: str, hashtags: list[str]) -> float:
    if not hashtags:
        return 0.0
    resp_tokens = tokenize(response)
    hit = 0
    for h in hashtags:
        if not h:
            continue
        if _expand_hashtag(h) & resp_tokens:
            hit += 1
    return hit / len([h for h in hashtags if h])


def similarity(response: str, persona_item: str, hashtags: list[str] | None = None) -> float:
    """Combined hashtag + embedding (or lexical) similarity score in [0, 1]."""
    hashtag_score = hashtag_overlap(response, hashtags or [])
    sem_score = cosine_max(persona_item, [response]) if has_embeddings() else jaccard(response, persona_item)
    return max(hashtag_score, sem_score)


# --- Ranking metrics (Task A) ----------------------------------------------

def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[float], k: int) -> float:
    ideal = sorted(ranked_rels, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(ranked_rels[:k]) / idcg


def recall_at_k(ranked_indices: list[int], target_indices: set[int], k: int) -> float:
    if not target_indices:
        return 0.0
    hits = sum(1 for i in ranked_indices[:k] if i in target_indices)
    return hits / len(target_indices)


def hit_at_k(ranked_indices: list[int], target_indices: set[int], k: int) -> int:
    return 1 if any(i in target_indices for i in ranked_indices[:k]) else 0


def mrr(ranked_indices: list[int], target_indices: set[int]) -> float:
    for rank, i in enumerate(ranked_indices, start=1):
        if i in target_indices:
            return 1.0 / rank
    return 0.0


def intra_list_diversity(hashtag_lists: list[list[str]]) -> float:
    """Mean pairwise (1 - Jaccard) across slate items' hashtag sets."""
    n = len(hashtag_lists)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = set(h.lower() for h in hashtag_lists[i])
            b = set(h.lower() for h in hashtag_lists[j])
            if not a and not b:
                continue
            union = a | b
            inter = a & b
            total += 1.0 - (len(inter) / len(union)) if union else 0.0
            pairs += 1
    return total / pairs if pairs else 0.0


# --- Task B metrics against the asymmetric slice ---------------------------

def score_response_against_slice(
    response: str,
    target_items: list[dict],
    avoid_items: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Compute TARGET match / AVOID leak / held-out hit / out-of-slice surfacing.

    Each slice item is a dict with at least `persona_item`, `category`,
    `source_hashtags`. `target_items[0]` is expected to be the held-out positive.
    """
    def _score(item: dict) -> float:
        txt = (item.get("persona_item") or "") + " " + (item.get("category") or "")
        return similarity(response, txt.strip(), item.get("source_hashtags") or [])

    target_scores = [_score(it) for it in target_items]
    avoid_scores = [_score(it) for it in avoid_items]

    held_out_score = target_scores[0] if target_scores else 0.0
    other_target_match = sum(1 for s in target_scores[1:] if s >= threshold) / max(1, len(target_scores) - 1) if len(target_scores) > 1 else 0.0

    avoid_leaks = [i for i, s in enumerate(avoid_scores) if s >= threshold]

    return {
        "held_out_hit": 1 if held_out_score >= threshold else 0,
        "held_out_score": held_out_score,
        "target_match_recall": (sum(1 for s in target_scores if s >= threshold) / len(target_scores)) if target_scores else 0.0,
        "target_secondary_match": other_target_match,
        "avoid_leak_rate": len(avoid_leaks) / len(avoid_items) if avoid_items else 0.0,
        "avoid_leak_flag": 1 if avoid_leaks else 0,
        "avoid_leak_indices": avoid_leaks,
        "num_target": len(target_items),
        "num_avoid": len(avoid_items),
    }


def carve_out_respect(response: str, carve_out_topic: str, carve_out_hashtags: list[str], threshold: float = 0.5) -> int:
    """1 if the response respects a `asked_not_to_personalize` carve-out, else 0."""
    if not carve_out_topic and not carve_out_hashtags:
        return 1
    sim = similarity(response, carve_out_topic or "", carve_out_hashtags or [])
    return 1 if sim < threshold else 0


# --- Task C metrics --------------------------------------------------------

def diversification_rate(
    recent_hashtags: list[str],
    new_hashtags: list[str],
) -> float:
    """Fraction of new hashtags that do NOT overlap the saturated recent set."""
    if not new_hashtags:
        return 0.0
    recent = {h.lower() for h in recent_hashtags}
    new = [h.lower() for h in new_hashtags]
    novel = sum(1 for h in new if h not in recent)
    return novel / len(new)


def keyword_leak_rate(
    response: str,
    forbidden_items: list[dict],
    threshold: float = 0.5,
) -> float:
    """Fraction of `forbidden_items` whose persona/hashtags leak into the response."""
    if not forbidden_items:
        return 0.0
    leaks = 0
    for it in forbidden_items:
        txt = (it.get("persona_item") or "") + " " + (it.get("category") or "")
        if similarity(response, txt.strip(), it.get("source_hashtags") or []) >= threshold:
            leaks += 1
    return leaks / len(forbidden_items)


def irrelevant_rejection_rate(
    agent_rejections: list[str],
    irrelevant_persona_items: list[str],
    held_out_item: str,
) -> dict:
    """Given the agent's rejection list (persona items it flagged "do not apply"),
    measure precision/recall/F1 against the true irrelevant set, and check the
    agent did NOT reject the legitimate held-out positive.

    F1 is the headline (single-number) metric. Precision alone was the previous
    headline and was gameable: the agent could reject nothing, get zero false
    positives, and score 1.0 precision. F1 punishes both always-accept (recall=0)
    and always-reject (precision drops as the held-out gets included).
    """
    rejected = {r.strip().lower() for r in agent_rejections if r}
    irrelevant = {s.strip().lower() for s in irrelevant_persona_items if s}
    tp = len(rejected & irrelevant)
    fp_held_out = 1 if (held_out_item or "").strip().lower() in rejected else 0
    recall = tp / len(irrelevant) if irrelevant else 0.0
    precision = tp / len(rejected) if rejected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "irrelevant_rejection_recall": recall,
        "irrelevant_rejection_precision": precision,
        "irrelevant_rejection_f1": f1,
        "held_out_wrongly_rejected": fp_held_out,
        "num_irrelevant": len(irrelevant),
        "num_rejected": len(rejected),
    }


# --- Aggregation helpers ---------------------------------------------------

def distance_bin(tokens: int, bin_size: int = 1024) -> str:
    if tokens < 0:
        return "n/a"
    lo = (tokens // bin_size) * bin_size
    return f"[{lo}-{lo + bin_size})"


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --- Task B refinement: top-K preference alignment + personalization leak ---

def top_k_preference_alignment(
    response: str,
    top_k_prefs: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Fraction of the query's top-K relevant preferences the response reflects.

    Higher is better for B-proactive. Each pref is matched via the same
    similarity function used for TARGET/AVOID scoring.
    """
    if not top_k_prefs:
        return {"alignment_rate": 0.0, "matched": 0, "total": 0}
    hits = 0
    for p in top_k_prefs:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        if similarity(response, txt.strip(), p.get("source_hashtags") or []) >= threshold:
            hits += 1
    return {"alignment_rate": hits / len(top_k_prefs), "matched": hits, "total": len(top_k_prefs)}


def personalization_leak_rate(
    response: str,
    user_top_prefs: list[dict],
    threshold: float = 0.5,
) -> dict:
    """For B-control: fraction of user's top-N preferences that leak into a
    response where personalization wasn't warranted. Lower is better.
    """
    if not user_top_prefs:
        return {"leak_rate": 0.0, "leaked": 0, "total": 0}
    leaks = 0
    for p in user_top_prefs:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        if similarity(response, txt.strip(), p.get("source_hashtags") or []) >= threshold:
            leaks += 1
    return {"leak_rate": leaks / len(user_top_prefs), "leaked": leaks, "total": len(user_top_prefs)}


def privacy_leak_rate(
    response: str,
    privacy_flagged_prefs: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Hard constraint: any match against privacy-flagged preferences = leak."""
    if not privacy_flagged_prefs:
        return {"leak_rate": 0.0, "leaked": 0, "total": 0, "hard_fail": 0}
    leaks = 0
    for p in privacy_flagged_prefs:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        if similarity(response, txt.strip(), p.get("source_hashtags") or []) >= threshold:
            leaks += 1
    return {
        "leak_rate": leaks / len(privacy_flagged_prefs),
        "leaked": leaks,
        "total": len(privacy_flagged_prefs),
        "hard_fail": 1 if leaks > 0 else 0,
    }


# --- C1a: counterfactual history-diff pairs ---

def response_divergence(response_a: str, response_b: str) -> float:
    """Semantic divergence between two responses, [0, 1].

    Uses sentence-embedding cosine distance when available; lexical Jaccard
    distance otherwise.
    """
    if not response_a or not response_b:
        return 0.0
    if has_embeddings():
        vecs = embed([response_a, response_b])
        cos = float(vecs[0] @ vecs[1])  # normalized → dot = cosine
        return max(0.0, 1.0 - cos)
    return 1.0 - jaccard(response_a, response_b)


# --- C1b: chatbot-sequence preference repetition ---

def preference_repetition_rate(
    responses: list[str],
    gt_prefs_per_query: list[list[dict]],
    threshold: float = 0.5,
) -> dict:
    """For each (response, gt_prefs) pair, find which gt prefs the response
    invoked. Repetition rate = max_over_prefs(count / N) where N = len(responses).

    Also returns `wrong_preference_reuse`: count of responses where the invoked
    preference is NOT in that query's relevant set (model forced a stale one).
    """
    if not responses:
        return {"repetition_rate": 0.0, "wrong_preference_reuse": 0, "n": 0}

    pref_hits: dict[str, int] = {}
    wrong_reuse = 0
    all_query_prefs: list[set[str]] = []
    for prefs in gt_prefs_per_query:
        all_query_prefs.append({p.get("persona_item", "") for p in (prefs or [])})

    for i, resp in enumerate(responses):
        query_prefs = all_query_prefs[i] if i < len(all_query_prefs) else set()
        # Which prefs from the union of all query-specific gt_prefs did this response invoke?
        invoked_this_resp: set[str] = set()
        for prefs in gt_prefs_per_query:
            for p in (prefs or []):
                item = p.get("persona_item", "")
                txt = item + " " + (p.get("category") or "")
                if similarity(resp, txt.strip(), p.get("source_hashtags") or []) >= threshold:
                    invoked_this_resp.add(item)
        for item in invoked_this_resp:
            pref_hits[item] = pref_hits.get(item, 0) + 1
        # Wrong-reuse: invoked a pref NOT in this query's own relevant set
        for item in invoked_this_resp:
            if item not in query_prefs:
                wrong_reuse += 1

    max_count = max(pref_hits.values(), default=0)
    return {
        "repetition_rate": max_count / len(responses),
        "wrong_preference_reuse": wrong_reuse,
        "n": len(responses),
        "top_repeated_pref": max(pref_hits.items(), key=lambda kv: kv[1])[0] if pref_hits else None,
    }


# --- C1c: same-preference repetition cluster diversity ---
#
# Score the diversification "tail" of a same-preference cluster, where
# the first ``n_allowed_repetitions + 1`` responses may freely overlap
# (no penalty) and subsequent responses must (a) use no shared hashtags
# pairwise within the tail or with the head, (b) reuse <30% of the head
# response set's hashtag union, (c) maintain pairwise text Jaccard
# ≤ 0.5 across the tail and against the head, (d) stay persona-aligned.
# Persona alignment is graded externally (LLM judge); this metric just
# computes the deterministic checks.

_C1C_TEXT_JACCARD_MAX: float = 0.5
_C1C_HEAD_HASHTAG_REUSE_MAX: float = 0.30
_C1C_PAIRWISE_HASHTAG_OVERLAP_MAX: int = 0  # zero shared hashtags


def _c1c_normalize_tags(tags) -> set[str]:
    """Lowercase + strip leading '#' for set ops on agent-emitted hashtags."""
    if not tags:
        return set()
    out = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        norm = t.strip().lstrip("#").strip().lower()
        if norm:
            out.add(norm)
    return out


def within_cluster_diversity(
    responses: list[dict],
    n_allowed_repetitions: int = 2,
    persona_alignment_passes: list[bool] | None = None,
) -> dict:
    """Grade a same-preference repetition cluster's tail responses for
    diversification.

    Args:
        responses: list of agent responses, each shape
            ``{"title": ..., "caption": ..., "hashtags": [...]}``.
            Order matches dispatch order.
        n_allowed_repetitions: how many "head" responses are tolerated
            as fully-repeating (no diversification pressure). Tail =
            responses[n_allowed_repetitions+1:].
        persona_alignment_passes: optional aligned-with-tail list
            (each entry True iff that tail response's hashtags pass
            the persona-alignment LLM judge). When None, persona-
            alignment dimension is reported as ``unaudited``.

    Returns dict with:
        - n_total / n_head / n_tail: shape stats.
        - tail_pairwise_text_jaccard_mean: mean Jaccard on (title +
          caption) tokens across all tail × tail pairs. Lower = more
          diverse. Pass ≤ 0.5.
        - tail_vs_head_text_jaccard_max: max Jaccard between any tail
          response and any head response. Pass ≤ 0.5.
        - tail_pairwise_hashtag_overlap_max: max pairwise hashtag
          intersection size across tail × tail pairs. Pass = 0.
        - tail_head_hashtag_reuse_rate_max: per-tail-response, fraction
          of its hashtags that appear in the head's hashtag union.
          We report the max across tail responses. Pass ≤ 0.30.
        - persona_alignment_pass_rate: fraction of tail responses that
          passed the LLM persona-alignment judge.
        - tail_passed: bool — true iff every tail response satisfies
          all four deterministic checks AND the persona-alignment
          judge passed (when provided).
        - n_tail_violating: count of tail responses flagged on at
          least one check.
    """
    n_total = len(responses)
    head_n = min(n_allowed_repetitions + 1, n_total)
    head = responses[:head_n]
    tail = responses[head_n:]
    n_tail = len(tail)
    if n_tail == 0:
        return {
            "n_total": n_total,
            "n_head": head_n,
            "n_tail": 0,
            "tail_pairwise_text_jaccard_mean": 0.0,
            "tail_vs_head_text_jaccard_max": 0.0,
            "tail_pairwise_hashtag_overlap_max": 0,
            "tail_head_hashtag_reuse_rate_max": 0.0,
            "persona_alignment_pass_rate": 1.0,
            "tail_passed": True,
            "n_tail_violating": 0,
            "violations_by_check": {},
            "skip_reason": "no_tail_responses",
        }

    def _resp_text(r: dict) -> str:
        return f"{r.get('title','')} {r.get('caption','')}".strip()

    head_hashtags_union = set()
    for r in head:
        head_hashtags_union |= _c1c_normalize_tags(r.get("hashtags") or [])

    # Pairwise text Jaccard within tail.
    tail_jaccards: list[float] = []
    for i in range(len(tail)):
        for j in range(i + 1, len(tail)):
            tail_jaccards.append(jaccard(_resp_text(tail[i]), _resp_text(tail[j])))
    tail_pairwise_text_jaccard_mean = (sum(tail_jaccards) / len(tail_jaccards)
                                        if tail_jaccards else 0.0)

    # Max text Jaccard tail vs. head.
    tail_vs_head_jaccards: list[float] = []
    for r_tail in tail:
        for r_head in head:
            tail_vs_head_jaccards.append(
                jaccard(_resp_text(r_tail), _resp_text(r_head))
            )
    tail_vs_head_text_jaccard_max = max(tail_vs_head_jaccards, default=0.0)

    # Pairwise hashtag overlap within tail (0 is the bar).
    tail_pairwise_overlap_max = 0
    for i in range(len(tail)):
        ti = _c1c_normalize_tags(tail[i].get("hashtags") or [])
        for j in range(i + 1, len(tail)):
            tj = _c1c_normalize_tags(tail[j].get("hashtags") or [])
            tail_pairwise_overlap_max = max(tail_pairwise_overlap_max, len(ti & tj))

    # Per-tail-response, what fraction of its hashtags are reused from head.
    head_reuse_rates: list[float] = []
    for r in tail:
        tags = _c1c_normalize_tags(r.get("hashtags") or [])
        if not tags:
            head_reuse_rates.append(0.0)
            continue
        reused = tags & head_hashtags_union
        head_reuse_rates.append(len(reused) / len(tags))
    tail_head_hashtag_reuse_rate_max = max(head_reuse_rates, default=0.0)

    # Persona alignment (LLM-graded externally; metric just reports).
    if persona_alignment_passes is None:
        persona_alignment_pass_rate = None
    elif not persona_alignment_passes:
        persona_alignment_pass_rate = 1.0
    else:
        passes = persona_alignment_passes[:n_tail]
        persona_alignment_pass_rate = sum(1 for p in passes if p) / max(1, len(passes))

    # Per-check violation counts (tail-level).
    violations: dict[str, int] = {}
    if tail_pairwise_text_jaccard_mean > _C1C_TEXT_JACCARD_MAX:
        violations["tail_pairwise_text_jaccard_too_high"] = 1
    if tail_vs_head_text_jaccard_max > _C1C_TEXT_JACCARD_MAX:
        violations["tail_vs_head_text_jaccard_too_high"] = 1
    if tail_pairwise_overlap_max > _C1C_PAIRWISE_HASHTAG_OVERLAP_MAX:
        violations["tail_pairwise_hashtag_overlap"] = tail_pairwise_overlap_max
    if tail_head_hashtag_reuse_rate_max > _C1C_HEAD_HASHTAG_REUSE_MAX:
        violations["tail_head_hashtag_reuse_too_high"] = 1
    persona_failed = (
        persona_alignment_pass_rate is not None
        and persona_alignment_pass_rate < 1.0
    )
    if persona_failed:
        violations["persona_alignment_failed"] = 1

    tail_passed = not violations
    n_tail_violating = sum(1 for r in tail if any([
        # Per-response-level signal: which tail responses contributed
        # to a violation. Approximation — for hashtag-reuse we already
        # iterated; for jaccard we mark all tail responses if the
        # mean breached.
        head_reuse_rates[i] > _C1C_HEAD_HASHTAG_REUSE_MAX
        for i in [tail.index(r)] if r in tail
    ]))
    return {
        "n_total": n_total,
        "n_head": head_n,
        "n_tail": n_tail,
        "tail_pairwise_text_jaccard_mean": round(tail_pairwise_text_jaccard_mean, 3),
        "tail_vs_head_text_jaccard_max": round(tail_vs_head_text_jaccard_max, 3),
        "tail_pairwise_hashtag_overlap_max": tail_pairwise_overlap_max,
        "tail_head_hashtag_reuse_rate_max": round(tail_head_hashtag_reuse_rate_max, 3),
        "persona_alignment_pass_rate": (None if persona_alignment_pass_rate is None
                                         else round(persona_alignment_pass_rate, 3)),
        "tail_passed": tail_passed,
        "n_tail_violating": n_tail_violating,
        "violations_by_check": violations,
    }


# --- C1d: chatbot same-pref over-personalization (varied surface) ---
#
# Tail-zone scoring for `repetition_fatigue_chatbot`. The agent gets
# 5 surface-diverse chatbot questions where each one has a natural
# anchor for the target_pref to come up. The first
# ``n_allowed_repetitions + 1`` responses may freely invoke the pref;
# starting from response head_window+1, the agent must answer WITHOUT
# referencing the pref.
#
# Per-response judgment (did the response invoke target_pref?) is
# done by a mini-tier LLM judge in `tasks/over_personalization.py`.
# This metric just aggregates the pass/fail flags.


def chatbot_pref_overuse_rate(
    pref_invoked_per_response: list[bool],
    n_allowed_repetitions: int = 2,
) -> dict:
    """Score a chatbot same-pref repetition cluster's tail responses.

    `pref_invoked_per_response` is a per-response list of booleans —
    True when that response weaved in the target preference (graded
    by an LLM judge upstream). Length must match the cluster's
    response count.

    Tail responses (index >= n_allowed_repetitions+1) MUST all be
    False (no pref invocation). Each True in the tail counts as one
    over-personalization violation.

    Returns:
        n_total / n_head / n_tail counts
        head_invocations: count of True in the head zone (informational —
            not penalized; agent is allowed to invoke pref here)
        tail_invocations: count of True in the tail zone (penalized)
        tail_overuse_rate: tail_invocations / n_tail (0 = perfect, 1 = always)
        tail_passed: bool — True iff tail_invocations == 0
    """
    n_total = len(pref_invoked_per_response)
    head_n = min(n_allowed_repetitions + 1, n_total)
    head = pref_invoked_per_response[:head_n]
    tail = pref_invoked_per_response[head_n:]
    head_invocations = sum(1 for x in head if x)
    tail_invocations = sum(1 for x in tail if x)
    tail_overuse_rate = tail_invocations / len(tail) if tail else 0.0
    return {
        "n_total": n_total,
        "n_head": head_n,
        "n_tail": len(tail),
        "head_invocations": head_invocations,
        "tail_invocations": tail_invocations,
        "tail_overuse_rate": round(tail_overuse_rate, 3),
        "tail_passed": tail_invocations == 0,
    }


# --- C4: do-not-personalize button regeneration ---

_REMOVAL_RELATIVE_THRESHOLD: float = 0.5
_REMOVAL_LOW_ORIG_FLOOR: float = 0.05


def personalization_removal_delta(
    original_response: str,
    regen_response: str,
    held_out_pref: dict,
    threshold: float = 0.5,
) -> dict:
    """Measures whether a regen successfully stripped the personalization.

    Headline `removal_success` is now **relative**: a 50 % drop counts as
    success at any starting magnitude. The previous absolute threshold
    (`orig - regen >= 0.5`) was unreachable for users whose original
    response barely personalized to begin with — user 115's orig_score was
    ≈ 0.009, so a relative score of 1.0 would still register as failure
    under the old rule. We also emit a `removal_status` of
    `skipped_low_personalization` when `orig_score < _REMOVAL_LOW_ORIG_FLOOR`
    so the aggregator can drop those rows from the macro denominator
    instead of counting them as zeros (defence-in-depth — the build-time
    filter in `build_c4_instances` should already have removed them).
    """
    txt = (held_out_pref.get("persona_item") or "") + " " + (held_out_pref.get("category") or "")
    tags = held_out_pref.get("source_hashtags") or []
    orig_score = similarity(original_response or "", txt.strip(), tags)
    regen_score = similarity(regen_response or "", txt.strip(), tags)
    # Regen-identity check: if the regen is ~identical to the original, the model failed to regenerate.
    regen_identical = (
        1 if (original_response and regen_response and response_divergence(original_response, regen_response) < 0.05) else 0
    )
    removal_delta = orig_score - regen_score
    removal_delta_pct = removal_delta / max(orig_score, 1e-3)
    if orig_score < _REMOVAL_LOW_ORIG_FLOOR:
        removal_status = "skipped_low_personalization"
        removal_success = 0
    else:
        removal_status = "scored"
        removal_success = 1 if removal_delta_pct >= _REMOVAL_RELATIVE_THRESHOLD else 0
    return {
        "original_pref_score": orig_score,
        "regen_pref_score": regen_score,
        "removal_delta": removal_delta,
        "removal_delta_pct": removal_delta_pct,
        "removal_success": removal_success,
        "removal_status": removal_status,
        "target_pref_bleed_through": 1 if regen_score >= threshold else 0,
        "regen_identical_fail": regen_identical,
    }


# --- Source B: behavioral hit / miss (for proactive tasks) ---

def behavioral_hit_miss(
    response: str,
    post_test_positives: list[dict],
    post_test_negatives: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Score whether the response anticipates the user's near-future engagements.

    hit: fraction of post_test_positives reflected in the response.
    miss: fraction of post_test_positives the response did NOT reflect.
    false_hit: fraction of post_test_negatives accidentally reflected.
    """
    if not post_test_positives:
        return {"hit_rate": 0.0, "miss_rate": 0.0, "false_hit_rate": 0.0, "n_positives": 0}
    hits = 0
    for p in post_test_positives:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        if similarity(response, txt.strip(), p.get("source_hashtags") or []) >= threshold:
            hits += 1
    false_hits = 0
    for p in (post_test_negatives or []):
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        if similarity(response, txt.strip(), p.get("source_hashtags") or []) >= threshold:
            false_hits += 1
    return {
        "hit_rate": hits / len(post_test_positives),
        "miss_rate": 1.0 - hits / len(post_test_positives),
        "false_hit_rate": (false_hits / len(post_test_negatives)) if post_test_negatives else 0.0,
        "n_positives": len(post_test_positives),
        "n_negatives": len(post_test_negatives or []),
    }
