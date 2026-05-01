"""Build a frozen evaluation benchmark file for a user.

Two-phase design: build-once / run-many. All randomness (slate composition,
shuffle order, Task C scenario instantiation, C1 probe selection) is resolved
here and written to `evaluation/benchmarks/{user_id}/benchmark.json`.
`run_inference.py` consumes that file and performs no runtime RNG, so:
  - two runs of the same config produce the same instances,
  - mode-A vs mode-B comparisons see identical inputs,
  - different models can be compared apples-to-apples.

Per-instance seeding: each test item derives its RNG from
`(rng_seed, source_object_id)`. Adding or removing one item does NOT cascade-
shift every other slate.

Rebuild when the underlying backend data changes — the file records a
`backend_hash` so staleness is detectable.
"""

from __future__ import annotations

# argparse/main() removed: this module is now library-only;
# `scripts/prepare_eval_data.py` owns the CLI.
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preparation import utils
from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import TestItem, build_gt_slice, load_test_items, DAY_SECONDS
from evaluation import scenarios as scenarios_mod
from evaluation import metrics as metrics_mod
from evaluation import prompts as prompts_mod
from evaluation import task_distribution as _task_dist
from evaluation.tasks import chatbot_response as cb_task

BENCHMARK_VERSION = "v2"
SOCIAL_APPS = ("instagram", "facebook", "threads")


# --- Per-instance RNG ------------------------------------------------------

def _instance_rng(global_seed: int, instance_key: str) -> random.Random:
    """Derive an independent RNG per instance — stable across item additions."""
    return random.Random(f"{global_seed}:{instance_key}")


# --- Backend hash (for staleness detection) --------------------------------

def compute_backend_hash(backend_dir: str | Path, user_id: str) -> str:
    h = hashlib.sha256()
    base = Path(backend_dir) / user_id
    for name in sorted(["profile.json", "instagram.json", "facebook.json", "threads.json", "chatbot.json"]):
        p = base / name
        if p.exists():
            with p.open("rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


# --- Task A: slate instances -----------------------------------------------

def _content_to_item(event_content: dict, hashtags: list, content_type: str) -> dict:
    return {
        "title": event_content.get("title") or "",
        "caption": event_content.get("caption") or "",
        "hashtags": hashtags,
        "content_type": content_type,
    }


def _preference_to_item(pref: dict) -> dict:
    persona = pref.get("persona_item") or ""
    return {
        "title": persona[:80],
        "caption": persona,
        "hashtags": pref.get("source_hashtags") or [],
        "content_type": "text",
    }


# Tokens to ignore when comparing caption overlap — keeps the similarity-floor
# from rejecting candidates that share only stopwords with the held-out.
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from has have in is it its of "
    "on or so that the their them they this to was were will with you your "
    "i me my we our us he she him her his hers what where when how why which "
    "than then there here some any all not no yes if just like".split()
)


def _tokens(s: str) -> set[str]:
    return {w for w in (s or "").lower().split() if w.isalpha() and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _too_similar_to_target(cand: dict, target: dict, hashtag_max: float = 0.6, caption_max: float = 0.5) -> bool:
    """Reject hard-negative candidates that look like near-duplicates of the
    held-out target — keeps the held-out as the unique best answer.
    """
    ah = {h.lower().lstrip("#") for h in (cand.get("hashtags") or [])}
    bh = {h.lower().lstrip("#") for h in (target.get("hashtags") or [])}
    if _jaccard(ah, bh) > hashtag_max:
        return True
    at = _tokens(cand.get("caption") or "")
    bt = _tokens(target.get("caption") or "")
    if at and bt and _jaccard(at, bt) > caption_max:
        return True
    return False


def _positive_engagement_items(
    bq: BackendQuery,
    user_id: str,
    exclude_ids: set,
    window_lo: int,
    window_hi: int,
) -> list[tuple[dict, str, int]]:
    """Sample positive-engagement events in a [window_lo, window_hi] timestamp range.

    Returns `(candidate_dict, source_object_id, ts)` for every event whose
    `source_interaction_type` is explicit_positive or implicit_positive,
    excluding the held-out target by `source_object_id`. Caller is
    responsible for shuffling + similarity-floor filtering + capping.
    """
    out: list[tuple[dict, str, int]] = []
    for app in APPS:
        # since_timestamp = 10**12 ⇒ no upper-bound mask, scan everything.
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=10**12):
            ts = int(e.get("source_timestamp") or 0)
            if not (window_lo <= ts <= window_hi):
                continue
            src_oid = str(e.get("source_object_id", ""))
            if src_oid in exclude_ids:
                continue
            itype = (e.get("source_interaction_type") or "").lower()
            if itype not in ("explicit_positive", "implicit_positive"):
                continue
            content = e.get("content") or {}
            hashtags = e.get("source_hashtags") or []
            content_type = content.get("content_type") or e.get("content_type") or "text"
            item = _content_to_item(content, hashtags, content_type)
            out.append((item, src_oid, ts))
    return out


def _negative_engagement_items(
    bq: BackendQuery,
    user_id: str,
    exclude_ids: set,
) -> list[tuple[dict, str, int]]:
    """Sample negative-engagement events anywhere in the timeline.

    Used to build slate hard-negatives: items whose hashtags overlap the
    held-out target enough to be confusable on the surface, but where the
    user actively *passed over* / disliked similar content. Mirrors the
    shape of `_positive_engagement_items` so the slate builder can reuse
    the same shuffle + similarity-cap logic.
    """
    out: list[tuple[dict, str, int]] = []
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=10**12):
            src_oid = str(e.get("source_object_id", ""))
            if src_oid in exclude_ids:
                continue
            itype = (e.get("source_interaction_type") or "").lower()
            if itype not in ("explicit_negative", "implicit_negative"):
                continue
            content = e.get("content") or {}
            hashtags = e.get("source_hashtags") or []
            content_type = content.get("content_type") or e.get("content_type") or "text"
            item = _content_to_item(content, hashtags, content_type)
            out.append((item, src_oid, int(e.get("source_timestamp") or 0)))
    return out


# Hashtag-Jaccard band for a hard-negative against the held-out target.
# Lower bound: must share enough hashtags to be confusable on the surface.
# Upper bound: must not exceed the duplicate-rejection threshold in
# `_too_similar_to_target` (0.6) so the held-out remains the unique best
# answer.
_HARD_NEG_J_MIN: float = 0.30
_HARD_NEG_J_MAX: float = 0.60


def build_slate_instance(test: TestItem, bq: BackendQuery, rng: random.Random) -> dict:
    candidates: list[dict] = []
    t = test.source_timestamp

    # 1x held-out positive
    held_out = _content_to_item(test.content, test.source_hashtags, test.content.get("content_type") or "text")
    held_out["_origin"] = "held_out"
    candidates.append(held_out)

    # 3x topically-irrelevant
    irrels = test.over_personalization_irrelevant[:]
    rng.shuffle(irrels)
    for p in irrels[:3]:
        c = _preference_to_item(p)
        c["_origin"] = "irrelevant"
        candidates.append(c)

    # 3x hard-negative — events the user passed over (interaction_type
    # explicit_negative / implicit_negative) whose hashtags overlap the
    # held-out target enough to be confusable on the surface (Jaccard in
    # [_HARD_NEG_J_MIN, _HARD_NEG_J_MAX]). Ranking gain stays 0; the agent
    # has to actually reason about what the user wants, not just what it
    # looks like. Replaces the previous persona-item-level "negative" tier
    # which was easy to reject by surface keyword match.
    held_out_tags = {h.lower().lstrip("#") for h in (held_out.get("hashtags") or [])}
    hard_neg_pool = _negative_engagement_items(
        bq, test.user_id, exclude_ids={test.source_object_id},
    )
    hard_neg_scored: list[tuple[float, dict]] = []
    for item, _, _ in hard_neg_pool:
        cand_tags = {h.lower().lstrip("#") for h in (item.get("hashtags") or [])}
        if not cand_tags or not held_out_tags:
            continue
        j = len(cand_tags & held_out_tags) / max(1, len(cand_tags | held_out_tags))
        if _HARD_NEG_J_MIN <= j <= _HARD_NEG_J_MAX:
            hard_neg_scored.append((j, item))
    rng.shuffle(hard_neg_scored)
    hard_neg_kept = 0
    for _, item in hard_neg_scored:
        if hard_neg_kept >= 3:
            break
        if _too_similar_to_target(item, held_out):
            continue
        item["_origin"] = "hard_negative"
        candidates.append(item)
        hard_neg_kept += 1
    # Backfill from known-disliked persona items if the user has too few
    # negative engagement events with matching hashtags (sparse-negative
    # personas). Falls back to the previous tier so the slate still hits 16.
    if hard_neg_kept < 3:
        neg_prefs = bq.get_preferences(user_id=test.user_id, since_timestamp=t, polarity="negative")
        rng.shuffle(neg_prefs)
        for p in neg_prefs:
            if hard_neg_kept >= 3:
                break
            c = _preference_to_item(p)
            c["_origin"] = "hard_negative"
            candidates.append(c)
            hard_neg_kept += 1

    # 3x past-positive (events the user already engaged with at ts < t_test).
    # Filtered for low similarity to held_out so they're plausible alternatives,
    # not near-duplicates that would muddy the unique best answer.
    SEVEN_D = 7 * 86400
    past_pool = _positive_engagement_items(
        bq, test.user_id,
        exclude_ids={test.source_object_id},
        window_lo=0, window_hi=t - 1,
    )
    rng.shuffle(past_pool)
    past_kept = 0
    for item, _, _ in past_pool:
        if past_kept >= 3:
            break
        if _too_similar_to_target(item, held_out):
            continue
        item["_origin"] = "past_positive"
        candidates.append(item)
        past_kept += 1

    # 3x future-positive (events the user WILL engage with in the next 7d, but
    # not the soonest one — that's the held-out). Same similarity floor.
    fut_pool = _positive_engagement_items(
        bq, test.user_id,
        exclude_ids={test.source_object_id},
        window_lo=t + 1, window_hi=t + SEVEN_D,
    )
    rng.shuffle(fut_pool)
    fut_kept = 0
    for item, _, _ in fut_pool:
        if fut_kept >= 3:
            break
        if _too_similar_to_target(item, held_out):
            continue
        item["_origin"] = "future_positive"
        candidates.append(item)
        fut_kept += 1

    # 3x plausible-random from unused hashtags
    used = {h.lower() for p in bq.get_preferences(user_id=test.user_id, since_timestamp=t) for h in p.get("source_hashtags", [])}
    unused_hashtags: list[str] = []
    for app in APPS:
        for e in bq.get_events(user_id=test.user_id, app=app, since_timestamp=t):
            for h in e.get("source_hashtags", []):
                if h.lower() not in used:
                    unused_hashtags.append(h)
    rng.shuffle(unused_hashtags)
    for h in unused_hashtags[:3]:
        candidates.append({
            "title": f"Trending content about {h}",
            "caption": f"A popular post mentioning {h}.",
            "hashtags": [h],
            "content_type": "text",
            "_origin": "random",
        })

    # Top up to 16 slots: prefer additional low-similarity past/future positives,
    # then fall back to generic filler if the persona doesn't have enough events
    # at all (rare — only on extremely sparse personas).
    while len(candidates) < 16:
        topup_pool = past_pool + fut_pool
        added = False
        for item, _, _ in topup_pool:
            if any(item is c for c in candidates):
                continue
            if _too_similar_to_target(item, held_out):
                continue
            item["_origin"] = "filler_lowsim"
            candidates.append(item)
            added = True
            if len(candidates) >= 16:
                break
        if not added:
            candidates.append({
                "title": "General content",
                "caption": "Unspecified item.",
                "hashtags": [],
                "content_type": "text",
                "_origin": "filler",
            })

    rng.shuffle(candidates)
    slate = []
    held_out_idx = 0
    origin_by_idx: list[str] = []
    for idx, c in enumerate(candidates):
        slate.append({
            "idx": idx,
            "app": test.app,
            "title": c["title"],
            "caption": c["caption"],
            "hashtags": c["hashtags"],
            "content_type": c["content_type"],
        })
        origin_by_idx.append(c["_origin"])
        if c["_origin"] == "held_out":
            held_out_idx = idx

    return {
        "test_id": test.source_object_id,
        "app": test.app,
        "source_timestamp": test.source_timestamp,
        "formatted_timestamp": test.formatted_timestamp,
        "query_hashtags": test.source_hashtags,
        "slate": slate,
        "held_out_idx": held_out_idx,
        "origin_by_idx": origin_by_idx,
    }


# --- Task B: proactive chatbot instances (with curation + control arm) ----

# Actions that move a query out of B (into C) — user is explicitly asking for
# or against personalization, so it no longer tests proactive capability.
_ACTION_EXCLUDE = {"asked_not_to_personalize", "asked_to_forget", "asked_to_personalize", "asked_for_recommendation"}

# Regex patterns detecting in-text personalization asks.
_EXPLICIT_ASK_RE = re.compile(
    r"\b(recommend (?:me|for me|something)|what (?:do|would) i (?:like|enjoy)|based on my|personalize for me|tailored to me)\b",
    re.IGNORECASE,
)

# Conversation-continuation pronouns that indicate the query references a prior
# assistant turn — these are instruction-following tests, not proactive tests.
_CONTINUATION_RE = re.compile(
    r"\b(that (?:last|one|answer|line|part|bit|down|thing|stuff|one|paragraph|bullet)|the (?:above|prior|previous|one you|last)|your (?:response|answer|reply))\b",
    re.IGNORECASE,
)

# Queries starting with these discourse markers are almost always follow-ups.
_CONTINUATION_STARTERS = (
    "yeah", "ok ", "okay", "good", "perfect", "great", "nice", "actually",
    "and ", "but ", "also", "so ", "hmm", "uh ", "um ", "wait", "right",
    "sure", "got it", "makes sense", "huh",
)


def _load_all_chatbot_events(backend_dir: str | Path, user_id: str) -> list[dict]:
    path = Path(backend_dir) / user_id / "chatbot.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def _candidate_from_event(event: dict, held_out_preference: dict | None = None) -> dict | None:
    """Extract a proactive-query candidate from a chatbot event.

    Per-turn-aware: when the conversation carries `embeds_pref_idx` on user
    turns (new persona-pipeline schema), pick the FIRST user turn whose
    `embeds_pref_idx` includes the held-out preference's 1-based index in
    `event["preferences"]`. Earlier turns become `prior_conversation` so the
    agent sees the actual chat lead-up to the test query.

    Legacy fallback: when no turn carries `embeds_pref_idx` (events synthesized
    before the schema change), use `interaction_format.user_message` (or the
    last user turn in the convo) — the topical-alignment guard
    `_query_aligns_with_held_out` downstream filters misaligned candidates.

    Returns None if no usable user query is present.
    """
    fmt = event.get("interaction_format") or {}
    convo = event.get("conversation") or []
    prefs = event.get("preferences") or []

    # --- Per-turn-aware path -----------------------------------------------
    # Compute the held-out preference's 1-based index inside event["preferences"].
    held_idx_in_event: int | None = None
    if held_out_preference and held_out_preference.get("persona_item"):
        held_pi = held_out_preference["persona_item"].strip()
        for i, p in enumerate(prefs, start=1):
            if (p.get("persona_item") or "").strip() == held_pi:
                held_idx_in_event = i
                break

    has_per_turn_tags = any(
        m.get("role") == "user" and isinstance(m.get("embeds_pref_idx"), list)
        for m in convo
    )

    if has_per_turn_tags and held_idx_in_event is not None:
        chosen_turn_pos: int | None = None
        for pos, m in enumerate(convo):
            if m.get("role") != "user":
                continue
            tags = m.get("embeds_pref_idx") or []
            if held_idx_in_event in tags:
                chosen_turn_pos = pos
                break
        if chosen_turn_pos is None:
            # No turn embeds the held-out pref — skip this candidate; the test
            # would be misaligned and `_query_aligns_with_held_out` would drop
            # it anyway. Better to drop early with no spurious candidate.
            return None
        user_msg = (convo[chosen_turn_pos].get("content") or "").strip()
        if not user_msg:
            return None
        prior = list(convo[:chosen_turn_pos])
        return {
            "source_object_id": str(event.get("source_object_id", "")),
            "source_timestamp": int(event.get("source_timestamp", 0)),
            "formatted_timestamp": event.get("formatted_timestamp", ""),
            "action": fmt.get("action", ""),
            "user_query": user_msg,
            "prior_conversation": prior,
            "source_hashtags": event.get("source_hashtags", []),
            "held_out_preference": held_out_preference,
            # Per-turn tag is authoritative — the LLM placed the pref in THIS
            # turn at synthesis time. Skip the token-overlap alignment guard
            # downstream (which would false-fail when the user uses casual
            # phrasing whose vocabulary doesn't overlap the persona_item text,
            # e.g. user_query "got my fantasy roster locked" vs persona_item
            # "Interested in NFL football").
            "_alignment_confirmed_by_per_turn_tag": True,
        }

    # --- Legacy fallback (no per-turn tags) -------------------------------
    user_msg = fmt.get("user_message") or ""
    if not user_msg and convo:
        for m in reversed(convo):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
    if not user_msg:
        return None
    prior: list[dict] = []
    if convo:
        for m in convo:
            if m.get("role") == "user" and (m.get("content", "") or "").strip() == user_msg.strip():
                break
            prior.append(m)
    return {
        "source_object_id": str(event.get("source_object_id", "")),
        "source_timestamp": int(event.get("source_timestamp", 0)),
        "formatted_timestamp": event.get("formatted_timestamp", ""),
        "action": fmt.get("action", ""),
        "user_query": user_msg,
        "prior_conversation": prior,
        "source_hashtags": event.get("source_hashtags", []),
        "held_out_preference": held_out_preference,
    }


def _fresh_start_ok(candidate: dict) -> bool:
    """Fresh-start filter: a *proactive* benchmark query must stand alone.

    For legacy-extracted candidates (no per-turn tags) we require the user
    query to be the very first message of a session — no prior assistant
    turns. The continuation-syntax check ("ok now what about…") still applies
    universally.

    For per-turn-tag-extracted candidates the prior_conversation is legitimate
    chat context (the LLM intentionally placed the pref mid-conversation);
    the agent-under-eval gets the same context at runtime, so the prior-
    assistant-turns rule does not apply. We still drop continuation-syntax
    queries because those phrasings aren't valid as standalone test prompts.
    """
    q = (candidate.get("user_query") or "").strip()
    if not q:
        return False
    q_low = q.lower()
    if any(q_low.startswith(s) for s in _CONTINUATION_STARTERS):
        return False
    if _CONTINUATION_RE.search(q_low):
        return False
    if candidate.get("_alignment_confirmed_by_per_turn_tag"):
        return True
    if any(m.get("role") == "assistant" for m in (candidate.get("prior_conversation") or [])):
        return False
    return True


def _proactive_filter_ok(candidate: dict) -> bool:
    """Drop candidates that explicitly ask for personalization or are carve-outs."""
    if candidate["action"] in _ACTION_EXCLUDE:
        return False
    if _EXPLICIT_ASK_RE.search(candidate["user_query"] or ""):
        return False
    return True


def _query_aligns_with_held_out(candidate: dict) -> bool:
    """Topical-alignment guardrail: a chatbot event tags ONE held-out preference
    on the whole event, but `_candidate_from_event` extracts only the LAST
    user_message. Multi-topic conversations can leave the last turn unrelated
    to the held-out pref (e.g. user opens with NFL, then asks about ring
    photography — held-out is NFL, query is rings). Skip those: they create
    nonsensical "the agent should weave NFL into a rings answer" tests.

    Pass criteria — at least one must hold:
      (a) user_query shares ≥ 1 non-stopword content token with the held-out
          persona_item text;
      (b) any candidate source_hashtag appears as a substring of user_query
          (case-insensitive), OR user_query shares ≥ 1 token with any
          source_hashtag (e.g. #nfl ↔ "nfl game tonight");
      (c) candidate has no held_out_preference (control-arm path — alignment
          irrelevant; we judge restraint, not surfacing).
    """
    # Phase L: trust the per-turn embeds_pref_idx tag when present — the LLM
    # placed the pref in THIS user turn at synthesis time, so topical
    # alignment is authoritatively confirmed even when natural-voice phrasing
    # has no token overlap with the formal persona_item text.
    if candidate.get("_alignment_confirmed_by_per_turn_tag"):
        return True
    held = candidate.get("held_out_preference") or {}
    pi = (held.get("persona_item") or "").strip()
    if not pi:
        return True  # control arm — no alignment to enforce
    q = (candidate.get("user_query") or "").strip()
    if not q:
        return False
    q_tokens = _tokens(q)
    if not q_tokens:
        return False
    pi_tokens = _tokens(pi)
    if pi_tokens & q_tokens:
        return True
    q_low = q.lower()
    for tag in (candidate.get("source_hashtags") or []):
        tag_low = tag.lstrip("#").lower()
        if not tag_low:
            continue
        if tag_low in q_low:
            return True
        if tag_low in q_tokens:
            return True
    return False


def _blind_check(query: str, judge_llm) -> dict:
    """Build-time LLM call: estimate how much personalization would help.

    Returns `{blind_score: int 0-3, generic_answer: str, reasoning: str}`.
    On failure, returns score = 2 (moderate — conservative default keeps query).
    """
    if judge_llm is None:
        return {"blind_score": 2, "generic_answer": "", "reasoning": "blind-check skipped (--skip_blind_check)"}
    prompt = prompts_mod.query_blind_check_prompt(query)
    try:
        resp = judge_llm(prompt)
        from data_preparation.utils import extract_json_from_response
        parsed = extract_json_from_response(resp) or {}
        score = int(parsed.get("personalization_value", 2) or 2)
        return {
            "blind_score": max(0, min(3, score)),
            "generic_answer": parsed.get("generic_answer", "") or "",
            "reasoning": parsed.get("reasoning", "") or "",
        }
    except Exception as exc:
        return {"blind_score": 2, "generic_answer": "", "reasoning": f"blind-check error: {exc}"}


def _dedup_candidates(cands: list[dict]) -> list[dict]:
    """Drop near-identical queries (normalized string equality; simple but effective)."""
    seen: set[str] = set()
    out = []
    for c in cands:
        key = " ".join((c["user_query"] or "").lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _dedup_user_prefs(bq: BackendQuery, user_id: str, t_test: int) -> list[dict]:
    """Collect structured, deduplicated user preferences from all app events
    before T_test. profile.json.preferences is a flat list of strings; app-event
    preferences are the structured form with persona_item / category / source_hashtags.
    Dedup key: persona_item text.
    """
    all_prefs = bq.get_preferences(user_id=user_id, since_timestamp=t_test)
    seen: set[str] = set()
    out: list[dict] = []
    for p in all_prefs:
        item = p.get("persona_item") or ""
        if not item or item in seen:
            continue
        seen.add(item)
        out.append({
            "persona_item": item,
            "category": p.get("category", ""),
            "source_hashtags": p.get("source_hashtags") or [],
            "source_app": p.get("source_app", ""),
            "source_interaction_type": p.get("source_interaction_type", ""),
        })
    return out


def _build_top_k_relevant_prefs(
    all_prefs: list[dict],
    query: str,
    query_hashtags: list[str],
    k: int = 3,
    held_out_preference: dict | None = None,
    require_topical_alignment: bool = True,
) -> list[dict]:
    """Build top-K prefs the agent could plausibly weave into a response.

    Strict topical-alignment filter (Phase L.9): every kept pref must share
    ≥ 1 source_hashtag with the held-out pref OR ≥ 1 content token with the
    user_query. Without this filter the K=5 most-engaged prefs across the
    user's whole history rise to the top regardless of query relevance — which
    is exactly the rings-query / NFL-pref bug. With it, only topically anchored
    prefs survive.

    Cap default at k=3 (down from 5) — we want the test card to surface the
    minimum necessary supporting context, not a kitchen-sink list.

    `require_topical_alignment=False` disables the filter (used for the
    distractor_reject arm, which intentionally surfaces irrelevant prefs).
    """
    if not all_prefs:
        return []
    q_tokens = metrics_mod.tokenize(query)
    q_hash = {h.lower().lstrip("#") for h in (query_hashtags or [])}
    held_hash: set[str] = set()
    held_pi_tokens: set[str] = set()
    if held_out_preference:
        held_pi = (held_out_preference.get("persona_item") or "").strip()
        held_pi_tokens = set(metrics_mod.tokenize(held_pi))
        for h in (held_out_preference.get("source_hashtags") or []) + list(query_hashtags or []):
            held_hash.add(h.lower().lstrip("#"))

    def _topically_aligned(p: dict) -> bool:
        if not require_topical_alignment:
            return True
        # Always keep the held-out preference itself (caller may re-include it)
        if held_out_preference and (p.get("persona_item") or "").strip() == \
                (held_out_preference.get("persona_item") or "").strip():
            return True
        p_hashes = {h.lower().lstrip("#") for h in (p.get("source_hashtags") or [])}
        if held_hash and (p_hashes & held_hash):
            return True
        if q_hash and (p_hashes & q_hash):
            return True
        p_text = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        p_tokens = set(metrics_mod.tokenize(p_text))
        if q_tokens and (p_tokens & set(q_tokens)):
            return True
        if held_pi_tokens and (p_tokens & held_pi_tokens):
            return True
        return False

    aligned = [p for p in all_prefs if _topically_aligned(p)]
    if not aligned:
        return []

    def score(p: dict) -> float:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        s = metrics_mod.jaccard(query, txt)
        cat = (p.get("category") or "").lower()
        if cat and any(tok in cat for tok in q_tokens):
            s += 0.5
        for h in (p.get("source_hashtags") or []):
            if h.lower().lstrip("#") in q_hash:
                s += 1.0
            if h.lower().lstrip("#") in held_hash:
                s += 1.0
        return s

    scored = sorted(aligned, key=score, reverse=True)
    top = scored[:k]
    return [
        {
            "persona_item": p["persona_item"],
            "category": p["category"],
            "source_hashtags": p["source_hashtags"],
            "relevance_score": round(score(p), 3),
        }
        for p in top
    ]


def _build_privacy_flagged_prefs(
    profile: dict,
    all_prefs: list[dict],
) -> list[dict]:
    hidden = (profile or {}).get("hidden_personas", []) or []
    privacy_types = {"intimate_interest", "covert_concern", "compensatory_need"}
    sensitive_tags: set[str] = set()
    sensitive_cats: set[str] = set()
    for h in hidden:
        if (h.get("privacy_ratio") or 0) > 0.7 or h.get("type") in privacy_types:
            for tag in (h.get("evidence_hashtags") or []):
                sensitive_tags.add(tag.lower().lstrip("#"))
            lbl = (h.get("label") or "").lower()
            if lbl:
                sensitive_cats.add(lbl)
    out = []
    for p in all_prefs:
        p_tags = {h.lower().lstrip("#") for h in (p.get("source_hashtags") or [])}
        p_cat = (p.get("category") or "").lower()
        if p_tags & sensitive_tags or p_cat in sensitive_cats:
            out.append({
                "persona_item": p["persona_item"],
                "category": p["category"],
                "source_hashtags": p["source_hashtags"],
            })
    return out


def _build_post_test_window(bq: BackendQuery, user_id: str, t_test: int, window_hours: int = 48) -> dict:
    """Source B — events in [T_test, T_test + window]. The agent never sees this.

    Loads raw events (bypassing time-mask) since this is for scoring only.
    """
    lo = t_test
    hi = t_test + window_hours * 3600
    base = Path(bq.base) / user_id
    pos_prefs: list[dict] = []
    neg_prefs: list[dict] = []
    engagements: list[dict] = []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            engagements.append({
                "event_id": str(e.get("source_object_id", "")),
                "app": app,
                "source_timestamp": ts,
                "source_hashtags": e.get("source_hashtags", []),
                "source_interaction_type": it,
                "persona_items": [p.get("persona_item") for p in (e.get("preferences") or []) if p.get("persona_item")],
            })
            for pref in (e.get("preferences") or []):
                item = {
                    "persona_item": pref.get("persona_item"),
                    "category": pref.get("category"),
                    "source_hashtags": e.get("source_hashtags", []),
                }
                if "positive" in it:
                    pos_prefs.append(item)
                elif "negative" in it:
                    neg_prefs.append(item)
    return {
        "window_hours": window_hours,
        "post_test_engagements": engagements,
        "post_test_positives": pos_prefs,
        "post_test_negatives": neg_prefs,
    }


def build_task_b_arms(
    backend_dir: str | Path,
    bq: BackendQuery,
    user_id: str,
    test_items: list[TestItem],
    blind_check_llm=None,
    blind_check_limit: int | None = None,
) -> dict:
    """Build the two B arms: proactive + control.

    Pulls from ALL chatbot events (not just test events), applies the 4-stage
    filter, labels by arm. Each instance carries same-day slice + top-K prefs
    + privacy-flagged slice + post-T_test Source B.
    """
    profile = bq.get_full_profile(user_id)
    all_events = _load_all_chatbot_events(backend_dir, user_id)

    # Map source_object_id → held-out preference for test events (used when
    # the candidate corresponds to a test event, to preserve continuity with v1).
    test_index: dict[str, dict] = {}
    for t in test_items:
        if t.app == "chatbot":
            test_index[t.source_object_id] = {
                "persona_item": t.preference.get("persona_item"),
                "category": t.preference.get("category"),
            }

    # Stage 1: extract raw candidates.
    candidates = []
    for e in all_events:
        held_out = test_index.get(str(e.get("source_object_id", "")))
        c = _candidate_from_event(e, held_out_preference=held_out)
        if c is not None:
            candidates.append(c)

    # Stage 2: fresh-start filter.
    candidates = [c for c in candidates if _fresh_start_ok(c)]
    # Stage 3: proactive filter.
    candidates = [c for c in candidates if _proactive_filter_ok(c)]
    # Stage 3.5: held-out / user_query topical-alignment guardrail.
    # Multi-turn chatbot events tag ONE pref on the whole convo; the last
    # user_message can be on a different topic. Skip misaligned candidates.
    candidates = [c for c in candidates if _query_aligns_with_held_out(c)]
    # Stage 4: dedup.
    candidates = _dedup_candidates(candidates)

    # Stage 5: blind-check → score each, split into proactive vs control.
    # Parallelized across candidates (default 16 workers) — was sequential
    # which made the blind-check stage the dominant build-time bottleneck.
    if blind_check_limit is not None:
        candidates = candidates[:blind_check_limit]
    proactive: list[dict] = []
    control: list[dict] = []
    if blind_check_llm is not None and candidates:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score_one(c: dict) -> tuple[dict, dict]:
            return c, _blind_check(c["user_query"], blind_check_llm)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_score_one, c) for c in candidates]
            for fut in as_completed(futures):
                c, bc = fut.result()
                c["blind_check_score"] = bc["blind_score"]
                c["blind_check_generic_answer"] = bc["generic_answer"]
                if bc["blind_score"] >= 2:
                    proactive.append(c)
                else:
                    control.append(c)
    else:
        # No client (--skip_blind_check): default every candidate to the
        # proactive arm. Loses the natural control split; restraint coverage
        # is still provided by the dedicated I.2/I.3/J.4/J.5 arms downstream.
        for c in candidates:
            c["blind_check_score"] = 2
            c["blind_check_generic_answer"] = ""
            proactive.append(c)

    # Demote unanchored proactives: candidates without a held-out preference
    # cannot be scored against a target, so route them to the control arm
    # where they exercise restraint against `top_k_relevant_prefs` instead.
    # This happens whenever the source event isn't in `test_index` (only the
    # R8 selector's per-app top-N events get held-out preferences attached).
    demoted = [c for c in proactive if not (c.get("held_out_preference") or {}).get("persona_item")]
    if demoted:
        proactive = [c for c in proactive if (c.get("held_out_preference") or {}).get("persona_item")]
        control.extend(demoted)

    # Fallback: if control arm is empty, grab the 3 lowest-scoring candidates.
    if not control and len(candidates) >= 3:
        control_picks = sorted(candidates, key=lambda x: x["blind_check_score"])[:3]
        control = control_picks
        proactive_ids = {c["source_object_id"] for c in control_picks}
        proactive = [c for c in proactive if c["source_object_id"] not in proactive_ids]

    # Enrich instances with ground truth.
    #
    # Phase L.10 — minimal GT. The earlier code emitted a `gt_slice` (full
    # same-day positives + negatives) and `privacy_flagged_prefs` on EVERY
    # arm, which produced redundant test cards and let irrelevant prefs into
    # the proactive metric. New rules:
    #   - `gt_slice.target` and `gt_slice.avoid` are dropped from the emitted
    #     instance (gt_slice carries only the t_test/window metadata kept for
    #     run_task_b's score_response_against_slice — which now scores against
    #     `held_out + top_k_relevant_prefs` as the natural target set).
    #   - `privacy_flagged_prefs` is emitted ONLY for arms that score against
    #     it (currently `distractor_reject` via the chatbot_response runner).
    #     Other arms get an empty list — the privacy_leak metric trivially
    #     returns 0 / no-fail in that case.
    def _finalize(c: dict, arm: str) -> dict:
        if arm in ("proactive", "contradiction"):
            assert (c.get("held_out_preference") or {}).get("persona_item"), (
                f"arm={arm!r} requires non-empty held_out_preference "
                f"(test_id={c['source_object_id']})"
            )
        t_test = c["source_timestamp"]
        all_prefs = _dedup_user_prefs(bq, user_id, t_test)
        # Phase L.9: strict topical-alignment filter on top-K (cap k=3).
        # Control arm has no held_out, so only query-token alignment applies.
        top_k = _build_top_k_relevant_prefs(
            all_prefs, c["user_query"], c["source_hashtags"],
            k=3, held_out_preference=c.get("held_out_preference"),
            require_topical_alignment=True,
        )
        # Build target list = held_out + top_k (deduplicated). This becomes
        # the `gt_slice.target` consumed by score_response_against_slice.
        target_set: list[dict] = []
        seen_pi: set[str] = set()
        held = c.get("held_out_preference") or {}
        if held.get("persona_item"):
            target_set.append({
                "persona_item": held["persona_item"],
                "category": held.get("category", ""),
                "source_hashtags": c.get("source_hashtags", []),
                "polarity": "positive",
            })
            seen_pi.add(held["persona_item"])
        for tk in top_k:
            pi = tk.get("persona_item")
            if pi and pi not in seen_pi:
                target_set.append({
                    "persona_item": pi,
                    "category": tk.get("category", ""),
                    "source_hashtags": tk.get("source_hashtags", []),
                    "polarity": "positive",
                })
                seen_pi.add(pi)
        gt_slice = {
            "t_test": t_test,
            "window_seconds": DAY_SECONDS,
            "target": target_set,
            "avoid": [],
        }
        # Privacy-flagged is meaningful only for the distractor_reject arm
        # downstream — for other arms it produced misleading "must-not-surface"
        # entries on otherwise-benign prefs (e.g. "values long-term romantic
        # commitment" got flagged as privacy-sensitive when it isn't). Default
        # empty; the dedicated build_c3_instance handles its own pool.
        privacy_flagged: list[dict] = []
        source_b = _build_post_test_window(bq, user_id, t_test)
        return {
            "test_id": c["source_object_id"],
            "arm": arm,
            "source_timestamp": t_test,
            "formatted_timestamp": c["formatted_timestamp"],
            "user_query": c["user_query"],
            "prior_conversation": c["prior_conversation"],
            "action": c["action"],
            "source_hashtags": c["source_hashtags"],
            "held_out_preference": c.get("held_out_preference"),
            "blind_check_score": c["blind_check_score"],
            "blind_check_generic_answer": c["blind_check_generic_answer"],
            "gt_slice": gt_slice,
            "top_k_relevant_prefs": top_k,
            "privacy_flagged_prefs": privacy_flagged,
            "post_test_window": source_b,
        }

    # Phase I.2: Adversarial restraint probes — synthesize 4-6 chatbot
    # questions that are deliberately TANGENT to or ANTI- the user's
    # preferences. These join the existing control arm and exercise
    # over-personalization in scenarios where a model that just defaults
    # to "don't volunteer preferences" gets caught (because the question
    # tempts it).
    adversarial = build_chatbot_restraint_adversarial(bq, user_id, profile, base_dir=backend_dir)

    # Phase J.4: persona-internal contradiction probes — find canonicals where
    # the user's stance has flipped over time; ask about the topic; agent
    # must surface the CURRENT (later) stance, not the old one.
    contradictions = build_persona_contradiction_probes(bq, user_id, profile)

    # Phase J.5: stale-vs-fresh probes — short-term prefs past their
    # expected_stop_ts; agent must NOT surface them as if still active.
    stale = build_stale_vs_fresh_probes(bq, user_id)

    return {
        "chatbot_proactive_personalization": (
            [_finalize(c, "proactive") for c in proactive]
            + [_finalize(c, "contradiction") for c in contradictions]
        ),
        "over_personalization_chatbot_text": (
            [_finalize(c, "control") for c in control]
            + [_finalize(c, "adversarial") for c in adversarial]
            + [_finalize(c, "stale") for c in stale]
        ),
    }


def build_persona_contradiction_probes(bq: BackendQuery, user_id: str, profile: dict) -> list[dict]:
    """Phase J.4: probes where the user's recent activity contradicts an
    earlier preference. Tests which signal the agent prioritizes.

    Walk all events; find preferences whose update_history contains a
    `contradicted` entry (the persona pipeline's Step 7 cross-polarity gate
    marks these). For each, build an open-ended chatbot probe that asks
    about the topic. The agent should prioritize the LATER (current) stance
    over the OLD (now-flipped) one — surfacing the old stance is wrong.

    Goes into the chatbot_proactive_personalization bucket since the agent
    IS supposed to personalize, just with the correct (current) stance.
    Routed through chatbot_response.run_task_b like the other chatbot arms.
    """
    # Find canonicals with contradicted history entries. Important: read the
    # RAW per-app JSONs here, NOT via bq.get_events — `update_history` is in
    # `_LEAK_FIELDS_PREF` and gets stripped by the agent-facing read path.
    base = Path(getattr(bq, "base", "backend")) / user_id
    seen: set = set()
    contradicted_prefs: list[dict] = []
    for app in APPS:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        try:
            evs = json.loads(path.read_text())
        except Exception:
            continue
        for e in evs:
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                pi = pref.get("persona_item") or ""
                if not pi or pi in seen:
                    continue
                hist = pref.get("update_history") or []
                # Look for a "contradicted" entry where the OPPOSING side was
                # suppressed or marked stance-flipped — these are the cases
                # where the user's stance HAS changed.
                contras = [h for h in hist if h.get("update_type") == "contradicted"]
                if not contras:
                    continue
                seen.add(pi)
                contradicted_prefs.append({
                    "persona_item": pi,
                    "category": pref.get("category"),
                    "hashtags": pref.get("source_hashtags") or e.get("source_hashtags") or [],
                    "current_stance_ts": int(e.get("source_timestamp") or 0),
                    "old_stance": (contras[0].get("preference") or "")[:100],
                })
    contradicted_prefs.sort(key=lambda p: p["current_stance_ts"], reverse=True)
    if not contradicted_prefs:
        return []

    # t_test should be AFTER the most-recent stance flip so the agent should
    # surface the new stance.
    latest_ts = max(p["current_stance_ts"] for p in contradicted_prefs)
    formatted = utils.unix_to_formatted(latest_ts) if hasattr(utils, "unix_to_formatted") else ""

    out: list[dict] = []
    # Templates: ask about the TOPIC of the flipped pref. Different framings.
    QUERY_TEMPLATES = [
        "What's a good {topic} to look at this week?",
        "Anything in {topic} you'd flag for me right now?",
        "How do you feel about {topic} these days?",
    ]
    for i, p in enumerate(contradicted_prefs[:3]):
        topic = (p.get("category") or "").lower()
        if not topic:
            # Fall back to a hashtag if no category
            tags = p.get("hashtags") or []
            topic = tags[0].lstrip("#").lower() if tags else "this topic"
        q = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)].format(topic=topic)
        out.append({
            "source_object_id": f"contradiction_{user_id}_{i:02d}",
            "source_timestamp": latest_ts - (60 * (i + 1)),
            "formatted_timestamp": formatted,
            "user_query": q,
            "prior_conversation": [],
            "action": "asked_chatbot",
            "source_hashtags": p["hashtags"][:3],
            # Held-out IS the current stance — that's what the agent should align to
            "held_out_preference": {"persona_item": p["persona_item"], "category": p["category"]},
            "blind_check_score": None,
            "blind_check_generic_answer": None,
            "_contradiction_probe": True,
            "_old_stance": p["old_stance"],
        })
    return out


def build_stale_vs_fresh_probes(bq: BackendQuery, user_id: str) -> list[dict]:
    """Phase J.5: probes where a short-term preference has expired.

    Walk all events; find prefs with `time_horizon == "short_term"` and a
    `stop_condition.expected_stop_ts` in the past. Build chatbot probes
    where t_test > expected_stop_ts. The agent should NOT surface the now-
    expired preference (e.g., asking about a vacation that already ended).

    Goes into chatbot_proactive_personalization bucket with the stale pref
    as the "do-not-surface" item. Graded by leak_rate against that single pref.
    """
    # Read RAW JSON to access time_horizon / stop_condition (likely stripped
    # by the agent-facing read path; safe to access at build time).
    base = Path(getattr(bq, "base", "backend")) / user_id
    seen: set = set()
    stale_prefs: list[dict] = []
    for app in APPS:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        try:
            evs = json.loads(path.read_text())
        except Exception:
            continue
        for e in evs:
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                if pref.get("time_horizon") != "short_term":
                    continue
                pi = pref.get("persona_item") or ""
                stop_cond = pref.get("stop_condition") or {}
                stop_ts = stop_cond.get("expected_stop_ts")
                if not pi or pi in seen or not stop_ts:
                    continue
                seen.add(pi)
                stale_prefs.append({
                    "persona_item": pi,
                    "category": pref.get("category"),
                    "hashtags": pref.get("source_hashtags") or e.get("source_hashtags") or [],
                    "stop_ts": int(stop_ts),
                    "stop_description": stop_cond.get("description", ""),
                })
    if not stale_prefs:
        return []

    out: list[dict] = []
    DAY = 86400
    QUERY_TEMPLATES = [
        "What should I be looking at right now in {topic}?",
        "Got any {topic} suggestions for the next few days?",
        "Anything new in {topic} I should care about?",
    ]
    for i, p in enumerate(stale_prefs[:3]):
        # t_test = stop_ts + 1 day — the pref is now expired.
        t_test = p["stop_ts"] + DAY
        topic = (p.get("category") or "").lower()
        if not topic:
            tags = p.get("hashtags") or []
            topic = tags[0].lstrip("#").lower() if tags else "current topics"
        q = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)].format(topic=topic)
        formatted = utils.unix_to_formatted(t_test) if hasattr(utils, "unix_to_formatted") else ""
        out.append({
            "source_object_id": f"stale_{user_id}_{i:02d}",
            "source_timestamp": t_test,
            "formatted_timestamp": formatted,
            "user_query": q,
            "prior_conversation": [],
            "action": "asked_chatbot",
            "source_hashtags": p["hashtags"][:3],
            "held_out_preference": None,  # No held-out — there's no new stance, just absence
            "blind_check_score": None,
            "blind_check_generic_answer": None,
            "_stale_probe": True,
            "_stale_pref": p["persona_item"],
            "_expected_stop_ts": p["stop_ts"],
        })
    return out


def build_chatbot_restraint_adversarial(bq: BackendQuery, user_id: str, profile: dict,
                                          base_dir: str = "backend") -> list[dict]:
    """Synthesize adversarial restraint probes (Phase I.2).

    Two archetypes per persona:
      A) TOPIC-TANGENT: question contains a topic token from one of the user's
         POSITIVE preferences but asks about a generic adjacent thing. The
         agent must NOT volunteer "you like X-related content" on its own.
         Example for a boxing fan: "what's a good cardio for someone who
         hates running?" — boxing is a tempting answer that would over-
         personalize.
      B) ANTI-PREFERENCE INVITATION: question explicitly invites the agent
         to share a preference on a topic the user is NOT into. Agent must
         NOT pretend to like it. Example for an anti-NFL user: "what's your
         favorite NFL team?" — agent should deflect, not invent a fake fave.

    Both archetypes count as "over-personalization on irrelevant preferences"
    when the agent volunteers persona context. The returned candidates have
    the same shape as control-arm chatbot_response candidates so they can
    flow through `_finalize(c, "adversarial")` and the existing runner.
    """
    # Pick the latest event timestamp as t_test for adversarial instances
    # (so the agent has the user's full history available, just like the
    # control arm probes that use real chatbot turns).
    latest_ts = 0
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=10**12):
            ts = int(e.get("source_timestamp") or 0)
            if ts > latest_ts:
                latest_ts = ts
    if latest_ts == 0:
        return []

    formatted = utils.unix_to_formatted(latest_ts) if hasattr(utils, "unix_to_formatted") else ""

    # Collect top-3 positive categories + top-3 negative persona-items
    from collections import Counter
    pos_categories: Counter = Counter()
    neg_items: list[str] = []
    seen_neg: set = set()
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=10**12):
            itype = (e.get("source_interaction_type") or "").lower()
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                cat = pref.get("category") or ""
                pi = pref.get("persona_item") or ""
                if not cat or not pi:
                    continue
                if "positive" in itype:
                    pos_categories[cat] += 1
                elif "negative" in itype and pi not in seen_neg:
                    seen_neg.add(pi); neg_items.append(pi)
    top_cats = [c for c, _ in pos_categories.most_common(3)]

    out: list[dict] = []
    next_idx = 0

    # Archetype A: tangent probe per top positive category
    tangent_templates = {
        # Each template: question, hashtags-it-implies (used to tempt the agent)
        "boxing": ("good cardio that isn't running on a treadmill?", ["#boxing", "#fitness"]),
        "nfl": ("any sunday-afternoon background tv that doesn't need focus?", ["#nfl", "#sports"]),
        "comedy": ("something quick that'll make me laugh on my coffee break?", ["#comedy", "#humor"]),
        "fitness": ("how do you keep a morning routine under 30 minutes?", ["#fitness", "#workout"]),
        "music": ("background sounds for focusing while reading?", ["#music"]),
        "cooking": ("easiest one-pan dinner I can make tonight?", ["#cooking", "#recipes"]),
        "travel": ("any tips for better airport coffee?", ["#travel"]),
        "fashion": ("how do you pack lighter for a 3-day trip?", ["#fashion", "#style"]),
    }
    for cat in top_cats[:3]:
        # Find the closest template by lowercase substring match
        cat_low = cat.lower()
        match_key = next((k for k in tangent_templates if k in cat_low), None)
        if match_key:
            q, tags = tangent_templates[match_key]
        else:
            q, tags = ("what's a good way to spend a free 20 minutes?", [])
        out.append({
            "source_object_id": f"adv_tangent_{user_id}_{next_idx:02d}",
            "source_timestamp": latest_ts - (60 * (next_idx + 1)),
            "formatted_timestamp": formatted,
            "user_query": q,
            "prior_conversation": [],
            "action": "asked_chatbot",
            "source_hashtags": tags,
            "held_out_preference": None,
            "blind_check_score": None,
            "blind_check_generic_answer": None,
            "_adversarial_kind": f"tangent_on_{cat}",
        })
        next_idx += 1

    # Archetype B: anti-preference invitation per top negative item
    anti_templates = [
        "what's your favorite kind of {}?",
        "tell me what you love about {}.",
        "which {} do you follow these days?",
        "recommend me some {} I'd really enjoy.",
    ]
    # Strip negation-prefixes so "Not interested in NFL" → "NFL"
    NEG_PREFIXES = (
        "not interested in ", "doesn't engage with ", "does not engage with ",
        "avoids ", "dislikes ", "hates ", "no interest in ", "uninterested in ",
        "rejects ",
    )
    for i, neg_item in enumerate(neg_items[:3]):
        topic = neg_item.split(".")[0].strip().rstrip(",;:")
        topic_low = topic.lower()
        for pre in NEG_PREFIXES:
            if topic_low.startswith(pre):
                topic = topic[len(pre):]
                break
        # Trim to a clean noun phrase: max 5 words, drop leading articles
        topic = topic.lstrip(",.; ").strip()
        words = topic.split()
        # Cut off at the first comma if present (commas usually start a
        # qualifying clause that wrecks the question grammar)
        if "," in topic:
            topic = topic.split(",")[0].strip()
            words = topic.split()
        if len(words) > 5:
            topic = " ".join(words[:5])
        if not topic:
            continue
        q = anti_templates[i % len(anti_templates)].format(topic)
        out.append({
            "source_object_id": f"adv_anti_{user_id}_{next_idx:02d}",
            "source_timestamp": latest_ts - (60 * (next_idx + 1)),
            "formatted_timestamp": formatted,
            "user_query": q,
            "prior_conversation": [],
            "action": "asked_chatbot",
            "source_hashtags": [],
            "held_out_preference": None,
            "blind_check_score": None,
            "blind_check_generic_answer": None,
            "_adversarial_kind": f"anti_pref:{topic[:40]}",
        })
        next_idx += 1

    return out


def _build_gt_slice_for_candidate(
    bq: BackendQuery,
    user_id: str,
    t_test: int,
    held_out_preference: dict | None,
    source_hashtags: list[str],
) -> dict:
    """Same-day TARGET/AVOID slice anchored on an arbitrary chatbot candidate
    (not necessarily a test event). Mirrors build_gt_slice but takes looser inputs.
    """
    lo, hi = t_test - DAY_SECONDS, t_test + DAY_SECONDS
    base = Path(bq.base) / user_id
    target: list[dict] = []
    avoid: list[dict] = []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            for pref in (e.get("preferences") or []):
                item = {
                    "persona_item": pref.get("persona_item"),
                    "category": pref.get("category"),
                    "source_hashtags": e.get("source_hashtags", []),
                    "polarity": "positive" if "positive" in it else "negative" if "negative" in it else "other",
                    "source_app": app,
                    "source_timestamp": ts,
                    "source_interaction_type": it,
                }
                if item["polarity"] == "positive":
                    target.append(item)
                elif item["polarity"] == "negative":
                    avoid.append(item)

    if held_out_preference and held_out_preference.get("persona_item"):
        held = {
            "persona_item": held_out_preference.get("persona_item"),
            "category": held_out_preference.get("category"),
            "source_hashtags": source_hashtags,
            "polarity": "positive",
            "source_app": "chatbot",
            "source_timestamp": t_test,
            "is_held_out": True,
        }
        seen = {held["persona_item"]}
        target_dedup = [held]
        for it in target:
            if it["persona_item"] not in seen:
                target_dedup.append(it)
                seen.add(it["persona_item"])
        target = target_dedup

    return {
        "t_test": t_test,
        "window_seconds": DAY_SECONDS,
        "target": target,
        "avoid": avoid,
    }


# --- Legacy Task B builder (kept for schema back-compat; unused in v2) ------

def build_chatbot_instance(bq: BackendQuery, test: TestItem) -> dict | None:
    user_query, prior = cb_task._extract_query_and_prior(test)
    if not user_query:
        return None
    action = (test.interaction_format or {}).get("action", "")
    gt_slice = build_gt_slice(bq, test)
    return {
        "test_id": test.source_object_id,
        "source_timestamp": test.source_timestamp,
        "formatted_timestamp": test.formatted_timestamp,
        "user_query": user_query,
        "prior_conversation": prior,
        "polarity": test.polarity,
        "action": action,
        "held_out_preference": {
            "persona_item": test.preference.get("persona_item"),
            "category": test.preference.get("category"),
        },
        "source_hashtags": test.source_hashtags,
        "gt_slice": gt_slice,
    }


# --- Task C1a: counterfactual history-diff pairs ---------------------------

def build_c1a_pairs(
    bq: BackendQuery,
    user_id: str,
    test_items: list[TestItem],
    max_pairs: int = 8,
    window_hours: int = 24,
    min_diff_events: int = 3,
) -> list[dict]:
    """Find pairs of (T_early, T_late) such that events in [T_early, T_late]
    include ≥ min_diff_events, and the diff's hashtag center differs from
    the user's long-term dominant category (shifts topical interest).

    Each pair freezes:
    - `t_early`, `t_late`, `target_app`
    - `diff_events`: summaries of the events added between the two moments
    - `shared_context_size`: event count at T_early (rough indicator)
    - `dominant_category_pre`: user's top category before T_early
    - `shift_category`: dominant category of the diff events

    Returns up to `max_pairs` pairs spanning diverse (app, category) slots.
    """
    if not test_items:
        return []
    # Anchor candidate moments at social-app test item timestamps (we already
    # know the user engages meaningfully at these times).
    anchors = sorted(
        {(t.app, t.source_timestamp) for t in test_items if t.app in SOCIAL_APPS},
        key=lambda x: x[1],
    )
    if len(anchors) < 2:
        return []

    # Dominant category = most-frequent positive preference category across history.
    base = Path(bq.base) / user_id
    cat_counts: dict[str, int] = {}
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            if "positive" not in e.get("source_interaction_type", ""):
                continue
            for pref in (e.get("preferences") or []):
                c = pref.get("category")
                if c:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
    dominant_pre = max(cat_counts.items(), key=lambda kv: kv[1])[0] if cat_counts else None

    pairs: list[dict] = []
    for i, (app_early, t_early) in enumerate(anchors[:-1]):
        for app_late, t_late in anchors[i + 1:]:
            if t_late - t_early > window_hours * 3600 or t_late - t_early <= 0:
                continue
            # Collect diff events: events in (t_early, t_late] for any app.
            diff_events: list[dict] = []
            for app in APPS:
                p = base / f"{app}.json"
                if not p.exists():
                    continue
                with p.open() as f:
                    events = json.load(f)
                for e in events:
                    ts = int(e.get("source_timestamp", 0))
                    if t_early < ts <= t_late:
                        diff_events.append({
                            "event_id": str(e.get("source_object_id", "")),
                            "app": app,
                            "t": ts,
                            "hashtags": e.get("source_hashtags", []),
                            "categories": [pref.get("category") for pref in (e.get("preferences") or []) if pref.get("category")],
                        })
            if len(diff_events) < min_diff_events:
                continue
            # Shift-check: does the diff introduce a category that's not dominant_pre?
            diff_cats = [c for e in diff_events for c in e["categories"]]
            non_dominant_diff = [c for c in diff_cats if c != dominant_pre]
            if not non_dominant_diff:
                continue
            shift_category = max(set(non_dominant_diff), key=non_dominant_diff.count)
            pair_id = f"{user_id}_c1a_{t_early}_{t_late}"
            pairs.append({
                "pair_id": pair_id,
                "target_app": app_late,
                "t_early": t_early,
                "t_late": t_late,
                "diff_events": diff_events[:15],
                "dominant_category_pre": dominant_pre,
                "shift_category": shift_category,
                "query": f"Recommend something on {app_late} this user would engage with next.",
            })
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


# --- Task C1b: chatbot-sequence preference repetition ----------------------

def build_c1b_sequence(
    b_proactive_instances: list[dict],
    max_seq_len: int = 5,
    min_distinct_categories: int = 3,
    max_sequences: int = 6,
) -> list[dict]:
    """Assemble one or more sequences from diverse-topic B-proactive queries.

    Each sequence requires queries spanning ≥ min_distinct_categories distinct
    top-1-preference categories. We now emit up to ``max_sequences`` distinct
    sequences by rotating which instance per category is picked — this fills
    the floor for repetition_fatigue_sequences (was 1 per user, now 6).
    """
    if not b_proactive_instances:
        return []
    by_cat: dict[str, list[dict]] = {}
    for inst in b_proactive_instances:
        top = inst.get("top_k_relevant_prefs") or []
        if not top:
            continue
        cat = top[0].get("category") or "uncategorized"
        by_cat.setdefault(cat, []).append(inst)

    if len(by_cat) < min_distinct_categories:
        return []

    # Sort each bucket by timestamp so ``rotation`` k picks the k-th instance
    # within each category (cycling). Each rotation produces a sequence with
    # different concrete user_queries but the same category coverage.
    for cat in by_cat:
        by_cat[cat] = sorted(by_cat[cat], key=lambda x: x.get("source_timestamp", 0))

    out: list[dict] = []
    for k in range(max_sequences):
        picks: list[dict] = []
        for cat, insts in list(by_cat.items())[:max_seq_len]:
            picks.append(insts[k % len(insts)])
        picks.sort(key=lambda x: x.get("source_timestamp", 0))
        if len({p["top_k_relevant_prefs"][0]["category"]
                for p in picks if p.get("top_k_relevant_prefs")}) < min_distinct_categories:
            continue
        # Skip if this rotation is identical to a previous one (small users
        # with few instances per category will collapse into duplicates).
        sig = tuple(p["test_id"] for p in picks)
        if any(tuple(q["source_test_id"] for q in s["queries"]) == sig for s in out):
            continue
        out.append({
            "sequence_id": f"c1b_seq_{k}",
            "queries": [
                {
                    "source_test_id": p["test_id"],
                    "source_timestamp": p["source_timestamp"],
                    "user_query": p["user_query"],
                    "top_k_relevant_prefs": p["top_k_relevant_prefs"],
                }
                for p in picks
            ],
        })
    return out


# --- Task C4: do-not-personalize button regeneration ----------------------

def build_c4_instances(b_proactive_instances: list[dict]) -> list[dict]:
    """Convert each B-proactive instance into a two-turn C4 regen probe.

    At eval time, the driver first gets the B-proactive response (or uses a
    held-out "original personalized response" field if we've cached one), then
    sends the button-click regen prompt and scores the regen.

    **Build-time filter**: only emit instances whose held-out preference shares
    at least one hashtag with the candidate query. Without surface-level
    overlap the model has no obvious cue to personalize on, so the original
    response barely mentions the preference (`orig_score` ~ 0) and the
    "removal" metric becomes degenerate — we observed orig_score ≈ 0.009
    across all 5 rows of user 115 before this filter was added. Defensive:
    the runtime metric also emits a `skipped_low_personalization` status
    when orig_score falls below 0.05.
    """
    out = []
    skipped_no_overlap = 0
    skipped_no_pref = 0
    for b in b_proactive_instances:
        held = b.get("held_out_preference") or {}
        if not held.get("persona_item"):
            skipped_no_pref += 1
            continue
        held_tags = {h.lower().lstrip("#") for h in (held.get("source_hashtags") or []) if h}
        query_tags = {h.lower().lstrip("#") for h in (b.get("source_hashtags") or []) if h}
        if held_tags and query_tags and not (held_tags & query_tags):
            skipped_no_overlap += 1
            continue
        out.append({
            "test_id": f"{b['test_id']}_c4",
            "source_test_id": b["test_id"],
            "source_timestamp": b["source_timestamp"],
            "user_query": b["user_query"],
            "prior_conversation": b["prior_conversation"],
            "held_out_preference": b["held_out_preference"],
            "top_k_relevant_prefs": b.get("top_k_relevant_prefs") or [],
            "blind_check_generic_answer": b.get("blind_check_generic_answer", ""),
        })
    if skipped_no_overlap or skipped_no_pref:
        print(
            f"[build_benchmark] preference_removal_regen filter: "
            f"kept={len(out)} skipped_no_overlap={skipped_no_overlap} "
            f"skipped_no_pref={skipped_no_pref}"
        )
    return out


# --- Task C1: repetition-fatigue probes (LEGACY — kept but deprecated) -----

def build_c1_instances(bq: BackendQuery, user_id: str, t_probe: int, min_positive_count: int = 10) -> list[dict]:
    hashtag_rows = bq.hashtag_summary(user_id=user_id, since_timestamp=t_probe)
    out: list[dict] = []
    for row in hashtag_rows:
        if row["positive"] < min_positive_count:
            continue
        for app in SOCIAL_APPS:
            probe = scenarios_mod.build_repetition_probe(bq, user_id, t_probe, app, row["hashtag"])
            if probe:
                out.append({
                    "probe_id": f"{user_id}_{app}_{row['hashtag'].lstrip('#')}",
                    "t_probe": t_probe,
                    **probe,
                })
                break
    return out


# --- Task C2: scenario instances -------------------------------------------

def build_c2_instances(bq: BackendQuery, user_id: str, t_probe: int, rng_seed: int) -> list[dict]:
    scs = scenarios_mod.build_all_scenarios(bq, user_id, t_probe, seed=rng_seed)
    return [
        {
            "scenario_id": f"{user_id}_{s['name']}",
            "t_probe": t_probe,
            **s,
        }
        for s in scs
    ]


# --- Task C3: restraint instances ------------------------------------------

def build_c3_instance(test: TestItem, rng: random.Random) -> dict | None:
    """Build an over_personalization_distractor_reject instance (Phase I.3).

    Converted from the old 4-way ranking task to an OPEN-ENDED CHATBOT
    response. The user asks a generic question (synthesized from a small
    set of universally-applicable templates) — the agent must NOT volunteer
    any of the user's irrelevant preferences in the answer. Judged by the
    chatbot_response runner using personalization_leak_rate over the
    irrelevant_persona_items list as the "do-not-surface" pool.

    Routing: `over_personalization_distractor_reject` is dispatched to
    `chatbot_response.run_task_b` (the same runner that handles the
    proactive + control + adversarial chatbot probes).
    """
    if not test.over_personalization_irrelevant:
        return None
    irrels = list(test.over_personalization_irrelevant)

    # Synthesize a generic question — choose deterministically by hashing
    # the test_id so two regens produce the same question for the same test.
    GENERIC_QUERIES = [
        "how do you organize your notes app? mine's a mess.",
        "any breathing exercise that actually works for stress?",
        "best way to make a new word stick in another language?",
        "static cling on my clothes is brutal in winter, fix?",
        "quick desk warm-up before I sit for 4 hours?",
        "keyboard's getting gross. how do you keep yours clean?",
        "what's your trick for remembering someone's name?",
        "can't sleep when it's hot. tips?",
    ]
    seed = abs(hash(test.source_object_id)) % len(GENERIC_QUERIES)
    user_query = GENERIC_QUERIES[seed]

    # Same shape as chatbot_response control-arm instances so the
    # existing _finalize() path in build_task_b_arms can handle it
    # (we'll wire it in alongside the adversarial probes when we build the
    # benchmark — for now the shape's enough that run_task_b can grade it).
    return {
        "test_id": f"distractor_{test.source_object_id}",
        "arm": "distractor_reject",
        "source_timestamp": test.source_timestamp,
        "formatted_timestamp": test.formatted_timestamp,
        "user_query": user_query,
        "prior_conversation": [],
        "action": "asked_chatbot",
        "source_hashtags": [],
        "held_out_preference": None,
        "blind_check_score": None,
        "blind_check_generic_answer": None,
        # The KEY field for grading: any of these persona items appearing
        # in the agent's response is a personalization-leak failure. We
        # repurpose the chatbot_response runner's `privacy_flagged_prefs`
        # field for this — the runner already grades leak_rate against it.
        "privacy_flagged_prefs": [
            {"persona_item": p.get("persona_item", ""), "category": p.get("category", "")}
            for p in irrels if p.get("persona_item")
        ],
        # No top-k for this task — the whole point is the agent shouldn't
        # surface ANY persona context.
        "top_k_relevant_prefs": [],
        # Empty gt_slice — control-arm scoring expects this shape.
        "gt_slice": {"target": [], "avoid": [], "t_test": test.source_timestamp, "window_seconds": 86400},
        # Empty post-test window — no behavioral signal needed for restraint.
        "post_test_window": {"post_test_positives": [], "post_test_negatives": []},
        # Keep the irrelevant_persona_items list as a flat string field so
        # the HTML test-card extractor can render them. (Distinct from the
        # privacy_flagged_prefs list which is what the runner grades on.)
        "irrelevant_persona_items": [p.get("persona_item", "") for p in irrels],
        "_distractor_reject": True,
    }


# --- Top-level build -------------------------------------------------------

def build_benchmark(
    backend_dir: str | Path,
    user_id: str,
    rng_seed: int = 0,
    blind_check_llm=None,
    blind_check_limit: int | None = None,
    discovery_llm=None,
) -> dict:
    """Build the full per-user benchmark.

    `discovery_llm` — optional LLM client used by E6 (active mistake
    prevention) for per-user discovery of paired (warn, foil) scenarios.
    When None, E6 yields zero instances; other tasks are unaffected.
    """
    bq = BackendQuery(backend_dir)
    test_items = load_test_items(backend_dir, user_id)
    if not test_items:
        raise SystemExit(f"No test items found for user {user_id} under {backend_dir}/")

    # Task A slates — per-item seeded with audit-and-regenerate retry loop.
    from evaluation.audit_helpers import audit_instance, BuildAuditReporter, make_blind_baseline_for_ranking
    auditor = BuildAuditReporter(user_id=user_id)
    # Phase I.4: when blind_check_llm is provided, run a blind-baseline LLM
    # probe on every candidate slate. If the model can pick the held-out by
    # text alone (no user history), the slate is contaminated → regenerate.
    blind_check = make_blind_baseline_for_ranking(blind_check_llm) if blind_check_llm is not None else None
    slate_instances = []
    for t in test_items:
        if t.app not in SOCIAL_APPS:
            continue

        def _build_with_seed_bump(_inst, bump: int):
            # Re-roll the per-instance RNG with a different salt — re-shuffles
            # distractor order + which past/future positives get picked.
            rng_retry = _instance_rng(rng_seed + bump, f"slate:{t.source_object_id}")
            return build_slate_instance(t, bq, rng_retry)

        rng = _instance_rng(rng_seed, f"slate:{t.source_object_id}")
        candidate = build_slate_instance(t, bq, rng)
        kept, audit_report = audit_instance(
            candidate, "personalized_feed_ranking",
            rebuild_fn=_build_with_seed_bump, max_attempts=3,
            blind_baseline=blind_check,
        )
        auditor.record("personalized_feed_ranking", audit_report, kept is not None)
        if kept is not None:
            slate_instances.append(kept)
        else:
            print(f"[build_benchmark] WARN: dropping personalized_feed_ranking "
                  f"instance for test_id={t.source_object_id} after 3 failed audit attempts")

    # Task B (v2) — proactive + control arms with build-time curation.
    b_arms = build_task_b_arms(
        backend_dir=backend_dir,
        bq=bq,
        user_id=user_id,
        test_items=test_items,
        blind_check_llm=blind_check_llm,
        blind_check_limit=blind_check_limit,
    )

    # Task C1a/C1b/C2/C3/C4.
    t_probe = max(t.source_timestamp for t in test_items)
    c1a_pairs = build_c1a_pairs(bq, user_id, test_items)
    c1b_sequences = build_c1b_sequence(b_arms["chatbot_proactive_personalization"])
    c2_instances = build_c2_instances(bq, user_id, t_probe, rng_seed=rng_seed)
    c4_instances = build_c4_instances(b_arms["chatbot_proactive_personalization"])

    # Agentic tasks T6-T19 — all share t_probe. Each builder's output is
    # passed through `_split_arms` (workstream G) which appends an
    # overpersonalization arm alongside the proactive one.
    from evaluation.tasks.agentic_tasks import ALL_BUILDERS as _AGENTIC_BUILDERS
    from evaluation.tasks.agentic_tasks import _split_arms as _agentic_split_arms
    agentic_buckets: dict[str, list[dict]] = {}
    for task_id, builder in _AGENTIC_BUILDERS.items():
        try:
            proactive = builder(bq, user_id, t_probe)
            agentic_buckets[task_id] = _agentic_split_arms(proactive, task_id)
        except Exception as exc:
            agentic_buckets[task_id] = []
            print(f"[build_benchmark] WARN: {task_id} builder failed: {exc}")

    # Task E2 — @ai proactive recommendation (R9 addition)
    try:
        from evaluation.tasks.e2_at_ai_followup import build_e2_at_ai_followup
        e2_instances = build_e2_at_ai_followup(bq, user_id, rng_seed=rng_seed)
    except Exception as exc:
        e2_instances = []
        print(f"[build_benchmark] WARN: e2_at_ai_followup builder failed: {exc}")

    # Task E3 — multi-day proactive daily briefing
    try:
        from evaluation.tasks.e3_daily_briefing_multi import build_e3_daily_briefing_multi
        e3_instances = build_e3_daily_briefing_multi(bq, user_id, t_probe)
    except Exception as exc:
        e3_instances = []
        print(f"[build_benchmark] WARN: e3_daily_briefing_multi builder failed: {exc}")

    # Task E4 — Google Search personalization (opt-in at run time; always built)
    try:
        from evaluation.tasks.e4_google_search import build_e4_google_search
        e4_instances = build_e4_google_search(bq, user_id, t_probe)
    except Exception as exc:
        e4_instances = []
        print(f"[build_benchmark] WARN: e4_google_search builder failed: {exc}")

    # Task E5 — short-term horizon lifecycle
    try:
        from evaluation.tasks.e5_horizon_lifecycle import build_e5_horizon_lifecycle
        e5_instances = build_e5_horizon_lifecycle(bq, user_id, rng_seed=rng_seed)
    except Exception as exc:
        e5_instances = []
        print(f"[build_benchmark] WARN: e5_horizon_lifecycle builder failed: {exc}")

    # Task E6 — active mistake prevention (paired warn/foil discovery)
    try:
        from evaluation.tasks.e6_active_mistake_prevention import build_e6_active_mistake_prevention
        if discovery_llm is None:
            print("[build_benchmark] e6: no discovery_llm — skipping E6 instances")
            e6_instances = []
        else:
            e6_instances = build_e6_active_mistake_prevention(
                bq, user_id, llm_client=discovery_llm, rng_seed=rng_seed,
            )
    except Exception as exc:
        e6_instances = []
        print(f"[build_benchmark] WARN: e6_active_mistake_prevention builder failed: {exc}")

    c3_instances = []
    for t in test_items:
        if t.app not in SOCIAL_APPS or not t.over_personalization_irrelevant:
            continue
        rng = _instance_rng(rng_seed, f"c3:{t.source_object_id}")
        inst = build_c3_instance(t, rng)
        if inst is not None:
            c3_instances.append(inst)

    # Persist Phase D audit report.
    try:
        out_dir = Path("benchmark") / user_id
        out_dir.mkdir(parents=True, exist_ok=True)
        auditor.write(out_dir)
    except Exception as exc:
        print(f"[build_benchmark] WARN: failed to write build_audit.json: {exc}")

    # Apply per-task quotas (stratified random truncation when over cap).
    # Floor enforcement is the synthesis layer's job — this only caps.
    pre_cap_buckets = {
        "personalized_feed_ranking":              slate_instances,
        "chatbot_proactive_personalization":      b_arms["chatbot_proactive_personalization"],
        "over_personalization_chatbot_text":      b_arms["over_personalization_chatbot_text"],
        "repetition_fatigue_pairs":               c1a_pairs,
        "repetition_fatigue_sequences":           c1b_sequences,
        "context_shift_scenarios":                c2_instances,
        "over_personalization_distractor_reject": c3_instances,
        "preference_removal_regen":               c4_instances,
        "at_ai_directive_followup":               e2_instances,
        "daily_personalized_briefing":            e3_instances,
        # workstream D: e4 builder now emits the personalized_recommendation
        # task_type. Old name retained as alias via OLD_TO_NEW for legacy CSVs.
        "personalized_recommendation":            e4_instances,
        "short_vs_long_term_lifecycle":           e5_instances,
        "active_mistake_prevention":              e6_instances,
        **agentic_buckets,
    }
    capped_buckets = _task_dist.apply_caps(dict(pre_cap_buckets), rng_seed=rng_seed)
    floor_gaps = _task_dist.report_floor_gaps(capped_buckets)
    if floor_gaps:
        print(f"[build_benchmark] floor gaps (will be filled by synthesis): {floor_gaps}")

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "user_id": user_id,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rng_seed": rng_seed,
        "backend_hash": compute_backend_hash(backend_dir, user_id),
        "blind_check_enabled": blind_check_llm is not None,
        "build_audit": auditor._stats,
        "counts": {
            "test_items": len(test_items),
            **{k: len(v) for k, v in capped_buckets.items()},
        },
        "floor_gaps": floor_gaps,
        **capped_buckets,
    }


def default_benchmark_path(user_id: str) -> Path:
    return Path("benchmark") / user_id / "benchmark.json"


def default_benchmark_csv_path(user_id: str) -> Path:
    return Path("benchmark") / user_id / "benchmark.csv"


def export_benchmark_csv(benchmark: dict, user_id: str, out_path: Path | None = None) -> Path:
    """Project a `benchmark.json` blob into a flat CSV for HuggingFace publication.

    Columns (stable, narrow):
      instance_id, task, user_id, t_test, t_test_iso, query, query_type,
      candidates_json, ground_truth_json, carveout_json, metadata_json

    Task-specific fields that don't fit the narrow schema are JSON-packed
    into `metadata_json`. The runner continues to consume the structured
    JSON; the CSV is a publication-friendly projection.
    """
    import csv
    import datetime as _dt

    out_path = Path(out_path) if out_path else default_benchmark_csv_path(user_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Iterate task buckets. A benchmark dict has two kinds of top-level keys:
    # metadata (counts, backend_hash, etc.) and task buckets (lists of instances).
    columns = [
        "instance_id", "task", "user_id", "t_test", "t_test_iso",
        "query", "query_type", "candidates_json", "ground_truth_json",
        "carveout_json", "metadata_json",
    ]

    def _isoformat_ts(ts) -> str:
        if not isinstance(ts, (int, float)) or ts <= 0:
            return ""
        try:
            return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""

    def _project_row(task: str, inst: dict) -> dict | None:
        if not isinstance(inst, dict):
            return None
        t_test = inst.get("t_test") or inst.get("test_timestamp") or inst.get("source_timestamp") or 0
        query = inst.get("query") or inst.get("user_message") or ""
        query_type = inst.get("query_type") or inst.get("entry_point") or ""
        candidates = inst.get("candidates") or inst.get("slate") or []
        # Ground truth — task-specific; collect the commonly named ones
        gt = {
            k: inst[k] for k in (
                "held_out_preference", "held_out_indices", "positive_indices",
                "target_match", "target_ids", "forbidden_items", "matching_indices",
                "post_test_positives", "post_test_negatives", "post_test_engagements",
            )
            if k in inst
        }
        carveout = {
            k: inst[k] for k in ("carveout_indices", "carve_out_topic", "carveout_topic")
            if k in inst
        }
        # Everything else goes into metadata_json
        known_keys = {
            "instance_id", "task_id", "t_test", "test_timestamp", "source_timestamp",
            "query", "user_message", "query_type", "entry_point",
            "candidates", "slate",
        } | set(gt.keys()) | set(carveout.keys())
        metadata = {k: v for k, v in inst.items() if k not in known_keys}
        return {
            "instance_id": str(inst.get("instance_id") or ""),
            "task": task,
            "user_id": str(user_id),
            "t_test": int(t_test) if isinstance(t_test, (int, float)) else 0,
            "t_test_iso": _isoformat_ts(t_test),
            "query": str(query) if query is not None else "",
            "query_type": str(query_type) if query_type else "",
            "candidates_json": json.dumps(candidates, ensure_ascii=False) if candidates else "",
            "ground_truth_json": json.dumps(gt, ensure_ascii=False) if gt else "",
            "carveout_json": json.dumps(carveout, ensure_ascii=False) if carveout else "",
            "metadata_json": json.dumps(metadata, ensure_ascii=False) if metadata else "",
        }

    rows: list[dict] = []
    for task_key, bucket in benchmark.items():
        if not isinstance(bucket, list):
            continue  # skip metadata keys
        for inst in bucket:
            row = _project_row(task_key, inst)
            if row is not None:
                rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path


# Note: this module no longer ships its own CLI. `scripts/prepare_eval_data.py`
# is the single entry point for benchmark construction — it owns the
# blind_check helper, wires both blind_check + E6 discovery LLMs, and writes
# `benchmark/{uid}/queries.csv` (the artifact `evaluation/run_eval.py` reads).
# `build_benchmark()`, `compute_backend_hash`, `export_benchmark_csv`, and the
# per-task helpers are still importable from here as a library.
