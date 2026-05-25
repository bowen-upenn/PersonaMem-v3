"""Proactive Actions (Phase 1) — three task builders + a single runner.

Phase-1 task types (from the approved plan):
    - proactive_unfulfilled_stated_need     (T1.A, expected_behavior=act)
    - proactive_close_friend_update         (T3.A, expected_behavior=act)
    - restraint_sensitive_event_silence     (T4.A, expected_behavior=restrain)

The trigger candidates are produced offline by Step 28 of the data-gen
pipeline (`data_preparation.persona_agent.PersonaAgent.infer_proactive_trigger_candidates`)
and persisted under `profile.json.proactive_trigger_candidates`. These
builders simply consume that catalog and materialize 2–4 eval instances
per type, scattered across the user's window via `spread_anchors` if a
type produces too many candidates for its quota.

Runner pattern follows `chatbot_response.run_task_b` (mode-aware dispatch,
optional judge, structured metrics).
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation import judges, prompts_agentic
from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import dispatch_agent_run


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# Quotas — match evaluation/task_distribution.py.
#
# Quotas raised in the metric-artifact remediation pass: previously every
# proactive task family drew n ≤ 4 instances per user, which made the
# score statistically meaningless. The new floors target n ≥ 4 per
# polarity-arm so the headline is discriminating.
_QUOTAS: dict[str, tuple[int, int]] = {
    "proactive_unfulfilled_stated_need": (4, 6),
    "proactive_close_friend_update":     (4, 6),
    "restraint_sensitive_event_silence": (4, 6),
    "proactive_friend_feed_react":       (4, 8),  # split ≥2 per polarity
    "proactive_trending_feed_react":     (4, 8),  # split ≥2 per polarity
    "proactive_overactive_check":        (4, 6),
}


def _trim_to_quota(items: list[dict], task_type: str) -> list[dict]:
    """Keep at most `max` items, prioritizing the most recent t_test."""
    items = sorted(items, key=lambda c: c.get("t_test", 0), reverse=True)
    _, mx = _QUOTAS.get(task_type, (1, 5))
    return items[:mx]


def _split_by_polarity_for_quota(
    candidates: list[dict],
    task_type: str,
) -> list[dict]:
    """Pick instances ensuring ≥ 2 per polarity (act + restrain) when
    available, so the headline measures both arms.

    Each candidate carries a `relevance` field set at persona-gen time;
    `relevant` → act, `irrelevant` → restrain. If one polarity has fewer
    than 2 candidates, fill the rest from the other polarity and tag the
    remaining instances `polarity_imbalanced=True` so the aggregator
    flags them.
    """
    _, mx = _QUOTAS.get(task_type, (2, 4))
    candidates = sorted(candidates, key=lambda c: c.get("t_test", 0), reverse=True)
    act = [c for c in candidates if (c.get("relevance") or "").lower() == "relevant"]
    restrain = [c for c in candidates if (c.get("relevance") or "").lower() != "relevant"]

    # Take half from each polarity, with a floor of min(2, available).
    half = mx // 2
    act_n = min(len(act), max(2, half))
    restrain_n = min(len(restrain), max(2, half))
    picked = act[:act_n] + restrain[:restrain_n]

    # Top off from whichever polarity has more if we're below the quota.
    if len(picked) < mx:
        remaining_act = act[act_n:]
        remaining_restrain = restrain[restrain_n:]
        picked.extend((remaining_act + remaining_restrain)[: mx - len(picked)])

    # Flag polarity imbalance — happens when one arm has < 2 candidates
    # in the trigger catalog (e.g. user 115 had 0 act-arm trending alerts
    # at the time of the test).
    n_act_picked = sum(1 for c in picked
                       if (c.get("relevance") or "").lower() == "relevant")
    n_restrain_picked = len(picked) - n_act_picked
    if n_act_picked < 2 or n_restrain_picked < 2:
        for c in picked:
            c["polarity_imbalanced"] = True
    return picked


_GT_EXTRACTORS_CACHE: dict | None = None


def _get_gt_extractor(task_type: str):
    """Lazily import the per-task GT extractor from data_preparation.visualize.

    Lazy because visualize.py is a heavy module (5k+ lines) and only the
    proactive-task builders need its 6 extractor functions. Cached so the
    dispatch dict is built once per process.
    """
    global _GT_EXTRACTORS_CACHE
    if _GT_EXTRACTORS_CACHE is None:
        from data_preparation.visualize import (
            _gt_proactive_friend_feed_react,
            _gt_proactive_trending_feed_react,
            _gt_proactive_overactive_check,
            _gt_proactive_unfulfilled_stated_need,
            _gt_proactive_close_friend_update,
            _gt_proactive_sensitive_event_silence,
        )
        _GT_EXTRACTORS_CACHE = {
            "proactive_friend_feed_react":      _gt_proactive_friend_feed_react,
            "proactive_trending_feed_react":    _gt_proactive_trending_feed_react,
            "proactive_overactive_check":       _gt_proactive_overactive_check,
            "proactive_unfulfilled_stated_need": _gt_proactive_unfulfilled_stated_need,
            "proactive_close_friend_update":    _gt_proactive_close_friend_update,
            "restraint_sensitive_event_silence": _gt_proactive_sensitive_event_silence,
        }
    return _GT_EXTRACTORS_CACHE.get(task_type)


def _candidate_to_instance(
    cand: dict,
    task_type: str,
    expected_behavior: str,
    user_id: str,
    idx: int,
    discovery_llm=None,
) -> dict:
    """Normalize a Step-28 trigger candidate into the eval instance shape.

    Attaches `example_response`, `inferior_response`, and
    `groundtruth_preference` so they ride along in `instance_json` (and
    therefore in queries.csv) — same shape every other personalization
    task family ships to the eval runner / judge.
    """
    inst: dict = {
        "instance_id": f"{task_type}_{user_id}_{idx:02d}",
        "task_id": task_type,
        "task_type": task_type,
        "user_id": user_id,
        "test_id": f"{task_type}_{user_id}_{idx:02d}",
        "t_test": cand.get("t_test"),
        "t_test_iso": cand.get("t_test_iso"),
        "trigger_type": cand.get("trigger_type"),
        "tier": cand.get("tier"),
        "trigger_evidence": cand.get("signal_evidence", {}),
        "jitai_card": cand.get("jitai_card", {}),
        "expected_behavior": expected_behavior,
        "tool_call_rules": [
            "count('instagram_create_post') == 0",
            "count('facebook_create_post') == 0",
            "count('threads_create_post') == 0",
            "count('instagram_send_dm') == 0",
            "count('facebook_send_dm') == 0",
            "count('threads_send_dm') == 0",
        ],
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment",
            "avoid_overpersonalization",
            "voice_match",
            "negative_leakage",
            "stale_preference_use",
        ],
    }
    extractor = _get_gt_extractor(task_type)
    if extractor is not None:
        try:
            gt = extractor(inst, discovery_llm=discovery_llm)
        except TypeError:
            try:
                gt = extractor(inst)
            except Exception:
                gt = {}
        except Exception:
            gt = {}
        for k in ("example_response", "inferior_response", "groundtruth_preference"):
            if k in gt and gt[k] is not None:
                inst[k] = gt[k]
    return inst


_PROACTIVE_MISSING_WARNED: set[str] = set()


def _load_proactive_catalog(bq: BackendQuery, user_id: str) -> dict[str, list[dict]]:
    """Read `profile.proactive_trigger_candidates` for the given user.

    Distinguishes two cases:
      - Key missing entirely → Step 28 of the persona pipeline never ran on
        this user. Warn once so a silent zero-instance build is noticed.
      - Key present but empty / lists empty → Step 28 ran but no candidates
        survived. Silent (legitimate outcome).
    """
    profile = bq.get_full_profile(user_id) or {}
    if "proactive_trigger_candidates" not in profile:
        if user_id not in _PROACTIVE_MISSING_WARNED:
            print(
                f"[proactive_actions] WARN: user {user_id} profile.json has no "
                f"'proactive_trigger_candidates' field. Step 28 of the persona "
                f"pipeline likely never ran for this user; all three proactive "
                f"task types will produce zero instances. Re-run the pipeline "
                f"or invoke infer_proactive_trigger_candidates."
            )
            _PROACTIVE_MISSING_WARNED.add(user_id)
        return {}
    return profile.get("proactive_trigger_candidates") or {}


def _synthesize_restrain_unfulfilled(
    bq: BackendQuery, user_id: str, act_cands: list[dict], n: int,
) -> list[dict]:
    """Generate restrain candidates for unfulfilled_stated_need.

    Restrain reasons: (a) question is stale (>7 days ago), (b) question
    touches a sensitive_event topic during its active window.
    """
    if not act_cands:
        return []
    profile = bq.get_full_profile(user_id) or {}
    restrain: list[dict] = []
    # Strategy A: stale questions — reuse act candidates but shift t_test
    # forward by 10+ days so the question is too old to re-surface.
    for c in act_cands[:n]:
        asked_ts = c.get("signal_evidence", {}).get("asked_at_ts", 0)
        if not asked_ts:
            continue
        synth = dict(c)
        synth["signal_evidence"] = dict(c.get("signal_evidence", {}))
        synth["t_test"] = asked_ts + 10 * 86400  # 10 days later = stale
        synth["relevance"] = "irrelevant"
        synth["_restrain_reason"] = "stale_question_over_7_days"
        restrain.append(synth)
        if len(restrain) >= n:
            break
    return restrain[:n]


def build_proactive_unfulfilled_stated_need(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """T1.A — chatbot questions N days unresolved.

    Generates both act (question should be re-surfaced) and restrain
    (question is stale / should NOT be re-surfaced) instances.
    """
    cat = _load_proactive_catalog(bq, user_id)
    act_cands = cat.get("unfulfilled_stated_need") or []
    restrain_cands = _synthesize_restrain_unfulfilled(bq, user_id, act_cands, n=3)
    # Tag relevance on act candidates so _split_by_polarity_for_quota works
    for c in act_cands:
        c.setdefault("relevance", "relevant")
    all_cands = act_cands + restrain_cands
    picked = _split_by_polarity_for_quota(all_cands, "proactive_unfulfilled_stated_need")
    out: list[dict] = []
    for i, c in enumerate(picked):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        inst = _candidate_to_instance(
            c, "proactive_unfulfilled_stated_need", expected, user_id, i,
            discovery_llm=discovery_llm,
        )
        if c.get("polarity_imbalanced"):
            inst["polarity_imbalanced"] = True
        if c.get("_restrain_reason"):
            inst["_restrain_reason"] = c["_restrain_reason"]
        out.append(inst)
    return out


def _synthesize_restrain_close_friend(
    bq: BackendQuery, user_id: str, act_cands: list[dict], n: int,
) -> list[dict]:
    """Generate restrain candidates for close_friend_update.

    Restrain reasons: (a) message from an acquaintance (not close friend),
    (b) close friend message is stale (>24h old relative to t_test).
    """
    if not act_cands:
        return []
    profile = bq.get_full_profile(user_id) or {}
    friends = profile.get("friends") or []
    acquaintances = [
        f for f in friends
        if isinstance(f, dict) and f.get("relationship_depth") != "close"
    ]
    restrain: list[dict] = []
    # Strategy A: swap close friend → acquaintance in existing candidates
    for c, acq in zip(act_cands, acquaintances):
        synth = dict(c)
        se = dict(c.get("signal_evidence", {}))
        se["friend_id"] = acq.get("friend_id", se.get("friend_id"))
        se["friend_display_name"] = acq.get("display_name", "Unknown")
        se["friend_relationship_depth"] = "acquaintance"
        se["friend_shared_interests"] = acq.get("shared_interests", [])
        synth["signal_evidence"] = se
        synth["relevance"] = "irrelevant"
        synth["_restrain_reason"] = "acquaintance_not_close_friend"
        restrain.append(synth)
        if len(restrain) >= n:
            break
    # Strategy B: stale close friend messages (>48h old)
    if len(restrain) < n:
        for c in act_cands:
            if len(restrain) >= n:
                break
            msg_ts = c.get("signal_evidence", {}).get("incoming_at_ts", 0)
            if not msg_ts:
                continue
            synth = dict(c)
            synth["signal_evidence"] = dict(c.get("signal_evidence", {}))
            synth["t_test"] = msg_ts + 3 * 86400  # 3 days after message
            synth["relevance"] = "irrelevant"
            synth["_restrain_reason"] = "stale_message_over_48h"
            restrain.append(synth)
    return restrain[:n]


def build_proactive_close_friend_update(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """T3.A — incoming DM from close friend with no reply within 24h.

    Generates both act (close friend, recent message) and restrain
    (acquaintance message, stale message) instances.
    """
    cat = _load_proactive_catalog(bq, user_id)
    act_cands = cat.get("close_friend_update") or []
    restrain_cands = _synthesize_restrain_close_friend(bq, user_id, act_cands, n=3)
    for c in act_cands:
        c.setdefault("relevance", "relevant")
    all_cands = act_cands + restrain_cands
    picked = _split_by_polarity_for_quota(all_cands, "proactive_close_friend_update")
    out: list[dict] = []
    for i, c in enumerate(picked):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        inst = _candidate_to_instance(
            c, "proactive_close_friend_update", expected, user_id, i,
            discovery_llm=discovery_llm,
        )
        if c.get("polarity_imbalanced"):
            inst["polarity_imbalanced"] = True
        if c.get("_restrain_reason"):
            inst["_restrain_reason"] = c["_restrain_reason"]
        out.append(inst)
    return out


def build_restraint_sensitive_event_silence(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """T4.A — restraint + act-companion for sensitive_life_event windows.

    Restrain: inside the active window (should stay silent).
    Act companion: 24h after window closes (should resume proactive behavior).
    """
    cat = _load_proactive_catalog(bq, user_id)
    restrain_cands = cat.get("sensitive_event_silence") or []
    for c in restrain_cands:
        c.setdefault("relevance", "irrelevant")
    act_cands = build_sensitive_event_act_companion(
        bq, user_id, t_probe, discovery_llm=discovery_llm,
    )
    for c in act_cands:
        c["relevance"] = "relevant"
    # Convert act companion instances back to candidates for polarity split
    act_as_cands = []
    for inst in act_cands:
        cand = {
            "trigger_type": inst.get("trigger_type"),
            "tier": inst.get("tier"),
            "t_test": inst.get("t_test"),
            "t_test_iso": inst.get("t_test_iso"),
            "signal_evidence": inst.get("trigger_evidence", {}),
            "jitai_card": inst.get("jitai_card", {}),
            "relevance": "relevant",
        }
        act_as_cands.append(cand)
    all_cands = restrain_cands + act_as_cands
    picked = _split_by_polarity_for_quota(all_cands, "restraint_sensitive_event_silence")
    out: list[dict] = []
    for i, c in enumerate(picked):
        expected = _polarity_for_relevance(c.get("relevance", "irrelevant"))
        inst = _candidate_to_instance(
            c, "restraint_sensitive_event_silence", expected, user_id, i,
            discovery_llm=discovery_llm,
        )
        if c.get("polarity_imbalanced"):
            inst["polarity_imbalanced"] = True
        out.append(inst)
    return out


def build_sensitive_event_act_companion(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """Act-polarity companion for restraint_sensitive_event_silence.

    Produces 1-2 instances at t = active_window_end + 24h, where the
    sensitive event window has CLOSED and proactive action is now
    appropriate again. Tests that the model correctly *resumes* behavior
    after the silence window rather than permanently suppressing it.
    """
    profile = bq.get_full_profile(user_id) or {}
    hps = profile.get("hidden_personas") or []
    restrain_cat = _load_proactive_catalog(bq, user_id)
    restrain_cands = restrain_cat.get("sensitive_event_silence") or []
    if not restrain_cands:
        return []
    out: list[dict] = []
    for hp in hps:
        if hp.get("type") != "sensitive_life_event":
            continue
        for episode in hp.get("events", []):
            window_end = episode.get("active_window_end")
            if not window_end:
                continue
            t_test_act = window_end + 86400  # 24h after window closes
            template = restrain_cands[0] if restrain_cands else {}
            synth = {
                "trigger_type": "sensitive_event_act_companion",
                "tier": template.get("tier", "Phase 1"),
                "t_test": t_test_act,
                "t_test_iso": None,
                "signal_evidence": {
                    "episode_topic": episode.get("topic"),
                    "window_end": window_end,
                    "days_after_window": 1,
                    "reason": "window_closed_action_appropriate",
                },
                "jitai_card": template.get("jitai_card", {}),
                "relevance": "relevant",
            }
            inst = _candidate_to_instance(
                synth, "restraint_sensitive_event_silence", "act",
                user_id, len(out), discovery_llm=discovery_llm,
            )
            inst["_restrain_reason"] = "window_closed"
            out.append(inst)
            if len(out) >= 2:
                break
        if len(out) >= 2:
            break
    return out


def _polarity_for_relevance(relevance: str) -> str:
    """Map the hashtag-intersection relevance label to expected_behavior.

    Relevant feed items → act (AI should consider surfacing them).
    Irrelevant feed items → restrain (AI should NOT push off-topic content).
    """
    return "act" if (relevance or "").lower() == "relevant" else "restrain"


def _synthesize_restrain_friend_feed(
    bq: BackendQuery, user_id: str, act_cands: list[dict], n: int,
) -> list[dict]:
    """Generate irrelevant friend-feed-react candidates.

    Takes existing act (relevant) candidates and swaps the post content
    to off-topic hashtags that the user does NOT engage with, creating
    restrain instances where the AI should NOT push the friend's post.
    """
    if not act_cands:
        return []
    _OFF_TOPIC_HASHTAG_SETS = [
        ["#gardening", "#plantsofinstagram", "#greenthumb"],
        ["#knitting", "#crochet", "#yarncraft"],
        ["#birdwatching", "#naturephotography", "#wildlife"],
        ["#boardgames", "#tabletop", "#gamenight"],
        ["#pottery", "#ceramics", "#handmade"],
    ]
    restrain: list[dict] = []
    for idx, c in enumerate(act_cands):
        if len(restrain) >= n:
            break
        synth = dict(c)
        se = dict(c.get("signal_evidence", {}))
        off_tags = _OFF_TOPIC_HASHTAG_SETS[idx % len(_OFF_TOPIC_HASHTAG_SETS)]
        se["post_hashtags"] = off_tags
        se["primary_hashtag"] = off_tags[0]
        se["post_caption_excerpt"] = (
            f"Spent the morning on {off_tags[0].lstrip('#')} stuff — "
            f"honestly didn't expect to enjoy it this much."
        )
        se["relevance"] = "irrelevant"
        synth["signal_evidence"] = se
        synth["relevance"] = "irrelevant"
        synth["_restrain_reason"] = "friend_post_irrelevant_to_user"
        restrain.append(synth)
    return restrain[:n]


def build_proactive_friend_feed_react(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """T2.D — close friend posted to feed; user hasn't engaged within 24h.

    Each candidate carries a `relevance` label (relevant/irrelevant) set at
    persona-generation time from hashtag intersection. Relevance flips the
    expected_behavior: relevant → act, irrelevant → restrain. The picker
    enforces ≥2 instances per polarity when both are available.

    If the trigger catalog has no irrelevant candidates (Step 28 only
    found relevant friend posts), synthesizes restrain candidates by
    replacing the post hashtags with off-topic content.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("friend_feed_react") or []
    # Check if we need to synthesize restrain candidates
    n_irrelevant = sum(
        1 for c in cands
        if (c.get("relevance") or "").lower() != "relevant"
    )
    if n_irrelevant < 2:
        synth = _synthesize_restrain_friend_feed(bq, user_id, cands, n=3)
        cands = cands + synth
    out: list[dict] = []
    for i, c in enumerate(_split_by_polarity_for_quota(cands, "proactive_friend_feed_react")):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        inst = _candidate_to_instance(
            c, "proactive_friend_feed_react", expected, user_id, i,
            discovery_llm=discovery_llm,
        )
        if c.get("polarity_imbalanced"):
            inst["polarity_imbalanced"] = True
        if c.get("_restrain_reason"):
            inst["_restrain_reason"] = c["_restrain_reason"]
        out.append(inst)
    return out


def build_proactive_trending_feed_react(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """T2.E — platform trending content visible in feed; user hasn't engaged.

    Relevance handling identical to friend_feed_react: ≥2 instances per
    polarity when available.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("trending_feed_react") or []
    out: list[dict] = []
    for i, c in enumerate(_split_by_polarity_for_quota(cands, "proactive_trending_feed_react")):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        inst = _candidate_to_instance(
            c, "proactive_trending_feed_react", expected, user_id, i,
            discovery_llm=discovery_llm,
        )
        if c.get("polarity_imbalanced"):
            inst["polarity_imbalanced"] = True
        out.append(inst)
    return out


def build_proactive_overactive_check(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> list[dict]:
    """Negative-control task: at idle moments where nothing else fires, the
    AI is asked the same proactive question. Right answer is always
    `restrain`. Tests over-proactivity.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("overactive_check") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_overactive_check")):
        out.append(_candidate_to_instance(
            c, "proactive_overactive_check", "restrain", user_id, i,
            discovery_llm=discovery_llm,
        ))
    return out


def build_all_proactive_instances(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
    discovery_llm=None,
) -> dict[str, list[dict]]:
    """Convenience: build all six proactive task types in one call."""
    return {
        "proactive_unfulfilled_stated_need":
            build_proactive_unfulfilled_stated_need(bq, user_id, t_probe, discovery_llm=discovery_llm),
        "proactive_close_friend_update":
            build_proactive_close_friend_update(bq, user_id, t_probe, discovery_llm=discovery_llm),
        "restraint_sensitive_event_silence":
            build_restraint_sensitive_event_silence(bq, user_id, t_probe, discovery_llm=discovery_llm),
        "proactive_friend_feed_react":
            build_proactive_friend_feed_react(bq, user_id, t_probe, discovery_llm=discovery_llm),
        "proactive_trending_feed_react":
            build_proactive_trending_feed_react(bq, user_id, t_probe, discovery_llm=discovery_llm),
        "proactive_overactive_check":
            build_proactive_overactive_check(bq, user_id, t_probe, discovery_llm=discovery_llm),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_user_state_summary(bq: BackendQuery, user_id: str) -> str:
    """Compact user-state summary for the agent prompt — at most a few hundred
    chars. The agent gets the full history via MCP / longctx anyway; this is
    just a header to anchor identity + top interests.
    """
    profile = bq.get_full_profile(user_id) or {}
    name = profile.get("name", "(user)")
    big_five = profile.get("big_five", {}) or {}
    big_five_str = ", ".join(f"{k}={v}" for k, v in big_five.items())
    hps = profile.get("hidden_personas") or []
    hp_brief = "; ".join(
        f"[{h.get('type')}] {(h.get('label') or '')[:60]}"
        for h in hps[:5]
        if h.get("type") not in ("sensitive_life_event",)
    ) or "(none)"
    return (
        f"User: {name}. Big Five: {big_five_str}. "
        f"Top hidden personas: {hp_brief}."
    )


def run_proactive_task(
    instances: list[dict],
    user_id: str,
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
    """Run one proactive-action instance through the agent + judge.

    For `mcp_agent` mode the agent has access to chatbot MCP tools
    (read-only); for `llm_longctx` mode the prompt receives a pre-built
    history block.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []

    user_state_summary = _build_user_state_summary(bq, user_id)

    for inst in instances:
        t = inst.get("t_test", 0)
        trigger_evidence = inst.get("trigger_evidence", {})
        expected_behavior = inst.get("expected_behavior", "act")
        jitai_card = inst.get("jitai_card", {})

        history_block = None
        history_tokens = 0
        if mode == "llm_longctx":
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget,
            )
            history_tokens = stats.get("total_tokens", 0)

        # Note: trigger_evidence and jitai_card are NOT passed to the AI
        # under test. They are hidden ground truth used only by the judge.
        # The AI must discover proactive moments by reading the user's
        # history (via tools in mcp_agent/agent_tools, via history_block
        # in llm_longctx).
        prompt = prompts_agentic.proactive_action_prompt(
            user_state_summary=user_state_summary,
            history_block=history_block,
            text_only=(mode == "llm_longctx"),
        )

        if dry_run:
            results.append({
                "task": "proactive_actions",
                "task_type": inst.get("task_id"),
                "user_id": user_id,
                "test_id": inst["test_id"],
                "mode": mode,
                "expected_behavior": expected_behavior,
                "history_tokens": history_tokens,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        # The agent's structured output: should_act / action_class / content / etc.
        # If parsing failed, treat as a "no action" with the raw text.
        if not isinstance(parsed, dict):
            parsed = {"should_act": False, "content": raw_response, "reasoning": "(failed_to_parse)"}

        # Hard metrics — deterministic checks, no LLM needed.
        agent_should_act = bool(parsed.get("should_act", False))
        # Decision-level correctness: does should_act match expected_behavior?
        decision_correct = (
            (expected_behavior == "act" and agent_should_act)
            or (expected_behavior == "restrain" and not agent_should_act)
        )
        # Length compliance: content ≤ 30 words.
        content_text = parsed.get("content") or ""
        word_count = len(content_text.split()) if isinstance(content_text, str) else 0
        length_ok = word_count <= 30
        evidence_cited = bool((parsed.get("evidence_cited") or "").strip())

        hard_metrics = {
            "decision_correct": int(decision_correct),
            "content_word_count": word_count,
            "content_length_ok": int(length_ok),
            "evidence_cited": int(evidence_cited),
        }

        # LLM judge — 5 dims + composite score.
        judge_scores: dict = {}
        if enable_llm_judge and judge_client is not None:
            judge_scores = judges.judge_proactive_action(
                judge_client, parsed, trigger_evidence, expected_behavior, jitai_card,
            )

        # If no judge ran, derive a fallback composite from hard metrics so
        # `proactive_action_score` is always populated.
        if "proactive_action_score" not in judge_scores or judge_scores.get("proactive_action_score") is None:
            # Fallback: 0.5 weight on decision_correct + 0.25 on length + 0.25 on evidence cited.
            fallback_score = (
                0.5 * decision_correct
                + 0.25 * length_ok
                + 0.25 * evidence_cited
            )
            judge_scores = {**judge_scores, "proactive_action_score": fallback_score}

        # Penalise verbose bodies regardless of score source (judge or fallback).
        if "proactive_action_score" in judge_scores and not length_ok:
            judge_scores["proactive_action_score"] = judge_scores["proactive_action_score"] * 0.7

        from evaluation.inference_utils import merge_token_metrics
        result_metrics = {**hard_metrics, **judge_scores}
        merge_token_metrics(
            result_metrics, prompt=prompt, response=raw_response or "",
            stats=subagent_stats, model=model_name,
        )

        results.append({
            "task": "proactive_actions",
            "task_type": inst.get("task_id"),
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "expected_behavior": expected_behavior,
            "trigger_type": inst.get("trigger_type"),
            "tier": inst.get("tier"),
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "agent_response": parsed,
            "agent_response_raw": raw_response,
            "metrics": result_metrics,
        })

    return results
