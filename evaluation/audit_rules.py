"""Modular audit rules for backend/{uid}/test.json.

Each rule consumes a record dict (one entry from test.json) and returns
zero or more Finding objects. The audit driver iterates rules over
records and aggregates findings.

Distribution checks operate on the FULL list of records and live below
the per-record rules.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from typing import Iterable


@dataclass
class Finding:
    query_id: str
    task_type: str
    severity: str          # "high" | "medium" | "low"
    rule: str              # short identifier — used to group in the report
    message: str
    suggested_action: str  # "regenerate" | "drop" | "review" | "rebalance"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-record rules
# ---------------------------------------------------------------------------

# Task types that mimic a real user-typed message.
_USER_QUERY_KINDS = {"user_query"}

# Task types whose ground truth is a single ``held_out_preference`` the
# agent's text response is graded against. Ranking tasks live in
# _RANKING_TASKS below — they encode GT inside the slate / candidates
# rather than a single held-out preference, so different rules apply.
_REACTIVE_PERSONALIZATION_TASKS = {
    "chatbot_proactive_personalization",
    "preference_removal_regen",
}

# Ranking-style tasks: GT lives in slate / candidates / recent_pref_summary
# rather than ``held_out_preference``. We check for those structures
# separately.
_RANKING_TASKS = {
    "personalized_feed_ranking",
    "at_ai_directive_followup",
    "personalized_recommendation",
    "personalized_search_ranking",  # legacy alias
    "short_vs_long_term_lifecycle",
}

# Tasks that test diversification across a sequence rather than
# text-restraint against a do-not-surface pool. Their GT signal lives
# elsewhere (category shift, hashtag spread); they don't need a
# distractor_preferences pool.
_DIVERSIFICATION_TASKS = {
    "repetition_fatigue_pairs",
    "repetition_fatigue_sequences",
}

# Tasks that are intentionally hollow at build time: the agent is
# expected to fetch state at runtime via MCP tools (list_dm_threads,
# get_top_hashtags_for_window, ...). Skipping audit checks that look
# for instance-side ground / preconditions for these.
_HOLLOW_BY_DESIGN = {
    "agentic_dm_digest",
    "agentic_group_dm_summary",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
    "agentic_user_tone_post",
    "daily_personalized_briefing",
    "personalized_search_ranking",
}

# Task types that test restraint and therefore need a populated distractor pool
# (queries that the agent should NOT respond to with personalization).
_RESTRAINT_TASKS = {
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_sensitive_event",
    "over_personalization_context_shift",
    "repetition_fatigue_pairs",
    "repetition_fatigue_sequences",
}

_AGENTIC_PRECONDITION_HINTS = {
    # Tasks where the precondition genuinely belongs in the BUILD-TIME
    # instance (the agent cannot fetch it at runtime). Tasks like T8/T15/
    # T16/T18/T19 are deliberately hollow — the agent calls
    # list_dm_threads etc. via MCP at runtime, so missing instance-side
    # context is by design.
    "agentic_auto_reply":            "a real inbound DM (sender_id + inbound_message)",
    "agentic_wrong_recipient_check": "a name collision (recipient_name + draft + collision_friend_ids)",
    "agentic_cross_app_repost":      "a source post (source_post)",
    "agentic_send_post":             "a brief from chat (context)",
    "agentic_composed_post":         "a life-update string (update)",
    "agentic_draft_audit":           "a draft to audit (draft + draft_label)",
    "agentic_vague_refind":          "a topic the user vaguely remembers (topic)",
    "agentic_moment_recommendation": "a moment to recommend for (moment)",
}


def check_realism(record: dict) -> list[Finding]:
    """Query text reads like a real user — not a template stub."""
    out: list[Finding] = []
    qk = record.get("query_kind")
    text = (record.get("user_query") or "").strip()
    if qk in _USER_QUERY_KINDS:
        if len(text) < 10:
            out.append(Finding(
                record["query_id"], record["task_type"], "high", "realism_too_short",
                f"user_query is only {len(text)} chars: {text!r}",
                "regenerate",
            ))
        if "{" in text and "}" in text:
            out.append(Finding(
                record["query_id"], record["task_type"], "high", "realism_unfilled_template",
                f"user_query still contains template placeholder: {text[:120]!r}",
                "regenerate",
            ))
        # Synthetic [task tag] markers indicate the extractor fell through
        # to a default.
        if text.startswith("[") and "]" in text:
            out.append(Finding(
                record["query_id"], record["task_type"], "medium", "realism_synthetic_marker",
                f"user_query starts with synthetic marker: {text[:60]!r}",
                "regenerate",
            ))
    return out


def check_ground_truth_presence(record: dict) -> list[Finding]:
    """Personalization tasks must carry a real ground-truth preference;
    ranking tasks must carry a non-trivial slate / candidate set.

    Workstream C renamed test.json's structured held-out block from
    `ground_truth_preference` to `groundtruth_preference_obj` (the
    plain `groundtruth_preference` field is now the rendered string)."""
    out: list[Finding] = []
    tt = record["task_type"]
    if tt in _REACTIVE_PERSONALIZATION_TASKS:
        gt = record.get("groundtruth_preference_obj") or record.get("ground_truth_preference")
        if not gt or not gt.get("persona_item"):
            out.append(Finding(
                record["query_id"], tt, "high", "ground_truth_missing",
                "task is graded against a held-out preference, but none was attached",
                "regenerate",
            ))
        return out
    if tt in _RANKING_TASKS:
        inst = record.get("instance_full") or {}
        if tt == "personalized_feed_ranking":
            slate = inst.get("slate") or []
            if not slate:
                out.append(Finding(
                    record["query_id"], tt, "high", "ranking_slate_missing",
                    "slate-ranking task has empty slate",
                    "regenerate",
                ))
        elif tt in ("at_ai_directive_followup", "short_vs_long_term_lifecycle"):
            candidates = inst.get("candidates") or []
            if not candidates:
                out.append(Finding(
                    record["query_id"], tt, "high", "ranking_candidates_missing",
                    "ranking task has empty candidates list",
                    "regenerate",
                ))
        elif tt in ("personalized_recommendation", "personalized_search_ranking"):
            # Post-Batch-4 builder carries `candidates` + `held_out_idx`.
            # Pre-Batch-4 path (legacy): only `recent_pref_summary`.
            cands = inst.get("candidates") or []
            summary = inst.get("recent_pref_summary") or []
            if not cands and not summary:
                out.append(Finding(
                    record["query_id"], tt, "high", "ranking_signal_missing",
                    "personalized_recommendation has neither candidates nor recent_pref_summary",
                    "regenerate",
                ))
        return out
    return out


def check_reference_example_traceability(record: dict) -> list[Finding]:
    """Every grounded personalization query must trace to a real evidence row."""
    out: list[Finding] = []
    tt = record["task_type"]
    if tt not in _REACTIVE_PERSONALIZATION_TASKS:
        return out
    gt = record.get("groundtruth_preference_obj") or record.get("ground_truth_preference")
    if not gt:
        return out  # already flagged by ground_truth_presence
    ref = record.get("reference_example")
    if ref is None:
        out.append(Finding(
            record["query_id"], tt, "high", "reference_example_missing",
            f"held-out preference {gt.get('persona_item','')[:60]!r} has no traceable evidence row in the app JSONs",
            "regenerate",
        ))
    return out


def check_distractor_sanity(record: dict) -> list[Finding]:
    """Restraint tasks need a populated do-not-surface pool."""
    out: list[Finding] = []
    distractors = record.get("distractor_preferences") or []
    tt = record["task_type"]
    if tt == "over_personalization_distractor_reject":
        privacy = [d for d in distractors if d.get("role") == "privacy_flagged"]
        if len(privacy) < 2:
            out.append(Finding(
                record["query_id"], tt, "high", "distractor_pool_too_small",
                f"distractor_reject has {len(privacy)} privacy_flagged items (need ≥2)",
                "regenerate",
            ))
    elif tt in _RESTRAINT_TASKS and tt not in _DIVERSIFICATION_TASKS:
        # context_shift / preference_removal / over_personalization arms
        # need a real do-not-surface pool. Diversification tasks don't.
        if not distractors:
            inst = record.get("instance_full") or {}
            # C2 scenarios put the do-not-surface set in forbidden_items.
            forbidden = inst.get("forbidden_items") or []
            if not forbidden:
                out.append(Finding(
                    record["query_id"], tt, "medium", "distractor_pool_empty",
                    "restraint task has no distractor / forbidden_items pool — restraint metric trivially passes",
                    "regenerate",
                ))
    elif tt == "personalized_feed_ranking":
        inst = record.get("instance_full") or {}
        slate = inst.get("slate") or []
        if len(slate) < 5:
            out.append(Finding(
                record["query_id"], tt, "high", "slate_too_small",
                f"ranking task slate has {len(slate)} candidates (need ≥5 distractors + GT)",
                "regenerate",
            ))
    return out


def check_label_honesty_blind_check(record: dict) -> list[Finding]:
    """Generic queries leaking into the personalization bucket.

    The user's central concern: a query labeled
    `chatbot_proactive_personalization` may not actually require
    personalization. We use the existing blind_check_score the build
    pipeline already computes (0 = highly user-specific … 3 = generic
    — the actual threshold the harness uses to split arms is 2).
    """
    out: list[Finding] = []
    if record["task_type"] != "chatbot_proactive_personalization":
        return out
    inst = record.get("instance_full") or {}
    score = inst.get("blind_check_score")
    if score is not None and score >= 3:
        out.append(Finding(
            record["query_id"], record["task_type"], "high", "label_honesty_generic_query",
            f"blind_check_score={score} — a user-blind LLM can answer this query just as well; "
            f"this query does not require personalization and should not be in the personalization bucket",
            "regenerate",
        ))
    return out


def _hashtag_set(items: list) -> set[str]:
    out: set[str] = set()
    for h in items or []:
        if not h:
            continue
        out.add(str(h).lower().lstrip("#"))
    return out


def check_label_honesty_held_out_relevance(record: dict) -> list[Finding]:
    """Held-out preference's hashtags should overlap the query's
    source_hashtags. If they don't, the held-out is unrelated to the
    query and the personalization signal is fake."""
    out: list[Finding] = []
    if record["task_type"] != "chatbot_proactive_personalization":
        return out
    gt = record.get("groundtruth_preference_obj") or record.get("ground_truth_preference")
    inst = record.get("instance_full") or {}
    if not gt:
        return out
    held_tags = _hashtag_set(gt.get("source_hashtags"))
    src_tags = _hashtag_set(inst.get("source_hashtags"))
    if not held_tags or not src_tags:
        return out
    inter = held_tags & src_tags
    union = held_tags | src_tags
    jaccard = len(inter) / len(union) if union else 0.0
    if jaccard < 0.1:
        out.append(Finding(
            record["query_id"], record["task_type"], "medium",
            "label_honesty_held_out_unrelated",
            f"held-out hashtags {sorted(held_tags)[:3]} have Jaccard {jaccard:.2f} with "
            f"query hashtags {sorted(src_tags)[:3]} — held-out is topically unrelated to the query",
            "regenerate",
        ))
    return out


def check_label_honesty_restraint_trap(record: dict) -> list[Finding]:
    """over_personalization_chatbot_text: if the query already names the
    held-out topic explicitly, restraint is impossible."""
    out: list[Finding] = []
    if record["task_type"] != "over_personalization_chatbot_text":
        return out
    gt = record.get("groundtruth_preference_obj") or record.get("ground_truth_preference")
    text = (record.get("user_query") or "").lower()
    if not gt or not text:
        return out
    cat = (gt.get("category") or "").lower()
    if cat and len(cat) > 3 and cat in text:
        out.append(Finding(
            record["query_id"], record["task_type"], "medium",
            "label_honesty_restraint_trap",
            f"user_query already names category {cat!r}; restraint cannot be measured",
            "regenerate",
        ))
    return out


def check_proactive_ground(record: dict) -> list[Finding]:
    """Proactive recommendations need ground in recent activity. We only
    flag when the builder embeds the signal in the instance — for E3/E4
    the agent fetches state at runtime, so absence is by design."""
    out: list[Finding] = []
    tt = record["task_type"]
    if tt in _HOLLOW_BY_DESIGN:
        return out  # hollow at build time on purpose
    if tt not in {"daily_personalized_briefing", "personalized_search_ranking"}:
        return out
    inst = record.get("instance_full") or {}
    recent = inst.get("recent_pref_summary") or inst.get("top_prefs") or []
    if len(recent) < 3:
        out.append(Finding(
            record["query_id"], tt, "medium", "proactive_no_ground",
            f"only {len(recent)} prefs in the last-24h window — not enough signal "
            f"for a meaningful proactive recommendation",
            "regenerate",
        ))
    return out


_AGENTIC_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "agentic_auto_reply":            ("sender_id", "inbound_message"),
    "agentic_wrong_recipient_check": ("recipient_name", "draft"),
    "agentic_cross_app_repost":      ("source_post",),
    "agentic_send_post":             ("context",),
    "agentic_composed_post":         ("update",),
    "agentic_draft_audit":           ("draft",),
    "agentic_vague_refind":          ("topic",),
    "agentic_moment_recommendation": ("moment",),
}


def check_agentic_preconditions(record: dict) -> list[Finding]:
    """Agentic tasks that need build-time trigger context (e.g. T10
    auto_reply needs an inbound message; T17 wrong_recipient needs a
    draft + name collision). Tasks like T8/T15/T16/T18/T19 are
    deliberately hollow — the agent fetches data at runtime via MCP
    tools — so they're skipped here."""
    out: list[Finding] = []
    tt = record["task_type"]
    required = _AGENTIC_REQUIRED_FIELDS.get(tt)
    hint = _AGENTIC_PRECONDITION_HINTS.get(tt)
    if required is None:
        return out
    inst = record.get("instance_full") or {}
    missing = [f for f in required if not inst.get(f)]
    if missing:
        out.append(Finding(
            record["query_id"], tt, "medium", "agentic_precondition_missing",
            f"agentic task missing required field(s) {missing} ({hint})",
            "regenerate",
        ))
    return out


# Tasks that require concrete content in user_query — bracket-tag-only
# placeholders break LLM-eval (the LLM has nothing to respond to).
_AGENTIC_NEEDS_USER_QUERY_CONTENT = {
    "agentic_auto_reply",            # needs the inbound DM body
    "agentic_vague_refind",          # needs the topic
    "agentic_composed_post",         # needs the user's update
    "agentic_send_post",             # needs the chat context to dispatch
    "agentic_cross_app_repost",      # needs the source-post caption
    "agentic_wrong_recipient_check", # needs the draft
}


def check_inferior_length_match(record: dict) -> list[Finding]:
    """Foil within ±15% of gold's char count — otherwise a grader can win
    by simply picking the shorter response."""
    inf = record.get("inferior_response") or {}
    foil = (inf.get("text") or "").strip()
    gold = (record.get("example_response") or "").strip()
    if not foil or not gold:
        return []
    pct = abs(len(foil) - len(gold)) / max(len(gold), 1)
    if pct > 0.15:
        return [Finding(
            record["query_id"], record["task_type"], "medium",
            "inferior_length_mismatch",
            f"foil={len(foil)} vs gold={len(gold)} chars ({pct:.0%} diff)",
            "regenerate",
        )]
    return []


def check_user_query_has_content(record: dict) -> list[Finding]:
    """For agentic tasks where the agent must respond to specific user
    input, the user_query must carry actual content past the leading
    [bracket-tag]. A query like '[incoming DM from friend_2] ' (trailing
    space, no body) leaves the LLM nothing to reply to."""
    if record.get("task_type") not in _AGENTIC_NEEDS_USER_QUERY_CONTENT:
        return []
    text = (record.get("user_query") or "").strip()
    body = text
    if body.startswith("["):
        rb = body.find("]")
        if rb >= 0:
            body = body[rb + 1:].strip()
    if len(body) < 8:
        return [Finding(
            record["query_id"], record["task_type"], "high",
            "user_query_no_content",
            f"agentic user_query has no real content after bracket tag: {text[:80]!r}",
            "regenerate",
        )]
    return []


# Order matters only for output stability — all rules run on every record.
_PER_RECORD_RULES = (
    check_realism,
    check_ground_truth_presence,
    check_reference_example_traceability,
    check_distractor_sanity,
    check_label_honesty_blind_check,
    check_label_honesty_held_out_relevance,
    check_label_honesty_restraint_trap,
    check_proactive_ground,
    check_agentic_preconditions,
    check_inferior_length_match,
    check_user_query_has_content,
)


def run_per_record_rules(records: Iterable[dict]) -> list[Finding]:
    findings: list[Finding] = []
    for r in records:
        for rule in _PER_RECORD_RULES:
            findings.extend(rule(r))
    return findings


# ---------------------------------------------------------------------------
# Distribution checks (run over the full list)
# ---------------------------------------------------------------------------

# Per-task quotas live in evaluation.task_distribution — single source of
# truth for both build-time enforcement and post-hoc auditing.
from evaluation.task_distribution import TASK_TARGETS, DATA_DEPENDENT_TASKS  # noqa: E402


def distribution_findings(records: list[dict]) -> list[Finding]:
    """Per-task and cross-cutting distribution checks."""
    findings: list[Finding] = []
    counts = Counter(r["task_type"] for r in records)

    # Per-task: under-min OR over-max
    for tt, target in TASK_TARGETS.items():
        n = counts.get(tt, 0)
        lo, hi = target["min"], target["max"]
        # Data-dependent tasks (T17 collisions, E5 short-term, T16 group
        # threads) produce as many instances as the user's source data
        # supports — the floor is advisory, not enforced. Log as low
        # severity so automation pipelines don't treat per-user data
        # variation as a build failure.
        if tt in DATA_DEPENDENT_TASKS:
            if n < lo:
                findings.append(Finding(
                    "(distribution)", tt, "low", "distribution_data_dependent_short",
                    f"task_type {tt} has {n} instances (target {lo}-{hi}); "
                    f"capped by user's source data — not a pipeline bug",
                    "skip",
                ))
            continue
        if n < lo:
            findings.append(Finding(
                "(distribution)", tt, "high", "distribution_under_min",
                f"task_type {tt} has {n} instances; target floor is {lo}",
                "rebalance",
            ))
        elif n > hi:
            findings.append(Finding(
                "(distribution)", tt, "low", "distribution_over_max",
                f"task_type {tt} has {n} instances; target cap is {hi}",
                "rebalance",
            ))

    # Cross-cutting: query_kind balance
    qk_counts = Counter(r.get("query_kind") for r in records)
    for kind in ("user_query", "agentic_task", "proactive_recommendation", "proactive_assistance"):
        if qk_counts.get(kind, 0) < 5:
            findings.append(Finding(
                "(distribution)", "*", "medium", "query_kind_floor",
                f"query_kind {kind!r} has {qk_counts.get(kind, 0)} instances (floor: 5)",
                "rebalance",
            ))

    return findings


def check_dm_coverage(uid: str, backend_dir: str = "backend") -> list[Finding]:
    """Workstream K: audit the DM-thread engagement coverage for the user.

    Several agentic tasks (T8 dm_digest, T10 auto_reply, T16
    group_dm_summary) rely on DM threads. If most threads carry no
    user-side reaction, the agent's gold reply / digest will look
    stilted and the test grades poorly. We surface counts + warnings
    here, no LLM calls.
    """
    import json
    from pathlib import Path

    findings: list[Finding] = []
    user_dir = Path(backend_dir) / str(uid)

    # DM threads come from the `is_dm` flag on social-app events. Each
    # thread is keyed by `thread_id` (extension B field).
    threads: dict[str, dict] = {}
    for app in ("instagram", "facebook", "threads"):
        path = user_dir / f"{app}.json"
        if not path.exists():
            continue
        try:
            evs = json.loads(path.read_text())
        except Exception:
            continue
        for e in evs:
            if not e.get("is_dm"):
                continue
            tid = e.get("thread_id") or e.get("source_object_id", "")
            if not tid:
                continue
            t = threads.setdefault(tid, {
                "forwarded": False, "has_explicit_pos": False,
                "has_implicit_pos": False, "app": app,
            })
            # Forwarded post: a DM whose content carries shared post
            # fields and the sender is not the user.
            if not e.get("is_self_authored") and (
                e.get("source_hashtags") or
                (e.get("content") or {}).get("caption")
            ):
                t["forwarded"] = True
            itype = e.get("source_interaction_type", "")
            if itype == "explicit_positive":
                t["has_explicit_pos"] = True
            elif itype == "implicit_positive":
                t["has_implicit_pos"] = True

    n_total     = len(threads)
    n_forwarded = sum(1 for t in threads.values() if t["forwarded"])
    n_explicit  = sum(1 for t in threads.values() if t["has_explicit_pos"])
    n_implicit  = sum(1 for t in threads.values()
                      if t["has_implicit_pos"] and not t["has_explicit_pos"])
    n_engaged   = n_explicit + n_implicit
    n_unengaged = n_total - n_engaged

    # Stash summary as a finding so the markdown renderer can pick it
    # up alongside the regular findings.
    findings.append(Finding(
        f"(dm_coverage:{uid})", "*", "low", "dm_coverage_summary",
        f"total={n_total} forwarded={n_forwarded} "
        f"explicit_pos={n_explicit} implicit_pos={n_implicit} "
        f"unengaged={n_unengaged}",
        "info",
    ))

    if n_total > 0 and (n_unengaged / n_total) > 0.20:
        findings.append(Finding(
            f"(dm_coverage:{uid})", "*", "medium", "dm_threads_unengaged_high",
            f"{n_unengaged}/{n_total} ({n_unengaged/n_total:.0%}) DM threads "
            f"carry no user-side reaction — agentic DM tasks will look stilted",
            "review",
        ))

    if n_total > 0 and n_forwarded == 0:
        findings.append(Finding(
            f"(dm_coverage:{uid})", "*", "low", "forwarded_posts_missing",
            "no friend-forwarded-post DM threads in the user's data — "
            "unusual for a normal social graph",
            "review",
        ))

    if n_total > 0 and n_engaged == 0:
        findings.append(Finding(
            f"(dm_coverage:{uid})", "*", "high", "dm_threads_no_positive_signal",
            f"all {n_total} DM threads have zero positive engagement — "
            f"T8/T10/T16 grading will be unreliable",
            "regenerate",
        ))

    return findings


def task_count_table(records: list[dict]) -> list[tuple[str, int, int, int]]:
    """Return a per-task table: (task_type, count, target_min, target_max)
    sorted by count descending. Used by the Markdown report."""
    counts = Counter(r["task_type"] for r in records)
    rows: list[tuple[str, int, int, int]] = []
    for tt, n in counts.most_common():
        target = TASK_TARGETS.get(tt) or {"min": 0, "max": 0}
        rows.append((tt, n, target["min"], target["max"]))
    # Append targets that didn't show up at all (zero instances)
    for tt, target in TASK_TARGETS.items():
        if tt not in counts:
            rows.append((tt, 0, target["min"], target["max"]))
    return rows
