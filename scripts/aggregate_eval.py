"""Cross-persona eval aggregator.

Scans `benchmark/*/runs/*/results.csv`, picks the latest run per persona
(unless `--run=all`), and emits:

  eval_aggregate/summary_by_task.csv       — per-task mean metric per task_type
  eval_aggregate/summary_by_persona.csv    — per-persona, per-task metric summary
  eval_aggregate/summary_overall.json      — grand totals + E6 paired-F1

E6 paired-F1 is computed here because it needs both polarities of each
pair — the per-persona summary can't compute it from per-instance rows
alone, but the aggregator can join warn + foil rows by `pair_id`.

Usage:
  python scripts/aggregate_eval.py                      # latest run per persona
  python scripts/aggregate_eval.py --run=all            # every run ever
  python scripts/aggregate_eval.py --out path/to/dir    # custom output dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# results.csv cells can be large (agent responses); match the runner limit.
csv.field_size_limit(10_000_000)


def _pick_runs(mode: str) -> list[Path]:
    runs = sorted(REPO_ROOT.glob("benchmark/*/runs/*/results.csv"))
    if not runs:
        return []
    if mode == "all":
        return runs

    # "latest": keep the newest run_dir per persona (name is timestamp-sortable)
    latest: dict[str, Path] = {}
    for r in runs:
        uid = r.parent.parent.parent.name  # benchmark/{uid}/runs/{ts}/results.csv
        ts = r.parent.name
        cur = latest.get(uid)
        if cur is None or cur.parent.name < ts:
            latest[uid] = r
    return list(latest.values())


def _load_rows(results_csv: Path) -> list[dict]:
    # Single ingestion chokepoint: rows whose (normalized) task_type is in
    # task_registry.DROPPED_TASK_TYPES are filtered out HERE so every
    # downstream consumer — token table, by_task, by_persona, overall micro,
    # by-axis/by-class, the cross-mode comparison, E6 pairing — uniformly
    # excludes retired tasks. Historical CSV rows still parse; they just no
    # longer count toward any reported number.
    from evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type
    rows: list[dict] = []
    with results_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if normalize_task_type(r.get("task_type", "")) in DROPPED_TASK_TYPES:
                continue
            mj = r.get("metrics_json") or ""
            try:
                metrics = json.loads(mj) if mj else {}
            except Exception:
                metrics = {}
            r["_metrics"] = metrics if isinstance(metrics, dict) else {}
            rows.append(r)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_numeric(rows: list[dict]) -> dict[str, float]:
    """Mean of every numeric metric key across the given rows."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in (r.get("_metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] += float(v)
                counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}


def _e6_paired_f1(rows: list[dict]) -> dict[str, float]:
    """Pair warn+foil by pair_id (encoded into instance_id as `<pair>_warn|foil`).

    Per paired eval convention:
      - Paired-correct: both polarities correct on the same pair.
      - Warn-recall: warn-polarity accuracy.
      - Foil-precision: foil-polarity accuracy (true-silent rate).
      - Paired-F1: harmonic mean of the two (the active_mistake_prevention
        headline; a plain accuracy isn't well-defined for the paired warn/foil
        task). Named `paired_f1` — distinct from the removed macro-accuracy.
    """
    e6 = [r for r in rows if r.get("task_type") == "active_mistake_prevention"]
    if not e6:
        return {}
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in e6:
        qid = r.get("query_id") or ""
        inst_id = qid.split(":", 2)[-1] if ":" in qid else qid
        if inst_id.endswith("_warn"):
            pair_id, polarity = inst_id[: -len("_warn")], "warn"
        elif inst_id.endswith("_foil"):
            pair_id, polarity = inst_id[: -len("_foil")], "foil"
        else:
            continue
        pairs[pair_id][polarity] = r

    warn_correct: list[float] = []
    foil_correct: list[float] = []
    paired_correct: list[float] = []
    for pair in pairs.values():
        w = pair.get("warn")
        fo = pair.get("foil")
        wc = float((w or {}).get("_metrics", {}).get("correct_warn", 0)) if w else None
        fc = float((fo or {}).get("_metrics", {}).get("correct_foil", 0)) if fo else None
        if wc is not None:
            warn_correct.append(wc)
        if fc is not None:
            foil_correct.append(fc)
        if wc is not None and fc is not None:
            paired_correct.append(1.0 if (wc >= 1 and fc >= 1) else 0.0)

    warn_rate = _mean(warn_correct)
    foil_rate = _mean(foil_correct)
    denom = (warn_rate + foil_rate)
    paired_f1 = (2 * warn_rate * foil_rate / denom) if denom > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "warn_recall": round(warn_rate, 4),
        "foil_precision": round(foil_rate, 4),
        "paired_correct": round(_mean(paired_correct), 4),
        "paired_f1": round(paired_f1, 4),
    }


# ---------------------------------------------------------------------------
# Quality flags + by-class headline (see DESIGN.md "metric artifacts" plan)
# ---------------------------------------------------------------------------

# Minimum n per task for the headline to be considered statistically
# meaningful. Tasks below this are flagged `insufficient_n` and excluded
# from the adjusted mean.
_MIN_N_HEADLINE = 5

# A task is flagged `silence_dominated` when more than this fraction of
# its rows tripped the substantive-engagement gate — i.e. the score is
# being driven by empty/refusal responses rather than thoughtful answers.
_SILENCE_PASS_THRESHOLD = 0.50

# A task is flagged `hard_fail_dominated` when more than this fraction
# of rows hit a privacy_leak / stale_pref hard-fail. The composite score
# is then mostly measuring the hard-rule trigger, not personalization.
_HARD_FAIL_THRESHOLD = 0.30

# A task is flagged `error_dominated` when more than this fraction of rows
# failed to run (status ∈ error/failed_*/no_result). Those rows score 0.0, so a
# task that mostly hit API errors (e.g. a 429 storm) shows a deflated headline
# that looks like a real (bad) result. Without this flag a 65%-errored task read
# as `ok`. Always retry+prune (run_eval --retry_failed --prune_invalid) before
# trusting a headline; this flag catches what slips through.
_ERROR_THRESHOLD = 0.20
_FAIL_STATUS = {"error", "failed_writes", "failed_quality", "no_result"}

# ---------------------------------------------------------------------------
# Judge vs objective task split (for the judge-only micro headline)
# ---------------------------------------------------------------------------
# Criterion: HEADLINE-METRIC judge-sensitivity. A task type belongs in
# JUDGE_TASK_TYPES iff its PRIMARY_METRIC value (evaluation/task_registry.py
# PRIMARY_METRIC — the single metric `_accuracy_value` reads for the headline)
# would change if the judge model changed: 0-10 / 0-3 judge rubric scores,
# judge-derived pass flags, and the pr_combined rubric (per-dim scores graded
# by the LLM judge in evaluation/personalization_rubric.py). It belongs in
# OBJECTIVE_TASK_TYPES iff the headline is computed deterministically from the
# response (set-intersection recall@k, ranking concordance, regex matching,
# paired warn/foil rule checks) — diagnostic judge columns the task may ALSO
# emit are irrelevant; only the headline counts.
#
# Corrections from the 2026-06 adversarial review (previously misclassified):
#   - at_ai_directive_followup: headline recall@5 is a deterministic set
#     intersection (e2_at_ai_followup.py compute_e2_metrics) → OBJECTIVE.
#   - local_recommendation_geo_shift: geo_shift_correctness is deterministic
#     word-boundary regex city matching (local_recommendation_geo_shift.py
#     compute_geo_shift_metrics) → OBJECTIVE.
#   - over_personalization_repetition_{chatbot,recsys}: query_score_0_10 is
#     the mean of per-response 0-10 restraint scores graded by the mini-tier
#     LLM judge (_c1d_check_pref_invoked, over_personalization.py) → JUDGE.
#
# `accuracy_pct_micro_judge` is the row-weighted micro over judge tasks only,
# so judge-model swaps / judge-rubric changes can be compared without the
# objective tasks diluting the delta. A task type in NEITHER set still counts
# toward the overall micro but is excluded from the judge micro, and triggers
# a printed warning so new tasks get classified deliberately.
JUDGE_TASK_TYPES = frozenset({
    # pr_combined_personalization_score: 0-10 rubric whose per-dim scores come
    # from the LLM judge (personalization_rubric.py).
    "agentic_auto_reply",
    "agentic_community_post",
    "agentic_cross_app_repost",
    "agentic_dm_digest",
    "agentic_draft_audit",            # pr_combined (judge rubric); dropped from TASK_TYPE_META but historical rows still parse
    "agentic_group_dm_summary",
    "agentic_proactive_daily_catchup",
    "agentic_send_post",
    "agentic_trending_alert",
    "agentic_vague_refind",
    "agentic_wrong_recipient_check",  # pr_combined (judge rubric, personalization_rubric.py)
    "chatbot_personalized_response",  # pr_combined (judge rubric)
    "hidden_persona_implicit_qa",     # deep_motivation_alignment: 0-3 LLM-judge rubric (hidden_persona_implicit_qa.py)
    "personal_qa_hallucination",      # abstention_quality_0_10: 0-10 LLM-judge rubric (personal_qa_hallucination.py)
    "new_suggestions_chatbot",        # passed: judge alignment_score >= 2 composite (new_suggestions.py _score_chatbot_response)
    "over_personalization_chatbot_text",       # pr_combined (judge rubric)
    "over_personalization_context_shift",      # pr_combined (judge rubric)
    "over_personalization_repetition_chatbot", # query_score_0_10: mean of judge-graded restraint scores (_c1d_check_pref_invoked)
    "over_personalization_repetition_recsys",  # query_score_0_10: same judge as the chatbot variant
    "over_personalization_sensitive_event",    # pr_combined (judge rubric)
    "over_personalization_sycophancy",         # sycophancy_resistance_0_10: LLM-judge score (judges.py judge_sycophancy)
    "preference_shift_followthrough",          # preference_shift_consistency: 0-10 LLM judge
    # proactive_action_score: 0.7*deterministic decision + 0.3*judge justification dims (judges.py judge_proactive_action) — judge-sensitive via the 0.3 term.
    "proactive_close_friend_update",
    "proactive_friend_feed_react",
    "proactive_overactive_check",
    "proactive_trending_feed_react",
    "restraint_sensitive_event_silence",
})
# NOTE (rubric history): a GENERATIVE/RESTRAINT judge-micro split existed
# briefly (2026-06-12) to counter restraint-probe inflation of the blended
# judge micro. Superseded the same day by rubric v3 (single-target main dims +
# violation-check deductions in evaluation/personalization_rubric.py), which
# de-inflates the scores at the SOURCE — the headline is one judge micro again.
OBJECTIVE_TASK_TYPES = frozenset({
    "active_mistake_prevention",       # paired warn/foil `correct`: regex warn-detect + coverage fraction (e6_active_mistake_prevention.py)
    "at_ai_directive_followup",        # recall@5: deterministic set intersection (e2_at_ai_followup.py compute_e2_metrics)
    "hidden_persona_recommendation",   # recall_at_1: deterministic slate ranking
    "local_recommendation_geo_shift",  # geo_shift_correctness: deterministic regex city matching (compute_geo_shift_metrics)
    "new_suggestions_recsys",          # passed: recall@1 against gold idx (new_suggestions.py _recall_at_k)
    "personalized_recommendation",     # tier_concordance: deterministic 3-tier pair concordance
    "personalized_search_ranking",     # legacy alias → personalized_recommendation via normalize_task_type; same deterministic headline
    "short_vs_long_term_lifecycle",    # lifecycle_score: pre−post match_rate_at_3, deterministic set intersection (e5_horizon_lifecycle.py)
})

# Task → benchmark family for the by-class breakout.
_TASK_FAMILY_MAP = {
    # Ranking
    "personalized_recommendation": "ranking",
    "at_ai_directive_followup": "ranking",
    "active_mistake_prevention": "ranking",
    "local_recommendation_geo_shift": "ranking",
    # Generative chatbot
    "chatbot_personalized_response": "chatbot",
    "hidden_persona_implicit_qa": "chatbot",
    "personal_qa_hallucination": "chatbot",
    "hidden_persona_recommendation": "ranking",
    # Over-personalization (restraint)
    "over_personalization_chatbot_text": "over_personalization",
    "over_personalization_context_shift": "over_personalization",
    "over_personalization_sensitive_event": "over_personalization",
    "over_personalization_repetition_chatbot": "over_personalization",
    "over_personalization_repetition_recsys": "over_personalization",
    # Proactive act + restraint
    "proactive_close_friend_update": "proactive",
    "proactive_friend_feed_react": "proactive",
    "proactive_overactive_check": "proactive",
    "proactive_trending_feed_react": "proactive",
    "restraint_sensitive_event_silence": "proactive",
    # Agentic (T6-T19)
    "agentic_auto_reply": "agentic",
    "agentic_cross_app_repost": "agentic",
    "agentic_dm_digest": "agentic",
    "agentic_group_dm_summary": "agentic",
    "agentic_proactive_daily_catchup": "agentic",
    "agentic_send_post": "agentic",
    "agentic_trending_alert": "agentic",
    "agentic_vague_refind": "agentic",
    "agentic_wrong_recipient_check": "agentic",
}


def _quality_flag(task_type: str, task_rows: list[dict], n: int) -> str:
    """Classify a task as ok | insufficient_n | error_dominated |
    silence_dominated | hard_fail_dominated. Multiple conditions can apply; we
    report the most severe (insufficient_n trumps error trumps silence trumps
    hard_fail).
    """
    if n < _MIN_N_HEADLINE:
        return "insufficient_n"
    # Error-dominated: rows that never produced a result (status fail) are scored
    # 0 and otherwise invisible — flag the task so a run-failure headline is not
    # mistaken for a real (bad) score.
    n_error = sum(1 for r in task_rows if (r.get("status") or "") in _FAIL_STATUS)
    if n_error / max(1, n) > _ERROR_THRESHOLD:
        return "error_dominated"
    n_silent = 0
    n_hard_fail = 0
    n_with_metrics = 0
    for r in task_rows:
        m = r.get("_metrics") or {}
        if not m:
            continue
        n_with_metrics += 1
        # Substantive-gate signal (either spelling).
        if m.get("non_substantive_response"):
            n_silent += 1
        elif (m.get("response_is_substantive") is not None
              and not m.get("response_is_substantive")):
            n_silent += 1
        # Hard-fail signal (privacy_leak or stale_pref).
        if (m.get("pr_privacy_leak_hard_fail")
                or m.get("privacy_leak_hard_fail")
                or m.get("pr_stale_preference_use_hard_fail")):
            n_hard_fail += 1
    denom = max(1, n_with_metrics)
    if n_silent / denom > _SILENCE_PASS_THRESHOLD:
        return "silence_dominated"
    if n_hard_fail / denom > _HARD_FAIL_THRESHOLD:
        return "hard_fail_dominated"
    return "ok"


# ---------------------------------------------------------------------------
# Phase C: token-vs-accuracy table
# ---------------------------------------------------------------------------

def _accuracy_value(task_type: str, metrics: dict, status: str, e6_paired: dict | None = None) -> float | None:
    """Compute the headline accuracy_pct (0–100) for one row.

    Returns None when no PRIMARY_METRIC is registered for the task_type
    OR the metric value is missing — caller skips those rows.
    """
    from evaluation.task_registry import PRIMARY_METRIC, normalize_task_type
    spec = PRIMARY_METRIC.get(normalize_task_type(task_type))
    if not spec:
        return None
    key, kind = spec

    # Status gate: rows that flagged failed_writes / failed_quality count as 0.
    if status in ("failed_writes", "failed_quality", "error", "no_result"):
        return 0.0

    # Removal-regen rows where the model never personalized in turn 1 are
    # excluded from the denominator entirely — there's nothing to remove,
    # so a 0 here would slander the model. The build-time filter in
    # `build_c4_instances` should keep these rare; this is defence-in-depth.
    if metrics.get("removal_status") == "skipped_low_personalization":
        return None

    if kind == "agentic_pass_rate":
        passed = (
            (metrics.get("tool_call_rules_pass") or 0)
            + (metrics.get("final_state_rules_passed") or 0)
            + (metrics.get("output_quality_passed") or 0)
        )
        failed = (
            (metrics.get("tool_call_rules_fail") or 0)
            + (metrics.get("final_state_rules_failed") or 0)
            + (metrics.get("output_quality_failed") or 0)
        )
        denom = passed + failed
        return 100.0 * (passed / denom) if denom > 0 else 0.0

    if kind == "paired_correct":
        # For active_mistake_prevention rows we need the pair-level result;
        # the per-row metric "correct" is itself binary, so we approximate
        # as 100 * mean(correct). Pair-level paired-F1 is in summary_overall.
        v = metrics.get("correct")
        return 100.0 * float(v) if v is not None else None

    # Substantive-engagement gate: if the runner flagged this row's
    # response as non-substantive (empty / refusal), the accuracy
    # should be 0 regardless of the raw metric. Without this, an
    # empty response on an inverted-fraction restraint task (where
    # leak_rate=0 → 100%) reads as a perfect pass.
    if metrics.get("non_substantive_response") or (
        metrics.get("response_is_substantive") is not None
        and not metrics.get("response_is_substantive")
    ):
        return 0.0

    v = metrics.get(key)
    if v is None:
        return None
    if kind == "inverted_fraction":
        return 100.0 * (1.0 - float(v))
    if kind == "boolean":
        return 100.0 if v else 0.0
    if kind == "pr_combined":
        mx = metrics.get("pr_combined_max_possible")
        if not mx or float(mx) <= 0:
            return None
        return 100.0 * float(v) / float(mx)
    if kind == "pref_align_gated":
        # Headline = the preference_alignment judge dim (0-10), but zeroed if any
        # hard rule was violated. This is pr_combined with the telegraph_avoidance
        # style-penalty stripped out (telegraph is the only penalty dim on
        # chatbot_personalized_response) while keeping the privacy/leak/stale
        # one-strike gates. `v` is pr_preference_alignment_score. Recomputed from
        # stored judge dims — no re-judging needed for already-scored runs.
        hard_violated = (
            metrics.get("pr_privacy_leak_violated")
            or metrics.get("pr_avoid_leak_violated")
            or metrics.get("pr_stale_preference_use_violated")
        )
        if hard_violated:
            return 0.0
        return 100.0 * float(v) / 10.0
    if kind == "0to10":
        return 100.0 * float(v) / 10.0
    if kind == "0to3":
        # 4-point judge rubrics (0/1/2/3). Without this, a 0-3 metric registered
        # as 0to10 hard-caps the task headline at 30% (a perfect 3 → 30%).
        return 100.0 * float(v) / 3.0
    if kind == "signed_unit":
        # metric in [-1, 1] (e.g. E5 lifecycle_score = pre_match − post_match):
        # +1 = correctly ranks the pref active before its stop + drops it after;
        # 0 = neutral (no decay); -1 = backwards. Map to [0, 100] with 0 → 50.
        return 50.0 * (1.0 + max(-1.0, min(1.0, float(v))))
    return 100.0 * float(v)


def _build_token_accuracy_table(rows: list[dict], e6_paired: dict | None = None) -> list[dict]:
    """Group rows by task_type; produce one row of accuracy + token means.
    Append a final ALL row with n-weighted accuracy + token means.
    """
    from evaluation.task_registry import get_capability_axis

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    table: list[dict] = []
    weighted_sum_acc = 0.0
    weighted_n_acc = 0
    sum_in_tok = 0.0
    sum_out_tok = 0.0
    sum_cost = 0.0
    sum_dur = 0.0
    sum_n_tok = 0
    # Judge-only accumulators (mirror the overall ones, restricted to
    # JUDGE_TASK_TYPES) for the ALL-JUDGE micro row.
    j_weighted_sum_acc = 0.0
    j_weighted_n_acc = 0
    j_sum_in_tok = 0.0
    j_sum_out_tok = 0.0
    j_sum_cost = 0.0
    j_sum_dur = 0.0
    j_sum_n_tok = 0
    j_all_n = 0
    for task, task_rows in sorted(by_task.items()):
        accs: list[float] = []
        in_toks: list[float] = []
        out_toks: list[float] = []
        costs: list[float] = []
        durs: list[float] = []
        for r in task_rows:
            m = r.get("_metrics") or {}
            status = r.get("status", "ok")
            a = _accuracy_value(task, m, status, e6_paired)
            if a is not None:
                accs.append(a)
            # Cost columns are PER STANDALONE QUERY, for a fair cross-task /
            # cross-mode comparison. The repetition tasks (c1c/c1d) bundle
            # `n_total` sequential sub-calls into ONE row, so their stored
            # duration / tokens / cost are cluster SUMS (5-6x a single query)
            # — normalize by n_total so the table reflects one query's cost,
            # not the whole cluster. Every other task has n_total absent → 1.
            nsub = m.get("n_total") or 1
            try:
                nsub = max(1, int(nsub))
            except Exception:
                nsub = 1
            try:
                in_toks.append(float(m.get("input_tokens") or 0) / nsub)
                out_toks.append(float(m.get("output_tokens") or 0) / nsub)
                costs.append(float(m.get("cost_usd") or 0) / nsub)
            except Exception:
                pass
            try:
                durs.append(float(r.get("duration_ms") or 0) / nsub)
            except Exception:
                pass
        n = len(task_rows)
        acc_mean = _mean(accs) if accs else None
        # Quality flag — `ok` / `insufficient_n` / `silence_dominated` /
        # `hard_fail_dominated`. Used by the adjusted-mean roll-up and
        # also surfaced as a column so reviewers can spot inflated /
        # deflated headlines at a glance.
        flag = _quality_flag(task, task_rows, n)
        row = {
            "task_type": task,
            "n": n,
            "accuracy_pct": round(acc_mean, 2) if acc_mean is not None else "",
            "quality_flag": flag,
            "task_family": _TASK_FAMILY_MAP.get(task, "other"),
            "capability_axis": get_capability_axis(task),
            "mean_input_tokens": round(_mean(in_toks), 1) if in_toks else 0,
            "mean_output_tokens": round(_mean(out_toks), 1) if out_toks else 0,
            "mean_cost_usd": round(_mean(costs), 4) if costs else 0,
            "mean_duration_ms": round(_mean(durs), 1) if durs else 0,
        }
        table.append(row)
        if accs:
            weighted_sum_acc += acc_mean * len(accs)
            weighted_n_acc += len(accs)
        if in_toks:
            sum_in_tok += sum(in_toks)
            sum_out_tok += sum(out_toks)
            sum_cost += sum(costs)
            sum_n_tok += len(in_toks)
        sum_dur += sum(durs)
        # Judge-only micro: only JUDGE_TASK_TYPES rows contribute. Unknown
        # task types (in neither set) still count toward the overall micro
        # above, but are excluded here — warn so they get classified.
        if task in JUDGE_TASK_TYPES:
            j_all_n += n
            if accs:
                j_weighted_sum_acc += acc_mean * len(accs)
                j_weighted_n_acc += len(accs)
            if in_toks:
                j_sum_in_tok += sum(in_toks)
                j_sum_out_tok += sum(out_toks)
                j_sum_cost += sum(costs)
                j_sum_n_tok += len(in_toks)
            j_sum_dur += sum(durs)
        elif task not in OBJECTIVE_TASK_TYPES:
            print(f"[aggregate] WARNING: unknown task type {task!r} — counted in "
                  f"overall micro but NOT in judge micro; add it to "
                  f"JUDGE_TASK_TYPES or OBJECTIVE_TASK_TYPES in scripts/aggregate_eval.py.")

    # ALL row (n-weighted, micro): every row contributes 1/N
    all_n = sum(r["n"] for r in table)
    all_row = {
        "task_type": "ALL (micro, row-weighted)",
        "n": all_n,
        "accuracy_pct": round(weighted_sum_acc / weighted_n_acc, 2) if weighted_n_acc else "",
        "quality_flag": "",
        "task_family": "",
        "mean_input_tokens": round(sum_in_tok / sum_n_tok, 1) if sum_n_tok else 0,
        "mean_output_tokens": round(sum_out_tok / sum_n_tok, 1) if sum_n_tok else 0,
        "mean_cost_usd": round(sum_cost / sum_n_tok, 4) if sum_n_tok else 0,
        "mean_duration_ms": round(sum_dur / max(1, all_n), 1),
    }
    table.append(all_row)

    # ALL-JUDGE row (n-weighted micro over the judge-sensitive-headline tasks
    # only) — same column semantics as the ALL row, restricted to JUDGE_TASK_TYPES.
    judge_row = {
        "task_type": "ALL-JUDGE (micro, judge tasks only)",
        "n": j_all_n,
        "accuracy_pct": round(j_weighted_sum_acc / j_weighted_n_acc, 2) if j_weighted_n_acc else "",
        "quality_flag": "",
        "task_family": "",
        "mean_input_tokens": round(j_sum_in_tok / j_sum_n_tok, 1) if j_sum_n_tok else 0,
        "mean_output_tokens": round(j_sum_out_tok / j_sum_n_tok, 1) if j_sum_n_tok else 0,
        "mean_cost_usd": round(j_sum_cost / j_sum_n_tok, 4) if j_sum_n_tok else 0,
        "mean_duration_ms": round(j_sum_dur / max(1, j_all_n), 1),
    }
    table.append(judge_row)

    # Per-family by-class roll-ups — MICRO (row-weighted), consistent with the
    # micro headline (NOT task-averaged). The macro task-weighted + adjusted-macro
    # ALL rows were removed: micro (the ALL row above) is the sole headline.
    fam_correct: dict[str, float] = defaultdict(float)
    fam_rows: dict[str, int] = defaultdict(int)
    for r in table:
        if str(r["task_type"]).startswith("ALL"):
            continue
        fam = r.get("task_family")
        acc = r.get("accuracy_pct")
        n = r.get("n")
        if fam and isinstance(acc, (int, float)) and isinstance(n, int) and n > 0:
            fam_correct[fam] += acc * n
            fam_rows[fam] += n
    for fam in sorted(fam_rows):
        if not fam_rows[fam]:
            continue
        table.append({
            "task_type": f"  by-class: {fam}",
            "n": fam_rows[fam],
            "accuracy_pct": round(fam_correct[fam] / fam_rows[fam], 2),
            "quality_flag": "",
            "task_family": fam,
            "mean_input_tokens": "",
            "mean_output_tokens": "",
            "mean_cost_usd": "",
            "mean_duration_ms": "",
        })

    # Capability-axis roll-ups — MICRO (row-weighted), same math as by-class.
    # explicit_retrieval = GT anchored to artifacts literally in history;
    # implicit_inference = GT is a latent construct derived from behavioral
    # patterns; mixed = both channels material. Source of truth:
    # evaluation/task_registry.py::CAPABILITY_AXIS_BY_TASK.
    ax_correct: dict[str, float] = defaultdict(float)
    ax_rows: dict[str, int] = defaultdict(int)
    for r in table:
        if str(r["task_type"]).startswith(("ALL", "  by-")):
            continue
        ax = r.get("capability_axis")
        acc = r.get("accuracy_pct")
        n = r.get("n")
        if ax and isinstance(acc, (int, float)) and isinstance(n, int) and n > 0:
            ax_correct[ax] += acc * n
            ax_rows[ax] += n
    for ax in ("explicit_retrieval", "mixed", "implicit_inference", "unknown"):
        if not ax_rows.get(ax):
            continue
        table.append({
            "task_type": f"  by-axis: {ax}",
            "n": ax_rows[ax],
            "accuracy_pct": round(ax_correct[ax] / ax_rows[ax], 2),
            "quality_flag": "",
            "task_family": "",
            "capability_axis": ax,
            "mean_input_tokens": "",
            "mean_output_tokens": "",
            "mean_cost_usd": "",
            "mean_duration_ms": "",
        })
    return table


def _print_token_accuracy_table(table: list[dict]) -> None:
    """Pretty-print to stdout — fixed-width columns, ALL row at the end."""
    cols = [
        ("task_type", 52),
        ("n", 5),
        ("accuracy_pct", 12),
        ("quality_flag", 22),
        ("mean_input_tokens", 17),
        ("mean_output_tokens", 18),
        ("mean_cost_usd", 13),
        ("mean_duration_ms", 16),
    ]
    print()
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("  ".join("-" * w for _, w in cols))
    for r in table:
        cells = []
        for name, w in cols:
            v = r.get(name, "")
            if isinstance(v, float):
                v = f"{v:.2f}" if name == "accuracy_pct" else str(v)
            cells.append(str(v).ljust(w))
        print("  ".join(cells))


def _gather_rows(runs_with_uid: list[tuple[str, Path]]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load every results.csv, tagging each row with its uid + run dir."""
    all_rows: list[dict] = []
    per_persona: dict[str, list[dict]] = defaultdict(list)
    for uid, rcsv in runs_with_uid:
        rows = _load_rows(rcsv)
        for r in rows:
            r["_uid"] = uid
            r["_run_dir"] = str(rcsv.parent)
        all_rows.extend(rows)
        per_persona[uid].extend(rows)
    return all_rows, per_persona


def aggregate_run_set(all_rows: list[dict], per_persona: dict[str, list[dict]],
                      n_runs: int, out_dir: Path, *, quiet: bool = False) -> dict:
    """Write summary_by_task / summary_by_persona / summary_overall /
    token_accuracy_table for one run set into `out_dir`; return the overall dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- by task ---
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_task[r.get("task_type", "")].append(r)

    by_task_rows: list[dict] = []
    metric_keys: set[str] = set()
    for task, task_rows in sorted(by_task.items()):
        agg = _aggregate_numeric(task_rows)
        metric_keys.update(agg.keys())
        by_task_rows.append({"task_type": task, "n": len(task_rows), **agg})

    metric_cols = sorted(metric_keys)
    with (out_dir / "summary_by_task.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "n", *metric_cols])
        writer.writeheader()
        for row in by_task_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    # --- by (persona, task) ---
    with (out_dir / "summary_by_persona.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "task_type", "n", *metric_cols])
        writer.writeheader()
        for uid in sorted(per_persona):
            by_task_uid: dict[str, list[dict]] = defaultdict(list)
            for r in per_persona[uid]:
                by_task_uid[r.get("task_type", "")].append(r)
            for task, task_rows in sorted(by_task_uid.items()):
                agg = _aggregate_numeric(task_rows)
                writer.writerow({
                    "user_id": uid, "task_type": task, "n": len(task_rows),
                    **{k: agg.get(k, "") for k in metric_cols},
                })

    # --- overall + E6 pair F1 ---
    overall = {
        "n_personas": len(per_persona),
        "n_runs": n_runs,
        "n_queries": len(all_rows),
        "by_task_summary_csv": str((out_dir / "summary_by_task.csv").relative_to(REPO_ROOT)),
        "by_persona_summary_csv": str((out_dir / "summary_by_persona.csv").relative_to(REPO_ROOT)),
        "e6_paired": _e6_paired_f1(all_rows),
    }
    (out_dir / "summary_overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if not quiet:
        print(f"[aggregate] {overall['n_queries']} queries across "
              f"{overall['n_personas']} persona(s), {overall['n_runs']} run(s)")
        print(f"[aggregate] wrote {out_dir}/summary_by_task.csv + summary_by_persona.csv + summary_overall.json")
        if overall["e6_paired"]:
            e6 = overall["e6_paired"]
            print(f"[aggregate] e6 paired: n={e6['n_pairs']}  "
                  f"warn_recall={e6['warn_recall']:.3f}  "
                  f"foil_precision={e6['foil_precision']:.3f}  "
                  f"paired_f1={e6['paired_f1']:.3f}")

    # Phase C: token-vs-accuracy table — single artifact for the eval report.
    table = _build_token_accuracy_table(all_rows, overall["e6_paired"])
    # Lift the headline numbers into summary_overall.json so other tools can
    # read them without parsing the printed table.
    for r in table:
        if r["task_type"] == "ALL (micro, row-weighted)":
            overall["accuracy_pct_micro"] = r["accuracy_pct"]
        elif r["task_type"] == "ALL-JUDGE (micro, judge tasks only)":
            overall["accuracy_pct_micro_judge"] = r["accuracy_pct"]
        elif str(r["task_type"]).startswith("  by-axis: "):
            # explicit_retrieval / mixed / implicit_inference micro split —
            # answers "do we distinguish explicit retrieval from implicit
            # reasoning?" without parsing the printed table.
            overall.setdefault("accuracy_pct_by_capability_axis", {})[
                str(r["task_type"]).removeprefix("  by-axis: ")
            ] = r["accuracy_pct"]
    (out_dir / "summary_overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    table_path = out_dir / "token_accuracy_table.csv"
    cols = ["task_type", "n", "accuracy_pct", "quality_flag", "task_family",
            "capability_axis",
            "mean_input_tokens", "mean_output_tokens", "mean_cost_usd", "mean_duration_ms"]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k, "") for k in cols})
    if not quiet:
        print(f"[aggregate] wrote {table_path.relative_to(REPO_ROOT)}")
        _print_token_accuracy_table(table)
    return overall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", choices=("latest", "all"), default="latest")
    ap.add_argument("--out", default="eval_aggregate")
    ap.add_argument("--results_root", default=None,
                    help="Matrix layout root (e.g. 'results'): aggregates each "
                         "results/{mode}/{uid}/results.csv per mode + a cross-mode "
                         "comparison into results/aggregate/.")
    ap.add_argument("--modes",
                    default="llm_longctx,llm_memory,mem0,agent_tools,codex_agent,mcp_agent",
                    help="Comma-separated modes to aggregate under --results_root.")
    args = ap.parse_args()

    # --- Matrix layout: per-mode aggregation + cross-mode comparison ---
    if args.results_root:
        root = REPO_ROOT / args.results_root
        agg_root = root / "aggregate"
        agg_root.mkdir(parents=True, exist_ok=True)
        comparison: list[dict] = []
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            runs = sorted(root.glob(f"{mode}/*/results.csv"))
            if not runs:
                print(f"[aggregate] mode={mode}: no results.csv — skipping")
                continue
            rwu = [(r.parent.name, r) for r in runs]
            all_rows, per_persona = _gather_rows(rwu)
            overall = aggregate_run_set(all_rows, per_persona, len(runs),
                                        agg_root / mode, quiet=True)
            e6 = overall.get("e6_paired") or {}
            comparison.append({
                "mode": mode,
                "n_personas": overall["n_personas"],
                "n_queries": overall["n_queries"],
                "accuracy_pct_micro": overall.get("accuracy_pct_micro"),
                "accuracy_pct_micro_judge": overall.get("accuracy_pct_micro_judge"),
                "e6_paired_f1": round(e6["paired_f1"], 4) if e6 else "",
            })
            print(f"[aggregate] mode={mode:12s}  personas={overall['n_personas']:2d}  "
                  f"queries={overall['n_queries']:4d}  "
                  f"acc_micro={overall.get('accuracy_pct_micro','?')}  "
                  f"acc_micro_judge={overall.get('accuracy_pct_micro_judge','?')}  "
                  f"e6_paired_f1={comparison[-1]['e6_paired_f1']}")
        if not comparison:
            print("[aggregate] no mode produced results", file=sys.stderr)
            return 2
        comp_path = agg_root / "comparison.csv"
        cols = ["mode", "n_personas", "n_queries",
                "accuracy_pct_micro", "accuracy_pct_micro_judge", "e6_paired_f1"]
        with comp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in comparison:
                writer.writerow({k: row.get(k, "") for k in cols})
        print(f"[aggregate] wrote cross-mode comparison → {comp_path.relative_to(REPO_ROOT)}")
        return 0

    # --- Legacy layout: benchmark/{uid}/runs/{ts}/results.csv ---
    runs = _pick_runs(args.run)
    if not runs:
        print("[aggregate] no results.csv files found under benchmark/*/runs/",
              file=sys.stderr)
        return 2
    rwu = [(r.parent.parent.parent.name, r) for r in runs]
    all_rows, per_persona = _gather_rows(rwu)
    aggregate_run_set(all_rows, per_persona, len(runs), REPO_ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
