#!/usr/bin/env python3
"""Stage A: deterministic per-user query stats + Axis-4 schema/evidence audit.

Inputs:
  benchmark/{uid}/queries.csv  for uid in USER_IDS
  backend/{uid}/{app}.json     for the hashtag universe per user

Outputs:
  benchmark/benchmark_analysis/query_stats.json
  benchmark/benchmark_analysis/schema_audit.json
  benchmark/benchmark_analysis/samples/{uid}.json    (~30 stratified rows/user)
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.task_registry import TASK_TYPE_META  # noqa: E402

USER_IDS = ["105", "115", "229", "282", "760"]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
BACKEND_DIR = REPO_ROOT / "backend"
OUT_DIR = REPO_ROOT / "benchmark" / "benchmark_analysis"
SAMPLES_DIR = OUT_DIR / "samples"
APPS = ["instagram", "facebook", "threads", "chatbot", "ai_studio"]

SEED = 42
TARGET_SAMPLES_PER_USER = 30
MIN_PER_TASK_TYPE = 2

RANKING_TASKS = {
    "personalized_recommendation",
    "at_ai_directive_followup",
    "new_suggestions_recsys",
}

# Task types where the surface naturally lacks an explicit user-typed query
# (ranking probes, agent-composed writes, proactive briefings, etc.).
# "Missing user_query" is NOT a defect for these.
NO_EXPLICIT_QUERY_TASKS = {
    "personalized_recommendation",
    "at_ai_directive_followup",
    "new_suggestions_recsys",
    "daily_personalized_briefing",
    "agentic_user_tone_post",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_dm_digest",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_group_dm_summary",
    "agentic_wrong_recipient_check",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
    "local_recommendation_geo_shift",
    "over_personalization_repetition_recsys",
    "over_personalization_repetition_chatbot",
}

# Tasks that emit pair-scored example/inferior responses (everything except
# repetition arrays and proactive cards).
PAIR_SCORED = lambda tt, fam: (
    fam not in ("proactive_actions",)
    and tt not in (
        "over_personalization_repetition_recsys",
        "over_personalization_repetition_chatbot",
    )
)

HASHTAG_RE = re.compile(r"#\w+")


def load_csv(uid: str) -> list[dict]:
    p = BENCHMARK_DIR / uid / "queries.csv"
    csv.field_size_limit(10_000_000)
    rows: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            try:
                inst = json.loads(r.get("instance_json") or "{}")
            except Exception:
                inst = {}
            r["_inst"] = inst
            rows.append(r)
    return rows


def user_hashtag_universe(uid: str) -> set[str]:
    hs: set[str] = set()
    for app in APPS:
        p = BACKEND_DIR / uid / f"{app}.json"
        if not p.exists():
            continue
        try:
            evs = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        for ev in evs or []:
            for h in ev.get("source_hashtags") or []:
                if h:
                    hs.add(h.lstrip("#").lower())
            content = ev.get("content") or {}
            for h in content.get("hashtags") or []:
                if h:
                    hs.add(h.lstrip("#").lower())
            for pref in ev.get("preferences") or []:
                for h in pref.get("source_hashtags") or []:
                    if h:
                        hs.add(h.lstrip("#").lower())
    return hs


HASHTAG_KEYS = {
    "source_hashtags", "hashtags", "target_hashtags", "evidence_hashtags",
    "positive_directive_hashtags", "negative_directive_hashtags",
    "cluster_hashtags", "off_persona_distractor_hashtags", "directive_hashtags",
    "_sensitive_event_evidence_row_hashtags",
}


def extract_hashtags(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, str):
        for m in HASHTAG_RE.finditer(obj):
            out.add(m.group(0).lstrip("#").lower())
        return out
    if isinstance(obj, list):
        for x in obj:
            out |= extract_hashtags(x)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in HASHTAG_KEYS and isinstance(v, list):
                for h in v:
                    if isinstance(h, str) and h:
                        out.add(h.lstrip("#").lower())
            else:
                out |= extract_hashtags(v)
    return out


def axis_4_check(row: dict, inst: dict) -> dict:
    """Row-format / required-field check (verifier Axis 4).

    Splits issues into HARD (row is not a valid test) and SOFT (incomplete but
    structurally OK). Returns a `postprocessed` flag separately so the report
    can distinguish "build emitted but LLM postprocess hasn't run" from
    "genuinely malformed".
    """
    hard: list[str] = []
    soft: list[str] = []
    task_type = row.get("task_type", "")
    task_family = row.get("task_family", "")
    rubric_tags_csv = (row.get("rubric_tags") or "").strip()

    if not rubric_tags_csv:
        hard.append("rubric_tags column empty")

    if task_family == "proactive_actions":
        if not inst.get("expected_behavior"):
            hard.append("missing expected_behavior")
        if not inst.get("trigger_evidence"):
            hard.append("missing trigger_evidence")
        if not inst.get("jitai_card"):
            hard.append("missing jitai_card")
        if not inst.get("tool_call_rules"):
            hard.append("missing tool_call_rules")
        postprocessed = True  # proactive uses a different schema; no postprocess needed
    else:
        has_query = bool(
            (row.get("query_text") or "").strip()
            or inst.get("user_query")
            or inst.get("query_text")
            or inst.get("query")
            or inst.get("queries")
            or inst.get("topic")  # vague_refind
            or inst.get("draft")  # wrong_recipient_check
        )
        if not has_query and task_type not in NO_EXPLICIT_QUERY_TASKS:
            hard.append("missing user_query/query_text")
        elif not has_query:
            soft.append("no surface user query (expected for this task_type)")

        if task_type in (
            "over_personalization_repetition_recsys",
            "over_personalization_repetition_chatbot",
        ):
            if not inst.get("queries"):
                hard.append("repetition task missing queries[]")
            if not inst.get("target_pref"):
                hard.append("repetition task missing target_pref")
            postprocessed = True  # different schema; example/inferior n/a
        else:
            has_example = bool(inst.get("example_response"))
            has_inferior = bool(inst.get("inferior_response"))
            gt_ok = bool(
                inst.get("groundtruth_preference")
                or inst.get("groundtruth")
                or inst.get("held_out_preference")
                or inst.get("target_pref")
                or inst.get("gt_slice")
            )
            if not has_example:
                hard.append("missing example_response (LLM postprocess didn't run)")
            if not has_inferior:
                hard.append("missing inferior_response (LLM postprocess didn't run)")
            if not gt_ok:
                hard.append("missing groundtruth_preference")
            postprocessed = has_example and has_inferior

    if task_type in RANKING_TASKS:
        cands = inst.get("candidates") or []
        if len(cands) < 5:
            hard.append(f"ranking task has only {len(cands)} candidates")
        if inst.get("held_out_idx") is None and not inst.get("positive_indices"):
            hard.append("ranking task missing held_out_idx/positive_indices")

    if task_family == "agentic":
        if not (inst.get("tool_call") or inst.get("tool_call_rules")):
            hard.append("agentic task missing tool_call/tool_call_rules")

    return {
        "pass": len(hard) == 0,
        "postprocessed": postprocessed,
        "hard_issues": hard,
        "soft_issues": soft,
    }


def evidence_traceback(inst: dict, user_hashtags: set[str]) -> dict:
    row_hashtags = extract_hashtags(inst)
    if not row_hashtags:
        return {"checked": False, "overlap": 0, "trace_pass": None,
                "note": "no hashtags in instance"}
    overlap = row_hashtags & user_hashtags
    return {
        "checked": True,
        "row_hashtags": len(row_hashtags),
        "user_overlap": len(overlap),
        "trace_pass": len(overlap) >= 1,
        "sample_matched": sorted(list(overlap))[:5],
        "sample_unmatched": sorted(
            [h for h in row_hashtags if h not in user_hashtags]
        )[:5],
    }


def stratified_sample(rows: list[dict], target: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_type[r.get("task_type", "")].append(i)
    chosen: set[int] = set()
    for tt, idxs in by_type.items():
        picks = rng.sample(idxs, min(MIN_PER_TASK_TYPE, len(idxs)))
        chosen.update(picks)
    remaining = [i for i in range(len(rows)) if i not in chosen]
    rng.shuffle(remaining)
    need = max(0, target - len(chosen))
    chosen.update(remaining[:need])
    return sorted(chosen)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    per_user_totals: dict[str, int] = {}
    per_user_task_family: dict[str, dict[str, int]] = {}
    per_user_task_type: dict[str, dict[str, int]] = {}
    overall_task_type: dict[str, int] = defaultdict(int)

    schema_audit: dict = {
        "per_user": {},
        "per_user_task_type_pass_rate": {},
    }

    canonical_task_types = sorted(TASK_TYPE_META.keys())

    for uid in USER_IDS:
        rows = load_csv(uid)
        per_user_totals[uid] = len(rows)

        fam_counts: dict[str, int] = defaultdict(int)
        type_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            fam_counts[r.get("task_family", "")] += 1
            type_counts[r.get("task_type", "")] += 1
            overall_task_type[r.get("task_type", "")] += 1
        per_user_task_family[uid] = dict(fam_counts)
        per_user_task_type[uid] = dict(type_counts)

        user_hs = user_hashtag_universe(uid)
        row_verdicts = []
        per_tt_pass: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "pass": 0, "postprocessed": 0}
        )
        evidence_pass = 0
        evidence_checked = 0
        n_postprocessed = 0
        for r in rows:
            verdict = axis_4_check(r, r["_inst"])
            trace = evidence_traceback(r["_inst"], user_hs)
            if trace["checked"]:
                evidence_checked += 1
                if trace["trace_pass"]:
                    evidence_pass += 1
            if verdict["postprocessed"]:
                n_postprocessed += 1
            row_verdicts.append({
                "query_id": r.get("query_id"),
                "task_type": r.get("task_type"),
                "task_family": r.get("task_family"),
                "axis_4": verdict,
                "evidence_traceback": trace,
            })
            tt = r.get("task_type", "")
            per_tt_pass[tt]["total"] += 1
            if verdict["pass"]:
                per_tt_pass[tt]["pass"] += 1
            if verdict["postprocessed"]:
                per_tt_pass[tt]["postprocessed"] += 1

        schema_audit["per_user"][uid] = {
            "n_rows": len(rows),
            "n_pass": sum(1 for v in row_verdicts if v["axis_4"]["pass"]),
            "n_postprocessed": n_postprocessed,
            "n_evidence_checked": evidence_checked,
            "n_evidence_pass": evidence_pass,
            "user_hashtag_universe_size": len(user_hs),
            "row_verdicts": row_verdicts,
        }
        schema_audit["per_user_task_type_pass_rate"][uid] = {
            tt: dict(v) for tt, v in per_tt_pass.items()
        }

        sample_idxs = stratified_sample(rows, TARGET_SAMPLES_PER_USER, SEED)
        sample_out = []
        for i in sample_idxs:
            r = rows[i]
            sample_out.append({
                "query_id": r.get("query_id"),
                "seq": r.get("seq"),
                "task_family": r.get("task_family"),
                "task_type": r.get("task_type"),
                "ts": r.get("ts"),
                "ts_iso": r.get("ts_iso"),
                "query_text": r.get("query_text"),
                "app_context": r.get("app_context"),
                "entry_point": r.get("entry_point"),
                "mcp_tools_allowed": r.get("mcp_tools_allowed"),
                "state_write_policy": r.get("state_write_policy"),
                "expected_response_kind": r.get("expected_response_kind"),
                "rubric_tags": r.get("rubric_tags"),
                "instance_json": r["_inst"],
            })
        (SAMPLES_DIR / f"{uid}.json").write_text(
            json.dumps(sample_out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    overall_ranking = sorted(overall_task_type.items(), key=lambda x: -x[1])

    stats = {
        "user_ids": USER_IDS,
        "per_user_totals": per_user_totals,
        "per_user_task_family": per_user_task_family,
        "per_user_task_type": per_user_task_type,
        "overall_task_type_ranking": overall_ranking,
        "canonical_task_types": canonical_task_types,
        "grand_total": sum(per_user_totals.values()),
    }
    (OUT_DIR / "query_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "schema_audit.json").write_text(
        json.dumps(schema_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Stage A] Wrote stats for {len(USER_IDS)} users.")
    print(f"  per_user_totals: {per_user_totals}")
    print(f"  grand_total: {stats['grand_total']}")
    for uid in USER_IDS:
        u = schema_audit["per_user"][uid]
        n = u["n_rows"]
        print(
            f"  {uid}: axis-4 {u['n_pass']}/{n} ({100*u['n_pass']/max(1,n):.1f}%)  "
            f"postprocessed {u['n_postprocessed']}/{n} ({100*u['n_postprocessed']/max(1,n):.1f}%)  "
            f"evidence-trace {u['n_evidence_pass']}/{u['n_evidence_checked']} checked"
        )


if __name__ == "__main__":
    main()
