#!/usr/bin/env python3
"""Build per-model failure datasets for error analysis.

For each of the 8 (mode, model) runs, collect every SCORED row whose normalized
accuracy (_accuracy_value) falls below FAIL_THRESHOLD. For each failing row emit
a compact JSON record carrying everything a classifier needs to read:
  - task_type, capability_axis, accuracy, status, error
  - judge_text  : concatenation of every reasoning-like field present
  - metric_hint : compact dict of the salient sub-metrics / hard-fail flags
  - response    : first 700 chars of the model's final answer

Writes results/_scripts/erroranalysis/failures_<key>.jsonl (one row per failure)
and prints a coverage summary.
"""
import csv, json, os, sys
csv.field_size_limit(10**8)
ROOT = "/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"
sys.path.insert(0, ROOT)
from scripts.aggregate_eval import _accuracy_value
from evaluation.task_registry import get_capability_axis

import os
MATCHED = {int(u) for u in os.environ.get("PERSONAS", "").split()} or \
          {int(u) for u in os.listdir("results/agent_tools_opus4.8") if u.isdigit()}
FAIL_THRESHOLD = 60.0

RUNS = {
    "longctx_gpt55":   ("Long Context · GPT-5.5",       "results/llm_longctx_gpt5.5_judged"),
    "textmem_gpt55":   ("Textual Memory · GPT-5.5",     "results/llm_memory_gpt5.5"),
    "mem0_gpt55":      ("Mem0 w/ RAG · GPT-5.5",         "results/mem0_gpt5.5"),
    "codex_gpt55":     ("Codex High · GPT-5.5",          "results/codex_agent_gpt5.5"),
    "longctx_gemini":  ("Long Context · Gemini-3.5-Flash","results/llm_longctx_gemini3.5flash_judged"),
    "textmem_gemini":  ("Textual Memory · Gemini-3.5-Flash","results/llm_memory_gemini3.5flash_judged"),
    "claudecode_opus": ("Claude Code High · Opus-4.8",    "results/agent_tools_opus4.8"),
    "claudecode_sonnet":("Claude Code High · Sonnet-4.6", "results/agent_tools_sonnet4.6"),
}

REASON_FIELDS = [
    "pr_judge_reasoning", "judge_reasoning", "judge_reasoning_at_ai",
    "drift_reasoning", "restraint_justification", "justification_quality",
]

# salient metric keys to surface as hints (only those present are emitted)
HINT_KEYS = [
    # ranking / retrieval
    "ndcg_at_5", "ndcg_at_3", "hit_at_1", "hit_at_3", "recall_at_3", "mrr",
    "tier_concordance", "hard_neg_violation_rate", "top3_alignment_rate",
    # agentic personalization rubric
    "pr_primary_dim", "pr_primary_dim_score", "pr_preference_alignment_score",
    "pr_voice_match_score", "pr_telegraph_avoidance_score",
    "pr_over_personalization_score", "pr_query_score_0_10",
    "pr_privacy_leak_violated", "pr_avoid_leak_violated",
    "pr_stale_preference_use_violated", "pr_personalization_hard_fail_count",
    "mode_grading",
    # over-personalization / restraint / chatbot
    "appropriate_restraint", "helpfulness", "avoid_leak_rate",
    "personalization_leak_rate", "privacy_leak_hard_fail", "drift_over_personalized",
    "carve_out_respect", "held_out_hit", "no_hallucinated_preference",
    "target_match_recall", "preference_alignment",
    "fatigue_overuse_rate", "fatigue_passed", "query_score_0_10",
    # proactive
    "decision_correct", "trigger_detection_correctness", "avoid_overpersonalization",
    "stale_preference_use", "negative_leakage", "voice_match", "proactive_action_score",
    "evidence_cited", "restraint_justification",
    # hallucination / qa
    "abstention_quality_0_10", "non_substantive_response", "response_is_substantive",
    # hidden persona implicit
    "deep_motivation_alignment", "surface_query_satisfaction",
    "telegraph_avoidance_fail", "privacy_leak_fail",
    # geo
    "geo_shift_correctness", "stale_geo_anchor", "current_city_grounded",
    "geo_neutral_response",
    # mistake prevention
    "correct", "warning_issued", "must_mention_coverage", "leak", "polarity",
    # preference shift / lifecycle
    "preference_shift_consistency", "used_outdated_stance", "lifecycle_score",
]


def reason_text(m):
    parts = []
    for k in REASON_FIELDS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    # de-dup identical strings, keep order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return "  ||  ".join(out)


def hints(m):
    h = {}
    for k in HINT_KEYS:
        if k in m and m[k] is not None:
            v = m[k]
            if isinstance(v, float):
                v = round(v, 3)
            h[k] = v
    return h


def main():
    summary = []
    for key, (label, d) in RUNS.items():
        d = os.path.join(ROOT, d)
        out_path = os.path.join(ROOT, f"results/_scripts/erroranalysis/failures_{key}.jsonl")
        n_scored = n_fail = n_text = 0
        recs = []
        users = sorted(int(u) for u in os.listdir(d) if u.isdigit() and int(u) in MATCHED)
        for u in users:
            f = os.path.join(d, str(u), "results.csv")
            if not os.path.exists(f):
                continue
            for r in csv.DictReader(open(f)):
                m = json.loads(r["metrics_json"] or "{}")
                status = r["status"] or "ok"
                acc = _accuracy_value(r["task_type"], m, status)
                if acc is None:
                    continue
                n_scored += 1
                if acc >= FAIL_THRESHOLD:
                    continue
                n_fail += 1
                jt = reason_text(m)
                if jt:
                    n_text += 1
                recs.append({
                    "key": key,
                    "user": u,
                    "query_id": r.get("query_id", ""),
                    "task_type": r["task_type"],
                    "axis": get_capability_axis(r["task_type"]),
                    "accuracy": round(acc, 1),
                    "status": status,
                    "error": (r.get("error") or "")[:160],
                    "judge_text": jt[:1400],
                    "metric_hint": hints(m),
                    "response": (r.get("agent_response") or "")[:700],
                })
        with open(out_path, "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary.append((key, label, n_scored, n_fail, n_text, out_path))
        print(f"{key:20s} scored={n_scored:4d} fail={n_fail:4d} with_judge_text={n_text:4d} -> {os.path.basename(out_path)}")
    total = sum(s[3] for s in summary)
    print(f"\nTOTAL failing rows across 8 runs: {total}")


if __name__ == "__main__":
    main()
