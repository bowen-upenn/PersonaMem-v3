#!/usr/bin/env python3
"""Offline recompute of the new graded-NDCG@5 headline for the 3 ranking tasks
on the matched-10 personas, for every model/mode. No eval run, no LLM:
joins backend/{uid}/test.json (relevance labels in instance_full) with each
run's results.csv (the model's ranking), and recomputes NDCG@5 with the shared
_graded_ndcg_at_k. Prints per-task NDCG@5 + Overall micro per model.

Run from repo root.  Importable: get_tables() returns (per_task, overall).
"""
import csv, json, glob, os, sys
csv.field_size_limit(10**9)
sys.path.insert(0, ".")
from data_preparation.utils import extract_json_from_response
from evaluation.tasks.personalized_recommendation import _graded_ndcg_at_k
from scripts.aggregate_eval import _accuracy_value

MATCHED = [1, 2, 3, 5, 6, 8, 9, 10, 13, 14]
THREE = {"personalized_recommendation", "at_ai_directive_followup", "hidden_persona_recommendation"}
MODES = [
    ("llm_longctx_gpt5.5_judged", "GPT-LC"), ("llm_memory_gpt5.5", "GPT-Mem"),
    ("mem0_gpt5.5", "GPT-Mem0"), ("codex_agent_gpt5.5", "GPT-Codex"),
    ("llm_longctx_gemini3.5flash_judged", "Gem-LC"), ("llm_memory_gemini3.5flash_judged", "Gem-Mem"),
    ("agent_tools_opus4.8", "OPUS-CC"), ("agent_tools_sonnet4.6", "Sonnet-CC"),
]


def _labels_for_user(uid):
    """query_id -> (task, positives:set, hard_negs:set, n_candidates)."""
    path = f"backend/{uid}/test.json"
    out = {}
    if not os.path.exists(path):
        return out
    for i in json.load(open(path)):
        task = i.get("task_type")
        if task not in THREE:
            continue
        inf = i.get("instance_full")
        if isinstance(inf, str):
            inf = json.loads(inf)
        if not isinstance(inf, dict):
            continue
        n = len(inf.get("candidates") or [])
        if task == "at_ai_directive_followup":
            pos = set(inf.get("positive_indices") or [])
            hard = set(inf.get("carveout_indices") or [])
        else:  # recsys + hidden_persona: single held-out target + hard negs
            ho = inf.get("held_out_idx")
            pos = {ho} if isinstance(ho, int) else set()
            hard = set(inf.get("hard_negative_idxs") or [])
        out[i["query_id"]] = (task, pos, hard, n)
    return out


def _ranking(resp, task, n):
    parsed = extract_json_from_response(resp) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    key = "ranked_indices" if task == "at_ai_directive_followup" else "ranked_indexes"
    ranked = parsed.get(key) or []
    if task == "at_ai_directive_followup":
        # match e2 scorer: invalid permutation -> identity order
        if not isinstance(ranked, list) or sorted(set(ranked)) != list(range(n)):
            ranked = list(range(n))
    return ranked if isinstance(ranked, list) else []


def _new_ndcg(resp, task, pos, hard, n):
    # hidden_persona uses binary relevance (filler=0) ONLY while its slate has no
    # hard-negs (avoids the graded-NDCG ~75% floor); once surface-match hard-negs
    # are present it scores graded like the other ranking tasks. Conditional on the
    # actual slate so the regen transition is automatic.
    from evaluation.tasks.personalized_recommendation import _NDCG_REL_FILLER
    fg = 0.0 if (task == "hidden_persona_recommendation" and not hard) else _NDCG_REL_FILLER
    return _graded_ndcg_at_k(_ranking(resp, task, n), pos, hard, 5, filler_grade=fg)


def get_tables():
    labels = {u: _labels_for_user(u) for u in MATCHED}
    per_task = {}   # label -> {task -> mean ndcg}
    overall = {}    # label -> micro
    for mode, lbl in MODES:
        task_vals = {t: [] for t in THREE}
        all_acc = []
        for u in MATCHED:
            p = f"results/{mode}/{u}/results.csv"
            if not os.path.exists(p):
                continue
            for r in csv.DictReader(open(p)):
                tt = r["task_type"]
                if tt in THREE:
                    lab = labels.get(u, {}).get(r["query_id"])
                    if lab is None:
                        continue
                    _, pos, hard, n = lab
                    v = _new_ndcg(r.get("agent_response") or "", tt, pos, hard, n) * 100.0
                    task_vals[tt].append(v)
                    all_acc.append(v)
                else:
                    a = _accuracy_value(tt, json.loads(r.get("metrics_json") or "{}"), r.get("status") or "ok")
                    if a is not None:
                        all_acc.append(a)
        per_task[lbl] = {t: (sum(v) / len(v) if v else None) for t, v in task_vals.items()}
        overall[lbl] = sum(all_acc) / len(all_acc) if all_acc else None
    return per_task, overall


if __name__ == "__main__":
    per_task, overall = get_tables()
    print(f"{'model':10s} | {'ProactiveFeed':>13} {'@AIdirective':>12} {'HiddenPersona':>13} | {'Overall':>8}")
    for _, lbl in MODES:
        pt = per_task[lbl]
        def f(x): return f"{x:.1f}" if x is not None else "  --"
        print(f"{lbl:10s} | {f(pt['personalized_recommendation']):>13} "
              f"{f(pt['at_ai_directive_followup']):>12} {f(pt['hidden_persona_recommendation']):>13} | "
              f"{f(overall[lbl]):>8}")
