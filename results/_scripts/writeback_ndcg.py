#!/usr/bin/env python3
"""Write the new graded-NDCG@5/@3 into results.csv for the 3 ranking tasks, so
the aggregate_eval pipeline reads the new headline (equivalent to a judge-off
`run_eval --replay_from`, but deterministic + no harness/LLM). Recomputes from
backend/{uid}/test.json labels + the stored ranking; only `ndcg_at_3/5` in the
3 ranking rows' metrics_json change. Snapshots each file first; verifies row
count + query_ids are preserved before the atomic swap (csv-truncation guard).

Usage: writeback_ndcg.py [mode1 mode2 ...]   (default: all 8 matrix modes)
"""
import csv, json, os, sys, glob
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
sys.path.insert(0, ".")
sys.path.insert(0, "results/_scripts")
from recompute_ndcg_matched10 import _labels_for_user, _ranking
from evaluation.tasks.personalized_recommendation import _graded_ndcg_at_k, _NDCG_REL_FILLER

THREE = {"personalized_recommendation", "at_ai_directive_followup", "hidden_persona_recommendation"}
BACKUP = "results/_ndcg_writeback_backup"
ALL_MODES = [
    "llm_longctx_gpt5.5_judged", "llm_memory_gpt5.5", "mem0_gpt5.5", "codex_agent_gpt5.5",
    "llm_longctx_gemini3.5flash_judged", "llm_memory_gemini3.5flash_judged",
    "agent_tools_opus4.8", "agent_tools_sonnet4.6",
]


def process_file(path, labels):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)
    orig_qids = [row["query_id"] for row in rows]
    updated = 0
    for row in rows:
        tt = row["task_type"]
        if tt not in THREE:
            continue
        lab = labels.get(row["query_id"])
        if lab is None:
            continue
        _, pos, hard, n = lab
        ranked = _ranking(row.get("agent_response") or "", tt, n)
        m = json.loads(row.get("metrics_json") or "{}")
        # hidden_persona uses binary relevance (filler=0) ONLY while its slate has
        # no hard-negs (avoids the ~75% floor); once surface-match hard-negs are
        # added it scores graded like the other ranking tasks. Conditional on the
        # actual slate so the regen transition is automatic.
        fg = 0.0 if (tt == "hidden_persona_recommendation" and not hard) else _NDCG_REL_FILLER
        m["ndcg_at_3"] = round(_graded_ndcg_at_k(ranked, pos, hard, 3, filler_grade=fg), 4)
        m["ndcg_at_5"] = round(_graded_ndcg_at_k(ranked, pos, hard, 5, filler_grade=fg), 4)
        row["metrics_json"] = json.dumps(m)
        updated += 1
    # write atomically + verify nothing lost
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(tmp, newline="") as f:
        check = list(csv.DictReader(f))
    if [c["query_id"] for c in check] != orig_qids:
        os.remove(tmp)
        raise RuntimeError(f"ROW MISMATCH in {path} — aborted (orig {len(orig_qids)}, new {len(check)})")
    os.replace(tmp, path)
    return updated, len(rows)


def main():
    modes = sys.argv[1:] or ALL_MODES
    labels_cache = {}
    grand = 0
    for mode in modes:
        files = sorted(glob.glob(f"results/{mode}/*/results.csv"))
        mtot = 0
        for path in files:
            uid = os.path.basename(os.path.dirname(path))
            if not uid.isdigit():
                continue
            # snapshot once
            bdir = f"{BACKUP}/{mode}/{uid}"
            os.makedirs(bdir, exist_ok=True)
            bpath = f"{bdir}/results.csv"
            if not os.path.exists(bpath):
                with open(path) as src, open(bpath, "w") as dst:
                    dst.write(src.read())
            if uid not in labels_cache:
                labels_cache[uid] = _labels_for_user(uid)
            upd, total = process_file(path, labels_cache[uid])
            mtot += upd
        print(f"  {mode:38s} updated {mtot} ranking rows across {len(files)} personas")
        grand += mtot
    print(f"TOTAL ndcg rows written: {grand}")


if __name__ == "__main__":
    main()
