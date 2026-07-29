#!/usr/bin/env python3
"""Step 3 of the standardized pair audit (see AUDIT.md, Slice A).

Aggregate the verify workflow's returned JSON into per-axis pass rates and an
enriched findings.json (each surviving problem re-attached to its real
query / GOLD / FOIL text from backend/{uid}/test.json).

Confirmation logic per flagged axis: 2 real_problem -> CONFIRMED, 1 -> PLAUSIBLE,
0 -> DROPPED (adversarially refuted).

Usage:
  python scripts/qa_pair_audit/aggregate.py --result <workdir>/wf_result.json \
      --workdir <workdir> [--users 1 2 3 ...]
"""
import argparse, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USERS = [1,2,3,5,6,8,9,10,13,14,26,105,115,209,229,282,461,655,760,835]
AXES = ["naturalness", "example", "inferior"]
TASKS = ("chatbot_personalized_response", "over_personalization_chatbot_text")


def build_lookup(users):
    lut = {}
    for u in users:
        d = json.load(open(ROOT / f"backend/{u}/test.json"))
        for r in d:
            if r.get("task_type") not in TASKS:
                continue
            inf = r.get("instance_full") or {}
            ir = r.get("inferior_response")
            lut[r["query_id"]] = {
                "user_id": str(u), "task_type": r["task_type"], "arm": inf.get("arm"),
                "user_query": r.get("user_query"), "example_response": r.get("example_response"),
                "inferior_text": ir.get("text") if isinstance(ir, dict) else ir,
                "flaw_evidence": ir.get("flaw_evidence") if isinstance(ir, dict) else None,
                "groundtruth_preference": r.get("groundtruth_preference"),
                "held_out_preference": inf.get("held_out_preference"),
            }
    return lut


def classify(votes):
    real = votes.count("real_problem")
    if real >= 2:
        return "CONFIRMED"
    if real == 1:
        return "PLAUSIBLE"
    return "NO_VERIFIER" if not votes else "DROPPED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--users", nargs="*", type=int, default=DEFAULT_USERS)
    a = ap.parse_args()

    res = json.load(open(a.result))
    if isinstance(res, dict) and "result" in res and "results" not in res:
        res = res["result"]
        if isinstance(res, str):
            res = json.loads(res)
    lut = build_lookup(a.users)
    results = res.get("results", [])

    tally = collections.defaultdict(collections.Counter)
    judged_rows = collections.defaultdict(set)
    findings = []

    for wu in results:
        if not wu:
            continue
        task = wu["task_type"]
        jrows = {r["query_id"]: r for r in ((wu.get("judge") or {}).get("rows") or [])}
        vv = collections.defaultdict(lambda: collections.defaultdict(list))
        vreason = collections.defaultdict(list)
        for v in (wu.get("verifications") or []):
            vd = v.get("verdict") or {}
            qid = v.get("query_id")
            for ax in AXES:
                val = vd.get(f"{ax}_verdict")
                if val in ("real_problem", "refuted"):
                    vv[qid][ax].append(val)
            if vd.get("reason"):
                vreason[qid].append(vd["reason"])
        for qid, jr in jrows.items():
            judged_rows[task].add(qid)
            for ax in AXES:
                tally[(task, ax)]["judged"] += 1
                if jr.get(f"{ax}_pass") is False:
                    tally[(task, ax)]["judge_flagged"] += 1
                    status = classify(vv[qid][ax])
                    tally[(task, ax)][status] += 1
                    if status in ("CONFIRMED", "PLAUSIBLE", "NO_VERIFIER"):
                        info = lut.get(qid, {})
                        findings.append({
                            "status": status, "task_type": task, "axis": ax, "query_id": qid,
                            "user_id": info.get("user_id"), "arm": info.get("arm"),
                            "severity": jr.get("severity"), "judge_reason": jr.get(f"{ax}_reason"),
                            "verifier_votes": vv[qid][ax], "verifier_reasons": vreason[qid],
                            "user_query": info.get("user_query"), "example_response": info.get("example_response"),
                            "inferior_text": info.get("inferior_text"), "flaw_evidence": info.get("flaw_evidence"),
                            "groundtruth_preference": info.get("groundtruth_preference"),
                            "held_out_preference": info.get("held_out_preference"),
                        })

    order = {"CONFIRMED": 0, "PLAUSIBLE": 1, "NO_VERIFIER": 2}
    sev = {"major": 0, "minor": 1, "none": 2, None: 3}
    findings.sort(key=lambda f: (order.get(f["status"], 9), sev.get(f["severity"], 9),
                                 f["task_type"], f["axis"], f["query_id"]))
    summary = {
        "n_rows": res.get("n_rows"), "n_units": res.get("n_units"),
        "n_units_returned": len([w for w in results if w]),
        "rows_judged_by_task": {t: len(s) for t, s in judged_rows.items()},
        "tally": {f"{t}|{ax}": dict(c) for (t, ax), c in tally.items()},
        "n_confirmed": sum(f["status"] == "CONFIRMED" for f in findings),
        "n_plausible": sum(f["status"] == "PLAUSIBLE" for f in findings),
        "n_no_verifier": sum(f["status"] == "NO_VERIFIER" for f in findings),
    }
    out = Path(a.workdir) / "findings.json"
    json.dump({"summary": summary, "findings": findings}, open(out, "w"), indent=1, ensure_ascii=False)

    print("=" * 72)
    print(f"units {summary['n_units_returned']}/{summary['n_units']}  rows {sum(summary['rows_judged_by_task'].values())}")
    for task in TASKS:
        if not judged_rows[task]:
            continue
        print(f"\n### {task}  ({len(judged_rows[task])} rows)")
        for ax in AXES:
            c = tally[(task, ax)]
            j, fl = c["judged"], c["judge_flagged"]
            pr = 100 * (j - fl) / j if j else 0
            print(f"  {ax:12s} pass {j-fl:3d}/{j:<3d} ({pr:5.1f}%)  flagged {fl:2d} -> "
                  f"CONFIRMED {c['CONFIRMED']:2d} | PLAUSIBLE {c['PLAUSIBLE']:2d} | DROPPED {c['DROPPED']:2d}")
    print(f"\nCONFIRMED {summary['n_confirmed']}  PLAUSIBLE {summary['n_plausible']}  -> {out}")


if __name__ == "__main__":
    main()
