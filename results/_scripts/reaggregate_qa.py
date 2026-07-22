#!/usr/bin/env python3
"""Re-aggregate the QA audit summary, overriding the `completeness` gate
with a DETERMINISTIC field-presence check.

`completeness` is, by definition, a structural presence check (are the
required fields populated?). The auditor LLM mis-reads populated-but-
structured `groundtruth_preference` values (dicts / descriptive strings)
as "empty" and emits false-positive failures. Since we can verify the
fields ARE present, the deterministic verdict is authoritative — this
re-aggregation replaces only the completeness verdict, leaving every
LLM-judged gate exactly as measured. No new LLM calls.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/audit/qa_audit_p1"

# Tasks where user_query is legitimately absent (proactive / ranking /
# structured-input). Mirrors the completeness prompt's own rule.
_NO_USER_QUERY_PREFIXES = ("agentic_", "proactive_", "restraint_", "over_personalization_repetition_")
_NO_USER_QUERY_EXACT = {
    "personalized_recommendation", "hidden_persona_recommendation",
    "at_ai_directive_followup", "hidden_persona_implicit_qa",
    "local_recommendation_geo_shift", "active_mistake_prevention",
}


def _nonempty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, dict):
        return any(_nonempty(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return bool(str(v).strip())


def _needs_user_query(tt: str) -> bool:
    if tt in _NO_USER_QUERY_EXACT:
        return False
    return not any(tt.startswith(p) for p in _NO_USER_QUERY_PREFIXES)


def deterministic_completeness(inst: dict) -> bool:
    ok = _nonempty(inst.get("example_response")) and \
        _nonempty(inst.get("inferior_response")) and \
        _nonempty(inst.get("groundtruth_preference") or inst.get("groundtruth_preference_obj"))
    if _needs_user_query(inst.get("task_type", "")):
        # the user message lives under different keys across builders:
        # user_query (most), query (context_shift), query_text (some over-pers).
        ok = ok and _nonempty(
            inst.get("user_query") or inst.get("query") or inst.get("query_text"))
    return ok


def main():
    rows = [json.loads(l) for l in (OUT / "audit_rows.jsonl").read_text().splitlines() if l.strip()]
    items = {x.get("query_id"): x for x in json.loads((ROOT / "backend/1/test.json").read_text())}

    overridden = 0
    for r in rows:
        item = items.get(r["query_id"], {})
        inst = dict(item.get("instance_full") or item)
        inst["task_type"] = r["task_type"]
        # promote top-level user_query (test.json hoists it above instance_full)
        if item.get("user_query") and not inst.get("user_query"):
            inst["user_query"] = item["user_query"]
        passed = deterministic_completeness(inst)
        for d in r["dimensions"]:
            if d["name"] == "completeness":
                if d["passed"] != passed or d.get("skipped"):
                    overridden += 1
                d["passed"] = passed
                d["skipped"] = False
                d["score"] = 1.0 if passed else 0.0
                d["reason"] = "deterministic field-presence check"

    # re-tally
    by_dim = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    by_task = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0}))
    fail_ex = defaultdict(list)
    for r in rows:
        for d in r["dimensions"]:
            s = by_dim[d["name"]]
            ts = by_task[r["task_type"]][d["name"]]
            if d["skipped"]:
                s["skipped"] += 1; ts["skipped"] += 1
            elif d["passed"]:
                s["passed"] += 1; ts["passed"] += 1
            else:
                s["failed"] += 1; ts["failed"] += 1
                if len(fail_ex[d["name"]]) < 5:
                    fail_ex[d["name"]].append({"query_id": r["query_id"], "task_type": r["task_type"], "reason": (d.get("reason") or "")[:240]})

    def rate(s):
        ev = s["passed"] + s["failed"]
        return (s["passed"] / ev) if ev else None

    dim_summary = {n: {**s, "evaluated": s["passed"] + s["failed"], "pass_rate": rate(s)} for n, s in by_dim.items()}
    tot_pass = sum(s["passed"] for s in by_dim.values())
    tot_fail = sum(s["failed"] for s in by_dim.values())

    old = json.loads((OUT / "audit_summary.json").read_text())
    old["overall"] = {"evaluated_checks": tot_pass + tot_fail, "passed": tot_pass,
                      "failed": tot_fail, "pass_rate": tot_pass / (tot_pass + tot_fail)}
    old["by_dimension"] = dim_summary
    old["by_task_dimension"] = {t: {n: {**s, "pass_rate": rate(s)} for n, s in dd.items()} for t, dd in by_task.items()}
    old["fail_examples"] = dict(fail_ex)
    old["completeness_method"] = "deterministic field-presence (LLM verdict overridden)"
    (OUT / "audit_summary.json").write_text(json.dumps(old, indent=2, ensure_ascii=False))
    print(f"[reaggregate] completeness overridden on {overridden} rows; "
          f"completeness now {dim_summary['completeness']['passed']}/"
          f"{dim_summary['completeness']['evaluated']} "
          f"({(dim_summary['completeness']['pass_rate'] or 0)*100:.1f}%)")


if __name__ == "__main__":
    main()
