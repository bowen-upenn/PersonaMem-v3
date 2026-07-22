#!/usr/bin/env python3
"""Surgical append of extra proactive_overactive_check queries to test.json.

WHY: the negative-control idle-moment supply was starved (n=6 across few users)
by the old _gather_idle_moments (one-shot per stratum + cap 3 < quota). That code
is now fixed; this script regenerates the overactive_check candidates with the
fixed gather and APPENDS the resulting instances to each user's test.json WITHOUT
touching any other row (preserves all existing query_ids so run_eval --resume stays
clean). Build is LLM-free: the gold is deterministic ("stay silent") and the
inferior is templated (build_proactive_overactive_check works with discovery_llm=None).

Append-only + a canonical row template (copied from an existing overactive_check
row) => byte-identical existing rows; only new rows added at the tail.

Usage:
  python scripts/append_overactive_check.py --users "1 2 3 ..." [--apply]
Without --apply it runs a DRY pass (no writes), printing what it WOULD add.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from data_preparation.persona_agent import PersonaAgent          # noqa: E402
from evaluation.backend_query import BackendQuery                # noqa: E402
from evaluation.tasks.proactive_actions import (                 # noqa: E402
    build_proactive_overactive_check,
)

APPS = ("instagram", "facebook", "threads", "chatbot")
# Constant fields shared by every proactive_overactive_check row (filled from a
# canonical existing row at runtime; these are the keys we copy verbatim).
_CONSTANT_KEYS = (
    "task_family", "task_type", "query_kind", "expected_behavior",
    "user_query", "prior_conversation", "groundtruth_preference_obj",
    "reference_example", "distractor_preferences", "tool_call",
    "example_response_self_check", "example_response_voice_evidence",
    "inferior_response_voice_evidence", "voice_evidence_smoke_check",
    "voice_evidence_smoke_check_after_regen",
)
# Per-instance fields (overridden from the freshly-built instance).
_PER_INSTANCE = ("example_response", "inferior_response",
                 "groundtruth_preference", "rubric_tags")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _find_template(users):
    """Return a canonical proactive_overactive_check row (schema skeleton) from
    the first user that already has one."""
    for uid in users:
        p = f"backend/{uid}/test.json"
        if not os.path.exists(p):
            continue
        for r in _load(p):
            if isinstance(r, dict) and r.get("task_type") == "proactive_overactive_check":
                return r
    raise SystemExit("No existing proactive_overactive_check row found to use as a template.")


def process_user(uid, template, bq, apply):
    udir = f"backend/{uid}"
    test_path = f"{udir}/test.json"
    prof_path = f"{udir}/profile.json"
    if not (os.path.exists(test_path) and os.path.exists(prof_path)):
        return f"  u{uid}: SKIP (missing test.json/profile.json)"
    rows = _load(test_path)
    profile = _load(prof_path)
    app_events = {a: (_load(f"{udir}/{a}.json") if os.path.exists(f"{udir}/{a}.json") else [])
                  for a in APPS}

    # 1. Regenerate overactive_check candidates with the FIXED gather, avoiding
    #    every existing trigger (incl. existing overactive moments) so the new
    #    ones are distinct from what's already in test.json.
    ag = PersonaAgent(str(uid), backend_dir="backend")
    sensitive_periods = ag._gather_sensitive_event_periods(profile)
    existing_cands = dict(profile.get("proactive_trigger_candidates", {}) or {})
    fresh = ag._gather_idle_moments(app_events, profile, existing_cands, sensitive_periods)
    if not fresh:
        return f"  u{uid}: 0 fresh idle moments (no append)"

    # 2. Build instances from the fresh candidates (LLM-free). build_* reads the
    #    catalog from profile.json, so write the fresh candidates first.
    pc = dict(profile.get("proactive_trigger_candidates", {}) or {})
    pc["overactive_check"] = fresh
    profile["proactive_trigger_candidates"] = pc
    if apply:
        Path(prof_path + ".bak_overactive").write_text(json.dumps(_load(prof_path)))
        with open(prof_path, "w") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    else:
        # DRY: write fresh candidates to a temp profile so build_* can read them
        tmp = f"/tmp/_oc_profile_{uid}.json"
        with open(tmp, "w") as f:
            json.dump(profile, f)
        # build_* reads backend/{uid}/profile.json; in dry mode we still build
        # off the in-memory regen by temporarily swapping — simplest is to just
        # report the candidate count without building.
        return f"  u{uid}: DRY would regen {len(fresh)} overactive candidates (existing OC rows: {sum(1 for r in rows if r.get('task_type')=='proactive_overactive_check')})"

    maxts = max((int(r.get("ts") or 0) for r in rows), default=0)
    insts = build_proactive_overactive_check(bq, str(uid), maxts, discovery_llm=None)
    if not insts:
        return f"  u{uid}: builder produced 0 instances (no append)"

    # 3. Template-append: copy the canonical row, swap per-instance values.
    new_rows = []
    for inst in insts:
        row = {k: copy.deepcopy(template.get(k)) for k in template}  # exact schema
        iid = inst.get("instance_id") or inst.get("test_id") or f"overactive_{uid}"
        # de-dup: skip if this instance_id already present
        if any(r.get("query_id", "").endswith(iid) for r in rows + new_rows):
            continue
        ts = int(inst.get("t_test") or maxts)
        row["query_id"] = f"{uid}:append:{iid}"
        row["ts"] = ts
        row["ts_iso"] = inst.get("t_test_iso") or template.get("ts_iso")
        for k in _PER_INSTANCE:
            if inst.get(k) is not None:
                row[k] = inst[k]
        row["instance_full"] = inst
        new_rows.append(row)

    if not new_rows:
        return f"  u{uid}: all candidates duplicates (no append)"

    rows.extend(new_rows)
    Path(test_path + ".bak_overactive").write_text(json.dumps(_load(test_path)))
    with open(test_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return (f"  u{uid}: +{len(new_rows)} overactive rows "
            f"(total OC now {sum(1 for r in rows if r.get('task_type')=='proactive_overactive_check')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="space-separated user ids")
    ap.add_argument("--apply", action="store_true", help="write changes (else dry)")
    args = ap.parse_args()
    users = args.users.split()
    template = _find_template(users)
    print(f"[append_overactive] template from query_id={template.get('query_id')} "
          f"| mode={'APPLY' if args.apply else 'DRY'}")
    bq = BackendQuery("backend")
    for uid in users:
        try:
            print(process_user(uid, template, bq, args.apply))
        except Exception as e:
            import traceback
            print(f"  u{uid}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
