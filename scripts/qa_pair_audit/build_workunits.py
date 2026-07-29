#!/usr/bin/env python3
"""Step 1 of the standardized pair audit (see AUDIT.md, Slice A).

Extract every chatbot_personalized_response + over_personalization_chatbot_text
row for a cohort into self-contained per-persona work-unit files (persona card +
a small batch of rows) that the verify workflow judges independently.

Usage:
  python scripts/qa_pair_audit/build_workunits.py --out /path/to/workdir \
      [--users 1 2 3 ...] [--tasks chatbot_personalized_response over_personalization_chatbot_text] \
      [--batch 5]
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USERS = [1,2,3,5,6,8,9,10,13,14,26,105,115,209,229,282,461,655,760,835]
DEFAULT_TASKS = ["chatbot_personalized_response", "over_personalization_chatbot_text"]


def trim(s, n):
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n] + " …[trunc]"


def persona_card(uid):
    p = json.load(open(ROOT / f"backend/{uid}/profile.json"))
    prefs = p.get("preferences") or []
    return {
        "user_id": str(uid), "name": p.get("name"), "gender": p.get("gender"),
        "race_ethnicity": p.get("race_ethnicity"), "career": p.get("career"),
        "education": p.get("education"), "big_five": p.get("big_five"),
        "bio": trim(p.get("bio"), 700),
        "preferences": [trim(x, 200) for x in prefs[:32]],
    }


def norm_prior(pc):
    if pc is None:
        return None
    if isinstance(pc, str):
        return trim(pc, 2200)
    if isinstance(pc, list):
        out = []
        for turn in pc[-8:]:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("speaker") or turn.get("from") or "?"
                content = turn.get("content") or turn.get("text") or turn.get("message") or ""
                out.append({"role": role, "content": trim(content, 500)})
            else:
                out.append({"role": "?", "content": trim(turn, 500)})
        return out
    return trim(json.dumps(pc), 2000)


def row_view(r):
    inf = r.get("instance_full") or {}
    ir = r.get("inferior_response")
    if isinstance(ir, dict):
        inferior = {"text": ir.get("text"), "flaw_kind": ir.get("flaw_kind"),
                    "flaw_evidence": ir.get("flaw_evidence")}
    else:
        inferior = {"text": ir, "flaw_kind": None, "flaw_evidence": None}
    tkr = inf.get("top_k_relevant_prefs") or []
    tkr_slim = [{"persona_item": x.get("persona_item"), "category": x.get("category")}
                for x in tkr[:6] if isinstance(x, dict)]
    return {
        "query_id": r.get("query_id"), "task_type": r.get("task_type"),
        "arm": inf.get("arm"), "expected_behavior": r.get("expected_behavior"),
        "user_query": r.get("user_query"), "prior_conversation": norm_prior(r.get("prior_conversation")),
        "example_response": r.get("example_response"), "inferior_response": inferior,
        "groundtruth_preference": r.get("groundtruth_preference"),
        "held_out_preference": inf.get("held_out_preference"),
        "top_k_relevant_prefs": tkr_slim,
        "privacy_flagged_prefs": inf.get("privacy_flagged_prefs") or [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="work directory for the audit run")
    ap.add_argument("--users", nargs="*", type=int, default=DEFAULT_USERS)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--batch", type=int, default=5)
    a = ap.parse_args()

    wu_dir = Path(a.out) / "workunits"
    wu_dir.mkdir(parents=True, exist_ok=True)
    manifest, counts = [], {t: 0 for t in a.tasks}
    for uid in a.users:
        card = persona_card(uid)
        d = json.load(open(ROOT / f"backend/{uid}/test.json"))
        for t in a.tasks:
            rows = [row_view(r) for r in d if r.get("task_type") == t]
            counts[t] += len(rows)
            for bi in range(0, len(rows), a.batch):
                chunk = rows[bi:bi + a.batch]
                wu_id = f"{uid}__{t}__b{bi // a.batch:02d}"
                path = wu_dir / f"wu_{wu_id}.json"
                json.dump({"work_unit_id": wu_id, "user_id": str(uid), "task_type": t,
                           "persona_card": card, "rows": chunk},
                          open(path, "w"), ensure_ascii=False, indent=1)
                manifest.append({"file": str(path), "work_unit_id": wu_id,
                                 "user_id": str(uid), "task_type": t, "n_rows": len(chunk)})
    json.dump(manifest, open(Path(a.out) / "manifest.json", "w"), indent=1)
    print("row counts:", counts, "total", sum(counts.values()))
    print("work units:", len(manifest), "->", Path(a.out) / "manifest.json")


if __name__ == "__main__":
    main()
