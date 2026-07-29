"""Compare the two arms of the cross-platform context ablation.

Reads results/ablation_platform_context/{full_ctx,single_platform}/{uid}/results.csv,
computes row-level accuracy via the shared `_accuracy_value` from
scripts/aggregate_eval.py, PAIRS rows by query_id (only rows status=ok with a
non-empty response in BOTH arms count — per EVAL.md, cross-config comparisons
exclude empty-response rows and report kept/dropped), and emits per-task /
per-scenario / overall micro accuracy for each arm plus the delta.

Output: results/ablation_platform_context/aggregate/platform_ablation_comparison.{csv,md}
Read-only over results; does NOT touch results_tables.html (no lock needed).

Usage: python scripts/compare_platform_ablation.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
csv.field_size_limit(10_000_000)

from scripts.aggregate_eval import _accuracy_value  # noqa: E402
from evaluation.task_registry import normalize_task_type  # noqa: E402

ROOT = REPO_ROOT / "results" / "ablation_platform_context"
ARMS = ("full_ctx", "single_platform")
USERS = ("1", "2", "3", "5", "6", "8", "9", "10", "13", "14")

CHATBOT_SCENARIO = {
    "chatbot_personalized_response", "new_suggestions_chatbot",
    "local_recommendation_geo_shift", "personal_qa_hallucination",
}
RECSYS_SCENARIO = {
    "personalized_recommendation", "at_ai_directive_followup",
    "short_vs_long_term_lifecycle",
}


def scenario_of(task: str) -> str:
    if task in CHATBOT_SCENARIO:
        return "chatbot"
    if task in RECSYS_SCENARIO:
        return "social_feed_recommendation"
    return "other"


def load_arm(arm: str) -> dict[str, dict]:
    """{query_id: {task, acc, ok, empty}} across all personas."""
    out: dict[str, dict] = {}
    for uid in USERS:
        p = ROOT / arm / uid / "results.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                task = normalize_task_type(r.get("task_type") or "")
                try:
                    metrics = json.loads(r.get("metrics_json") or "{}")
                except Exception:
                    metrics = {}
                status = r.get("status") or ""
                acc = _accuracy_value(task, metrics, status)
                out[r["query_id"]] = {
                    "task": task,
                    "user_id": uid,
                    "acc": acc,
                    "ok": status == "ok",
                    "empty": not (r.get("agent_response") or "").strip(),
                }
    return out


def main() -> int:
    arms = {a: load_arm(a) for a in ARMS}
    for a, rows in arms.items():
        if not rows:
            print(f"[compare] arm {a}: no results found under {ROOT / a}", file=sys.stderr)
            return 2

    all_qids = set(arms[ARMS[0]]) | set(arms[ARMS[1]])
    paired, dropped = [], defaultdict(int)
    for qid in sorted(all_qids):
        a = arms[ARMS[0]].get(qid)
        b = arms[ARMS[1]].get(qid)
        if a is None or b is None:
            dropped["missing_in_one_arm"] += 1
            continue
        if not (a["ok"] and b["ok"]):
            dropped["non_ok"] += 1
            continue
        if a["empty"] or b["empty"]:
            dropped["empty_response"] += 1
            continue
        if a["acc"] is None or b["acc"] is None:
            dropped["no_accuracy_metric"] += 1
            continue
        paired.append((qid, a, b))

    def micro(rows_sel, which):
        vals = [x[1]["acc"] if which == 0 else x[2]["acc"] for x in rows_sel]
        return 100.0 * sum(vals) / len(vals) if vals else float("nan")

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for qid, a, b in paired:
        groups[("task", a["task"])].append((qid, a, b))
        groups[("scenario", scenario_of(a["task"]))].append((qid, a, b))
    groups[("overall", "all_in_scope")] = list(paired)

    out_rows = []
    for (level, name), sel in sorted(groups.items()):
        f, s = micro(sel, 0), micro(sel, 1)
        out_rows.append({
            "level": level, "name": name, "n_paired": len(sel),
            "full_ctx_acc": round(f, 2), "single_platform_acc": round(s, 2),
            "delta_single_minus_full": round(s - f, 2),
        })

    agg_dir = ROOT / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    csv_path = agg_dir / "platform_ablation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    md = ["# Cross-platform context ablation — full vs current-platform history",
          "", f"Paired rows: {len(paired)}  |  dropped: {dict(dropped) or 'none'}", "",
          "| level | group | n | full-ctx acc % | single-platform acc % | Δ (single − full) |",
          "|---|---|---|---|---|---|"]
    for r in out_rows:
        md.append(f"| {r['level']} | {r['name']} | {r['n_paired']} | "
                  f"{r['full_ctx_acc']} | {r['single_platform_acc']} | "
                  f"{r['delta_single_minus_full']:+.2f} |")
    (agg_dir / "platform_ablation_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[compare] paired={len(paired)} dropped={dict(dropped)}")
    for r in out_rows:
        print(f"  {r['level']:<9} {r['name']:<38} n={r['n_paired']:<4} "
              f"full={r['full_ctx_acc']:>6}  single={r['single_platform_acc']:>6}  "
              f"Δ={r['delta_single_minus_full']:+.2f}")
    print(f"[compare] wrote {csv_path} + .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
