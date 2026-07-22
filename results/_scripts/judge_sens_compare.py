#!/usr/bin/env python3
"""Compare per-task accuracy under the live GPT-5.5 judge vs the two alt judges
(gpt-5.4-mini, claude-opus-4.8) for the judge-sensitivity study.

Baseline = pr_query_score_0_10 in each live results.csv (GPT-5.5, identical
rejudge_existing code path). Alt judges = the by_task avg_pct in
/tmp/eval_regen/judge_sens/{judge}.{cfg}.json. Both are row-weighted means over
the evaluated personas, scored on the SAME saved responses, so the delta is a
clean judge-only effect.

Reports, per config: each pr.score task's accuracy under the 3 judges + deltas,
and the macro (task-weighted) shift. Output: console table + a JSON dump.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.personalization_rubric import APPLICABILITY

OUT = Path("/tmp/eval_regen/judge_sens/p3")
MATCHED = ["1", "2", "3"]  # 3-persona agreement sample
CFGS = ["llm_longctx_gpt5.5_judged", "llm_memory_gpt5.5", "mem0_gpt5.5",
        "codex_agent_gpt5.5", "llm_longctx_gemini3.5flash_judged",
        "llm_memory_gemini3.5flash_judged", "agent_tools_opus4.8",
        "agent_tools_sonnet4.6"]
ALT = ["gpt-5.4-mini", "claude-opus-4.8"]
SCOPE = set(APPLICABILITY) - {"over_personalization_repetition_recsys",
    "over_personalization_repetition_chatbot", "new_suggestions_recsys",
    "new_suggestions_chatbot"}


def baseline_by_task(cfg):
    """Row-weighted GPT-5.5 avg_pct per task from the live csvs (evaluated personas)."""
    vals = defaultdict(list)
    for u in MATCHED:
        rf = ROOT / "results" / cfg / u / "results.csv"
        if not rf.exists():
            continue
        for row in csv.DictReader(open(rf)):
            if (row.get("status") or "").strip() != "ok":
                continue
            tt = row.get("task_type")
            if tt not in SCOPE:
                continue
            try:
                m = json.loads(row.get("metrics_json") or "{}")
            except Exception:
                continue
            s = m.get("pr_query_score_0_10")
            if isinstance(s, (int, float)):
                vals[tt].append(float(s))
    return {tt: 10 * sum(v) / len(v) for tt, v in vals.items() if v}, \
           {tt: len(v) for tt, v in vals.items() if v}


def alt_by_task(judge, cfg):
    f = OUT / f"{judge}.{cfg}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {r["task_type"]: r["avg_pct"] for r in d.get("by_task", [])}


report = {}
print(f"\n{'='*92}\nJUDGE SENSITIVITY — accuracy (avg rubric x10) per pr.score task, evaluated personas\n{'='*92}")
for cfg in CFGS:
    base, ns = baseline_by_task(cfg)
    alts = {j: alt_by_task(j, cfg) for j in ALT}
    if not base or any(a is None for a in alts.values()):
        miss = [j for j, a in alts.items() if a is None]
        print(f"\n## {cfg}: INCOMPLETE (missing alt summaries: {miss or 'baseline empty'})")
        continue
    print(f"\n## {cfg}")
    print(f"{'task':42s} {'n':>3s} {'GPT5.5':>7s} {'5.4mini':>8s} {'Δ':>6s} "
          f"{'Opus4.8':>8s} {'Δ':>6s}")
    rows = []
    for tt in sorted(base):
        b = base[tt]
        a1 = alts['gpt-5.4-mini'].get(tt)
        a2 = alts['claude-opus-4.8'].get(tt)
        if a1 is None or a2 is None:
            continue
        d1, d2 = a1 - b, a2 - b
        rows.append((tt, ns[tt], b, a1, d1, a2, d2))
        print(f"{tt:42s} {ns[tt]:>3d} {b:>7.1f} {a1:>8.1f} {d1:>+6.1f} {a2:>8.1f} {d2:>+6.1f}")
    if rows:
        mb = sum(r[2] for r in rows) / len(rows)
        m1 = sum(r[3] for r in rows) / len(rows)
        m2 = sum(r[5] for r in rows) / len(rows)
        mad1 = sum(abs(r[4]) for r in rows) / len(rows)
        mad2 = sum(abs(r[6]) for r in rows) / len(rows)
        print(f"{'-'*82}")
        print(f"{'MACRO (task-weighted)':42s} {'':>3s} {mb:>7.1f} {m1:>8.1f} {m1-mb:>+6.1f} "
              f"{m2:>8.1f} {m2-mb:>+6.1f}")
        print(f"{'mean abs per-task shift (MAD)':42s} {'':>3s} {'':>7s} {'':>8s} {mad1:>6.1f} "
              f"{'':>8s} {mad2:>6.1f}")
        report[cfg] = {"macro_gpt5.5": round(mb, 1), "macro_gpt5.4mini": round(m1, 1),
                       "macro_opus4.8": round(m2, 1), "macro_delta_5.4mini": round(m1 - mb, 1),
                       "macro_delta_opus4.8": round(m2 - mb, 1),
                       "mad_5.4mini": round(mad1, 1), "mad_opus4.8": round(mad2, 1),
                       "per_task": [{"task": r[0], "n": r[1], "gpt5.5": round(r[2], 1),
                                     "gpt5.4mini": round(r[3], 1), "d_5.4mini": round(r[4], 1),
                                     "opus4.8": round(r[5], 1), "d_opus4.8": round(r[6], 1)}
                                    for r in rows]}

(OUT / "comparison.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT/'comparison.json'}")
