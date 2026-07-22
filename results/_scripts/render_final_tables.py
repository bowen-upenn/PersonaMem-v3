#!/usr/bin/env python3
"""Final render of the changed rows (Overall + chatbot + the 3
ranking tasks) using the SAME `_accuracy_value` path as aggregate_eval, so the
HTML matches the aggregate pipeline exactly for the compared modes. ndcg is
read from the written-back metrics_json. Colors from the existing table's
value->rgb map; "best" recomputed. Run from repo root.
"""
import csv, json, glob, os, re, sys
csv.field_size_limit(2**31 - 1)
sys.path.insert(0, ".")
from scripts.aggregate_eval import _accuracy_value
from evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type
from _htmllock import html_lock  # single-writer lock for results_tables.html

HTML = "results/aggregate/html/results_tables.html"
def _cohort():
    """Personas to render: $PERSONAS env override, else the personas present in every mode dir."""
    import os as _os
    env = _os.environ.get("PERSONAS", "").split()
    if env:
        return {int(u) for u in env}
    sets = []
    for m, _lbl in MODES:
        d = _os.path.join("results", m)
        if _os.path.isdir(d):
            sets.append({int(u) for u in _os.listdir(d) if u.isdigit()})
    return set.intersection(*sets) if sets else set()
VDIV_IDX = {3, 5}
MODES = [
    ("llm_longctx_gpt5.5_judged", "GPT-LC"), ("llm_memory_gpt5.5", "GPT-Mem"),
    ("mem0_gpt5.5", "GPT-Mem0"), ("codex_agent_gpt5.5", "GPT-Codex"),
    ("llm_longctx_gemini3.5flash_judged", "Gem-LC"), ("llm_memory_gemini3.5flash_judged", "Gem-Mem"),
    ("agent_tools_opus4.8", "OPUS-CC"), ("agent_tools_sonnet4.6", "Sonnet-CC"),
]
MATCHED = _cohort()
# HTML row label -> task_type (None = Overall micro over all tasks)
ROWS = {
    "Overall": None,
    "Personalized chatbot response": "chatbot_personalized_response",
    "Proactive feed ranking": "personalized_recommendation",
    "@AI directive follow-up": "at_ai_directive_followup",
    "Hidden-persona recommendation": "hidden_persona_recommendation",
}


def mode_vals(mode):
    """task_type -> mean accuracy_pct over the evaluated personas, plus '_overall'."""
    by_task = {}
    all_acc = []
    for p in glob.glob(f"results/{mode}/*/results.csv"):
        u = os.path.basename(os.path.dirname(p))
        if not u.isdigit() or int(u) not in MATCHED:
            continue
        for r in csv.DictReader(open(p)):
            if normalize_task_type(r.get("task_type", "")) in DROPPED_TASK_TYPES:
                continue  # match aggregate_eval._load_rows
            a = _accuracy_value(r["task_type"], json.loads(r.get("metrics_json") or "{}"), r.get("status") or "ok")
            if a is None:
                continue
            by_task.setdefault(r["task_type"], []).append(a)
            all_acc.append(a)
    out = {t: sum(v) / len(v) for t, v in by_task.items()}
    out["_overall"] = sum(all_acc) / len(all_acc) if all_acc else None
    return out


def color_lookup(html):
    pts = {}
    for m in re.finditer(r"background:rgb\((\d+),(\d+),(\d+)\);color:#243039\">([0-9.]+)<", html):
        pts[round(float(m[4]), 1)] = (int(m[1]), int(m[2]), int(m[3]))
    keys = sorted(pts)
    return lambda v: pts.get(round(v, 1)) or pts[min(keys, key=lambda k: abs(k - round(v, 1)))]


def cells(values, lookup):
    # Accuracy: higher is better -> bold the single top cell (top-1 policy;
    # mark_top2_bolds.py owns the canonical bold pass / restore base).
    best = max(range(len(values)), key=lambda i: values[i])
    out = []
    for i, v in enumerate(values):
        r, g, b = lookup(v)
        cls = "val" + (" best" if i == best else "") + (" vdiv" if i in VDIV_IDX else "")
        out.append(f'<td class="{cls}" style="background:rgb({r},{g},{b});color:#243039">{v:.1f}</td>')
    return "".join(out)


def patch_row(html, lookup, label, values):
    i = html.find(f">{label}</td>")
    s = html.rfind("<tr", 0, i)
    e = html.find("</tr>", i) + len("</tr>")
    old = html[s:e]
    cut = old.find(f">{label}</td>") + len(f">{label}</td>")
    new = old[:cut] + cells(values, lookup) + "</tr>"
    best = MODES[max(range(len(values)), key=lambda j: values[j])][1]
    print(f"  {label:32s} {[round(v,1) for v in values]}  best={best}")
    return html[:s] + new + html[e:]


def main():
    allv = {lbl: mode_vals(mode) for mode, lbl in MODES}
    # Hold the single-writer lock across the whole read-modify-write so a
    # concurrent section renderer can't carry a stale Accuracy table forward.
    with html_lock():
        html = open(HTML).read()
        lookup = color_lookup(html)
        for label, task in ROWS.items():
            vals = [(allv[lbl]["_overall"] if task is None else allv[lbl].get(task)) for _, lbl in MODES]
            html = patch_row(html, lookup, label, vals)
        open(HTML, "w").write(html)
    print("wrote", HTML)


if __name__ == "__main__":
    main()
