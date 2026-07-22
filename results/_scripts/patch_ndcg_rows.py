#!/usr/bin/env python3
"""Re-render the 3 ranking-task rows + Overall in results_tables.html with the
new graded-NDCG@5 headline, the evaluated personas. Values from recompute_ndcg
(offline join of test.json labels + results.csv rankings). Colors looked up
from the existing table's value->rgb map; row "best" recomputed. Run from repo root.
"""
import re, sys, os
sys.path.insert(0, "results/_scripts")
sys.path.insert(0, ".")
from recompute_ndcg import get_tables, MODES

HTML = "results/aggregate/html/results_tables.html"
VDIV_IDX = {3, 5}
# HTML row label -> task key (Overall handled specially)
ROWMAP = {
    "Proactive feed ranking": "personalized_recommendation",
    "@AI directive follow-up": "at_ai_directive_followup",
    "Hidden-persona recommendation": "hidden_persona_recommendation",
}
COL_ORDER = [lbl for _, lbl in MODES]  # GPT-LC, GPT-Mem, ... Opus-CC, Sonnet-CC


def color_lookup(html):
    pts = {}
    for m in re.finditer(r"background:rgb\((\d+),(\d+),(\d+)\);color:#243039\">([0-9.]+)<", html):
        pts[round(float(m[4]), 1)] = (int(m[1]), int(m[2]), int(m[3]))
    keys = sorted(pts)
    return lambda v: pts.get(round(v, 1)) or pts[min(keys, key=lambda k: abs(k - round(v, 1)))]


def cells(values, lookup):
    best = max(range(len(values)), key=lambda i: values[i])
    out = []
    for i, v in enumerate(values):
        r, g, b = lookup(v)
        cls = "val" + (" best" if i == best else "") + (" vdiv" if i in VDIV_IDX else "")
        out.append(f'<td class="{cls}" style="background:rgb({r},{g},{b});color:#243039">{v:.1f}</td>')
    return "".join(out)


def patch_row(html, lookup, label, values):
    i = html.find(f">{label}</td>")
    if i < 0:
        raise KeyError(f"row not found: {label}")
    s = html.rfind("<tr", 0, i)
    e = html.find("</tr>", i) + len("</tr>")
    old = html[s:e]
    cut = old.find(f">{label}</td>") + len(f">{label}</td>")
    new = old[:cut] + cells(values, lookup) + "</tr>"
    print(f"  {label:32s} {[round(v,1) for v in values]}  best={COL_ORDER[max(range(len(values)),key=lambda j:values[j])]}")
    return html[:s] + new + html[e:]


def main():
    per_task, overall = get_tables()
    html = open(HTML).read()
    lookup = color_lookup(html)
    # 3 ranking rows
    for label, task in ROWMAP.items():
        vals = [per_task[c][task] for c in COL_ORDER]
        html = patch_row(html, lookup, label, vals)
    # Overall
    html = patch_row(html, lookup, "Overall", [overall[c] for c in COL_ORDER])
    open(HTML, "w").write(html)
    print("wrote", HTML)


if __name__ == "__main__":
    main()
