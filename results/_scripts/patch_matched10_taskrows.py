#!/usr/bin/env python3
"""Rebuild ONLY the two headline-changed task rows (chatbot_personalized_response,
at_ai_directive_followup) in results_tables.html on the MATCHED 10-persona set
{1,2,3,5,6,8,9,10,13,14} — fair cross-model/mode comparison (no 20p/5p mixing).

Values come straight from results.csv via the real aggregator metric
(_accuracy_value), restricted to the matched personas. Colors are looked up
from the existing table's empirical value->rgb map; the row "best" highlight is
recomputed. Overall and every other row are left untouched. Run from repo root.
"""
import csv, glob, os, re, sys
csv.field_size_limit(10**9)
sys.path.insert(0, ".")
import json
from scripts.aggregate_eval import _accuracy_value

HTML = "results/aggregate/html/results_tables.html"
MATCHED = {1, 2, 3, 5, 6, 8, 9, 10, 13, 14}
COLS = [
    "llm_longctx_gpt5.5_judged", "llm_memory_gpt5.5", "mem0_gpt5.5",
    "codex_agent_gpt5.5", "llm_longctx_gemini3.5flash_judged",
    "llm_memory_gemini3.5flash_judged", "agent_tools_opus4.8", "agent_tools_sonnet4.6",
]
VDIV_IDX = {3, 5}
ROWS = {
    "Personalized chatbot response": "chatbot_personalized_response",
    "@AI directive follow-up": "at_ai_directive_followup",
}


def matched10_acc(mode, task):
    vals = []
    for p in glob.glob(f"results/{mode}/*/results.csv"):
        uid = os.path.basename(os.path.dirname(p))
        if not uid.isdigit() or int(uid) not in MATCHED:
            continue
        for row in csv.DictReader(open(p)):
            if row.get("task_type") != task:
                continue
            mj = json.loads(row.get("metrics_json") or "{}")
            v = _accuracy_value(task, mj, row.get("status") or "ok")
            if v is not None:
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def build_color_lookup(html):
    pts = {}
    for m in re.finditer(r"background:rgb\((\d+),(\d+),(\d+)\);color:#243039\">([0-9.]+)<", html):
        pts[round(float(m[4]), 1)] = (int(m[1]), int(m[2]), int(m[3]))
    keys = sorted(pts)
    return lambda v: pts.get(round(v, 1)) or pts[min(keys, key=lambda k: abs(k - round(v, 1)))]


def render_cells(values, lookup):
    best_i = max(range(len(values)), key=lambda i: values[i])
    out = []
    for i, v in enumerate(values):
        r, g, b = lookup(v)
        cls = "val" + (" best" if i == best_i else "") + (" vdiv" if i in VDIV_IDX else "")
        out.append(f'<td class="{cls}" style="background:rgb({r},{g},{b});color:#243039">{v:.1f}</td>')
    return "".join(out)


def patch_row(html, lookup, label, task):
    vals = [matched10_acc(m, task) for m in COLS]
    i = html.find(f">{label}</td>")
    s = html.rfind("<tr", 0, i)
    e = html.find("</tr>", i) + len("</tr>")
    old = html[s:e]
    cut = old.find(f">{label}</td>") + len(f">{label}</td>")
    new = old[:cut] + render_cells(vals, lookup) + "</tr>"
    print(f"  {label}: {[round(v,1) for v in vals]}  best={COLS[max(range(len(vals)),key=lambda i:vals[i])]}")
    return html[:s] + new + html[e:]


def main():
    html = open(HTML).read()
    lookup = build_color_lookup(html)
    for label, task in ROWS.items():
        html = patch_row(html, lookup, label, task)
    open(HTML, "w").write(html)
    print("wrote", HTML)


if __name__ == "__main__":
    main()
