#!/usr/bin/env python3
"""Patch the 3 accuracy-table rows affected by the headline-metric change
(chatbot_personalized_response -> pref_align_gated; at_ai_directive_followup
-> recall@3) directly in results/aggregate/html/results_tables.html.

Only value text, per-cell heat-map background, and the row's "best" highlight
change. Colors are looked up from the existing table's empirical value->rgb
map (nearest value) so they match the original generator exactly — no formula
reverse-engineering. Run from repo root.
"""
import csv, json, os, re

HTML = "results/aggregate/html/results_tables.html"
AGG = "results/aggregate"

# Column order in the accuracy table -> aggregate mode dir.
COLS = [
    "llm_longctx_gpt5.5_judged",     # GPT-5.5 Long Context
    "llm_memory_gpt5.5",             # GPT-5.5 Textual Memory
    "mem0_gpt5.5",                   # GPT-5.5 Mem0
    "codex_agent_gpt5.5",            # GPT-5.5 Codex High  (vdiv after this, idx 3)
    "llm_longctx_gemini3.5flash_judged",  # Gemini Long Context
    "llm_memory_gemini3.5flash_judged",   # Gemini Textual Memory (vdiv after this, idx 5)
    "agent_tools_opus4.8",           # Opus-4.8 Claude Code High
    "agent_tools_sonnet4.6",         # Sonnet-4.6 Claude Code High
]
VDIV_IDX = {3, 5}  # value-column indices that carry the "vdiv" group separator


def task_acc(mode, task):
    p = f"{AGG}/{mode}/token_accuracy_table.csv"
    for r in csv.DictReader(open(p)):
        if r.get("task_type") == task:
            return float(r["accuracy_pct"])
    raise KeyError(f"{task} not in {p}")


def overall_acc(mode):
    return float(json.load(open(f"{AGG}/{mode}/summary_overall.json"))["accuracy_pct_micro"])


def new_values(task=None, overall=False):
    if overall:
        return [overall_acc(m) for m in COLS]
    return [task_acc(m, task) for m in COLS]


def build_color_lookup(html):
    """value(1dp) -> (r,g,b) from every heat-map cell already in the table."""
    pts = {}
    for m in re.finditer(r"background:rgb\((\d+),(\d+),(\d+)\);color:#243039\">([0-9.]+)<", html):
        r, g, b, v = int(m[1]), int(m[2]), int(m[3]), round(float(m[4]), 1)
        pts[v] = (r, g, b)
    keys = sorted(pts)
    def lookup(v):
        v = round(v, 1)
        if v in pts:
            return pts[v]
        nearest = min(keys, key=lambda k: abs(k - v))
        return pts[nearest]
    return lookup


def render_cells(values, lookup):
    best_i = max(range(len(values)), key=lambda i: values[i])
    out = []
    for i, v in enumerate(values):
        r, g, b = lookup(v)
        cls = "val"
        if i == best_i:
            cls += " best"
        if i in VDIV_IDX:
            cls += " vdiv"
        out.append(f'<td class="{cls}" style="background:rgb({r},{g},{b});color:#243039">{v:.1f}</td>')
    return "".join(out)


def patch_row(html, lookup, task_label, values, lead_html):
    """Replace the value cells of the row whose task cell == task_label."""
    i = html.find(f">{task_label}</td>")
    if i < 0:
        raise KeyError(f"row not found: {task_label}")
    s = html.rfind("<tr", 0, i)
    e = html.find("</tr>", i) + len("</tr>")
    old_row = html[s:e]
    # keep everything up to and including the task <td>, replace the rest with cells
    cut = old_row.find(f">{task_label}</td>") + len(f">{task_label}</td>")
    new_row = old_row[:cut] + render_cells(values, lookup) + "</tr>"
    return html[:s] + new_row + html[e:], old_row, new_row


def main():
    html = open(HTML).read()
    lookup = build_color_lookup(html)

    rows = [
        ("Overall", new_values(overall=True)),
        ("Personalized chatbot response", new_values(task="chatbot_personalized_response")),
        ("@AI directive follow-up", new_values(task="at_ai_directive_followup")),
    ]
    for label, vals in rows:
        html, old_row, new_row = patch_row(html, lookup, label, vals, None)
        print(f"=== {label} ===")
        print(f"  new values: {[round(v,1) for v in vals]}  best_col={COLS[max(range(len(vals)),key=lambda i:vals[i])]}")

    open(HTML, "w").write(html)
    print("\nwrote", HTML)


if __name__ == "__main__":
    main()
