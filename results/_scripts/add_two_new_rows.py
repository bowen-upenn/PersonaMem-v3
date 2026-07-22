#!/usr/bin/env python3
"""Surgically add two new task rows to the Accuracy / Latency / Total-tokens
tables in results_tables.html. Codex column left BLANK (user runs it separately).
Holds the single-writer lock. Re-run mark_top2_bolds.py afterward."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _htmllock import html_lock, HTML

# Column order: GPT-LC, GPT-Mem, GPT-Mem0, GPT-Codex(BLANK), Gem-LC, Gem-Mem, OPUS-CC, Sonnet-CC
B = None  # blank (codex)
# task -> {table: [8 values]}
DATA = {
    # Recommendation family
    "Short-term preference lifecycle": {
        "acc": [56.4, 46.2, 57.7, B, 51.3, 44.9, 46.2, 52.6],
        "lat": [30.6, 23.5, 13.1, B, 37.6, 10.4, 113.6, 177.1],   # seconds
        "tok": [364.5, 4.1, 1.6, B, 364.5, 4.0, 488.9, 205.4],    # thousands
        "after": "Hidden-persona recommendation", "family": "Recommendation",
    },
    # Personalization family
    "Fresh chatbot suggestion": {
        "acc": [0.0, 0.0, 10.0, B, 0.0, 0.0, 10.0, 0.0],
        "lat": [16.1, 14.6, 15.5, B, 20.2, 11.6, 87.0, 169.0],
        "tok": [132.8, 2.0, 1.1, B, 132.8, 2.0, 261.8, 42.7],
        "after": "Personal-fact hallucination probe", "family": "Personalization",
    },
}
# (h2, key, fmt, higher_is_better) — accuracy higher=best; latency/tokens lower=best
TABLES = [("<h2>Accuracy</h2>", "acc", "{:.1f}", True),
          ("<h2>Latency</h2>", "lat", "{:.1f}s", False),
          ("<h2>Total tokens</h2>", "tok", "{:.1f}k", False)]
VDIV = {3, 5}


def color_map(table_html):
    """value(float) -> rgb tuple, from this table's existing cells."""
    pts = {}
    for m in re.finditer(r'background:rgb\((\d+),(\d+),(\d+)\)[^>]*>([0-9.]+)[sk]?<', table_html):
        pts[float(m.group(4))] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    keys = sorted(pts)
    def lookup(v):
        if not keys:
            return (255, 255, 255)
        return pts.get(v) or pts[min(keys, key=lambda k: abs(k - v))]
    return lookup


def build_row(label, values, fmt, lookup, higher_better):
    # top-2 bold among non-blank cells (direction-aware)
    idx = [i for i, v in enumerate(values) if v is not None]
    top2 = set(sorted(idx, key=lambda i: values[i], reverse=higher_better)[:2])
    cells = []
    for i, v in enumerate(values):
        vd = " vdiv" if i in VDIV else ""
        if v is None:  # codex blank
            cells.append(f'<td class="val{vd}" style="background:rgb(255,255,255);color:#243039"></td>')
        else:
            r, g, b = lookup(v)
            bestcls = " best" if i in top2 else ""
            cells.append(f'<td class="val{bestcls}{vd}" style="background:rgb({r},{g},{b});color:#243039">{fmt.format(v)}</td>')
    return f"<tr><td class='task tdiv'>{label}</td>{''.join(cells)}</tr>"


def insert_into_table(table_html, label, spec, key, fmt, higher_better):
    lookup = color_map(table_html)
    new_row = build_row(label, spec[key], fmt, lookup, higher_better)
    # bump the family's catstart rowspan
    fam = spec["family"]
    m = re.search(rf'(<td class="cat" rowspan=")(\d+)(">{re.escape(fam)}</td>)', table_html)
    if not m:
        raise SystemExit(f"family {fam!r} catstart not found")
    table_html = table_html[:m.start()] + f'{m.group(1)}{int(m.group(2))+1}{m.group(3)}' + table_html[m.end():]
    # insert new row right after the </tr> of the `after` row
    anchor = f">{spec['after']}</td>"
    ai = table_html.find(anchor)
    end = table_html.find("</tr>", ai) + len("</tr>")
    return table_html[:end] + new_row + table_html[end:]


def main():
    with html_lock():
        html = open(HTML).read()
        for h2, key, fmt, hb in TABLES:
            start = html.find(h2)
            nxt = html.find("<h2>", start + len(h2))
            seg = html[start:nxt]
            for label, spec in DATA.items():
                seg = insert_into_table(seg, label, spec, key, fmt, hb)
            html = html[:start] + seg + html[nxt:]
        open(HTML, "w").write(html)
    print("inserted 2 rows x 3 tables")


if __name__ == "__main__":
    main()
