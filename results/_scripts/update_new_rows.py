#!/usr/bin/env python3
"""Patch the two new rows in results_tables.html: fill the codex column (now run)
and update the Fresh chatbot suggestion accuracy with the re-scored numbers.
Re-bolds top-2 direction-aware, recolors from each table's scale, holds the lock."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _htmllock import html_lock, HTML

# col order: GPT-LC, GPT-Mem, GPT-Mem0, Codex, Gem-LC, Gem-Mem, OPUS, Sonnet
DATA = {
    "Short-term preference lifecycle": {
        "acc": [56.4, 46.2, 57.7, 60.3, 51.3, 44.9, 46.2, 52.6],
        "lat": [30.6, 23.5, 13.1, 74.4, 37.6, 10.4, 113.6, 177.1],
        "tok": [364.5, 4.1, 1.6, 207.1, 364.5, 4.0, 488.9, 205.4],
    },
    # n=40 (8/persona x 5); codex (idx 3) unchanged — user reruns its 40 instances
    "Fresh chatbot suggestion": {
        "acc": [62.5, 47.5, 42.5, 30.0, 40.0, 50.0, 50.0, 27.5],
        "lat": [26.7, 22.0, 17.5, 41.3, 25.8, 17.1, 87.3, 89.6],
        "tok": [202.5, 2.6, 1.2, 80.8, 202.5, 2.6, 246.9, 92.0],
    },
}
TABLES = [("<h2>Accuracy</h2>", "acc", "{:.1f}", True),
          ("<h2>Latency</h2>", "lat", "{:.1f}s", False),
          ("<h2>Total tokens</h2>", "tok", "{:.1f}k", False)]
VDIV = {3, 5}


def color_map(seg):
    pts = {}
    for m in re.finditer(r'background:rgb\((\d+),(\d+),(\d+)\)[^>]*>([0-9.]+)[sk]?<', seg):
        pts[float(m.group(4))] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    keys = sorted(pts)
    return lambda v: (255, 255, 255) if not keys else (pts.get(v) or pts[min(keys, key=lambda k: abs(k - v))])


def cells(values, fmt, lookup, higher_better):
    top2 = set(sorted(range(len(values)), key=lambda i: values[i], reverse=higher_better)[:2])
    out = []
    for i, v in enumerate(values):
        vd = " vdiv" if i in VDIV else ""
        bc = " best" if i in top2 else ""
        r, g, b = lookup(v)
        out.append(f'<td class="val{bc}{vd}" style="background:rgb({r},{g},{b});color:#243039">{fmt.format(v)}</td>')
    return "".join(out)


def main():
    with html_lock():
        html = open(HTML).read()
        for h2, key, fmt, hb in TABLES:
            s = html.find(h2); e = html.find("<h2>", s + len(h2)); seg = html[s:e]
            lookup = color_map(seg)
            for label, spec in DATA.items():
                anchor = f">{label}</td>"
                i = seg.find(anchor)
                cells_start = i + len(anchor)
                row_end = seg.find("</tr>", cells_start)
                seg = seg[:cells_start] + cells(spec[key], fmt, lookup, hb) + seg[row_end:]
            html = html[:s] + seg + html[e:]
        open(HTML, "w").write(html)
    print("patched codex column + chatbot accuracy in all 3 tables")


if __name__ == "__main__":
    main()
