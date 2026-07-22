#!/usr/bin/env python3
"""Re-mark the single TOP cell per data row as bold ("best") in the first three
tables of results_tables.html (Accuracy, Latency, Total tokens).

Direction-aware:
  table 0 (Accuracy)      -> higher is better  -> bold the largest
  table 1 (Latency)       -> lower is better   -> bold the smallest ("12.3s")
  table 2 (Total tokens)  -> lower is better   -> bold the smallest ("12.3k")

On a tie the FIRST (left-most) column is bolded (stable sort). Only rows with the
full 8 model columns are touched. Applies to the live HTML (under the single-
writer lock) AND the restore base, so a restore keeps top-1. Run from repo root.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _htmllock import html_lock, HTML

VAL_TD = re.compile(r'<td class="(?P<cls>[^"]*\bval\b[^"]*)"(?P<rest>[^>]*)>(?P<txt>[^<]*)</td>')
# table index -> True means higher-is-better (largest 2), False means lower (smallest 2)
HIGHER_BETTER = {0: True, 1: False, 2: False}


def _num(txt):
    s = re.sub(r"[^0-9.\-]", "", txt)
    return float(s) if s not in ("", "-", ".") else None


def remark_table(table_html, higher_better):
    rows = []
    last = 0
    for rm in re.finditer(r"<tr>.*?</tr>", table_html, re.S):
        rows.append(table_html[last:rm.start()])
        row = rm.group(0)
        cells = list(VAL_TD.finditer(row))
        vals = [_num(c.group("txt")) for c in cells]
        if len(cells) >= 8 and all(v is not None for v in vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=higher_better)
            top1 = set(order[:1])
            counter = {"i": -1}

            def repl(m):
                counter["i"] += 1
                i = counter["i"]
                had_vdiv = "vdiv" in m.group("cls")
                cls = "val" + (" best" if i in top1 else "") + (" vdiv" if had_vdiv else "")
                return f'<td class="{cls}"{m.group("rest")}>{m.group("txt")}</td>'

            row = VAL_TD.sub(repl, row)
        rows.append(row)
        last = rm.end()
    rows.append(table_html[last:])
    return "".join(rows)


def _remark_html(html):
    spans = [(m.start(), m.end()) for m in re.finditer(r"<table.*?</table>", html, re.S)]
    out, prev, changed = [], 0, 0
    for ti, (s, e) in enumerate(spans):
        out.append(html[prev:s])
        block = html[s:e]
        if ti in HIGHER_BETTER:               # only the first three tables
            new = remark_table(block, HIGHER_BETTER[ti])
            changed += int(new != block)
            out.append(new)
        else:
            out.append(block)
        prev = e
    out.append(html[prev:])
    return "".join(out), changed

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results_tables_base.html")


def main():
    with html_lock():
        new, changed = _remark_html(open(HTML).read())
        open(HTML, "w").write(new)
    print(f"top-1 bold: re-marked {changed} of first-3 tables in live HTML")
    if os.path.exists(BASE):                   # keep the restore base consistent
        new, c = _remark_html(open(BASE).read())
        open(BASE, "w").write(new)
        print(f"top-1 bold: re-marked {c} of first-3 tables in restore base")


if __name__ == "__main__":
    main()
