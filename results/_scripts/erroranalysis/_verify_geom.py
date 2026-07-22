#!/usr/bin/env python3
"""Verification only: render the 8 failure donuts + merged legend as an SVG->PNG
(cairosvg) using the EXACT colors/labels/mapping from build_html_section, to eye
the desaturated-pride palette before shipping."""
import json, math, os, sys
from collections import Counter
import cairosvg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_html_section as B

HERE = os.path.dirname(os.path.abspath(__file__))
fail = [json.loads(l) for l in open(f"{HERE}/perrow_failures.jsonl")]
stats = json.load(open(f"{HERE}/model_stats.json"))
F = {k: Counter(B.fcat(k, r["cause"]) for r in fail if r["key"] == k) for k in B.LABELS}
ORDER = [k for g in B.GROUPS for k in g]


def pt(cx, cy, rho, deg):
    t = math.radians(deg)
    return cx + rho * math.sin(t), cy - rho * math.cos(t)


def sector(cx, cy, R, r, a0, a1):
    lg = 1 if (a1 - a0) > 180 else 0
    x0o, y0o = pt(cx, cy, R, a0); x1o, y1o = pt(cx, cy, R, a1)
    x1i, y1i = pt(cx, cy, r, a1); x0i, y0i = pt(cx, cy, r, a0)
    return (f"M{x0o:.2f},{y0o:.2f} A{R},{R} 0 {lg} 1 {x1o:.2f},{y1o:.2f} "
            f"L{x1i:.2f},{y1i:.2f} A{r},{r} 0 {lg} 0 {x0i:.2f},{y0i:.2f} Z")


def donut(cx, cy, R, counts, title):
    r = R * 0.54
    total = sum(counts.values()) or 1
    p = [f'<text x="{cx}" y="{cy-R-14}" font-size="15" font-weight="700" text-anchor="middle" fill="#16242c">{title}</text>']
    cum = 0.0
    for k in B.CAUSES:
        v = counts.get(k, 0)
        if v <= 0:
            continue
        f = 100 * v / total
        end = cum + f
        se = end - 0.45
        p.append(f'<path d="{sector(cx,cy,R,r,cum/100*360,se/100*360)}" fill="{B.CAUSE_COLOR[k]}"/>')
        if f >= 7:
            th = math.radians((cum + end) / 2 / 100 * 360)
            lx, ly = pt(cx, cy, (R + r) / 2, (cum + end) / 2 / 100 * 360)
            p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="13" font-weight="700" fill="#fff" text-anchor="middle" dominant-baseline="central">{f:.0f}%</text>')
        cum = end
    return "".join(p)


W, H = 1500, 760
parts = ['<rect width="%d" height="%d" fill="#fff"/>' % (W, H)]
R = 95
xs = [150, 410, 670, 930]
for i, key in enumerate(ORDER):
    col = i % 4; row = i // 4
    cx = xs[col]; cy = 130 + row * 290
    m, d = B.LABELS[key]
    parts.append(donut(cx, cy, R, F[key], f"{m} / {d}"))
# legend
import re
lx = 1140; ly = 70
parts.append(f'<text x="{lx}" y="{ly}" font-size="16" font-weight="700" fill="#9e2b2b">Why it got answers wrong</text>')
yy = ly + 30
for k in B.CAUSES:
    lab = re.sub("&[a-z]+;", "'", B.CAUSE_LABEL[k])
    parts.append(f'<rect x="{lx}" y="{yy-13}" width="16" height="16" rx="3" fill="{B.CAUSE_COLOR[k]}"/>')
    parts.append(f'<text x="{lx+24}" y="{yy}" font-size="14" fill="#33424b" dominant-baseline="central">{lab}</text>')
    yy += 30
svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(parts)}</svg>'
cairosvg.svg2png(bytestring=svg.encode(), write_to="/tmp/ea_palette_check.png", output_width=1500)
print("wrote /tmp/ea_palette_check.png")
for key in ORDER:
    print(key, dict(F[key]))
