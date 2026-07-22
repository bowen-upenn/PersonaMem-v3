#!/usr/bin/env python3
"""Inject the needle-depth-vs-accuracy sections (by days + by tokens) INTO the
main results_tables.html, reusing its styles. Idempotent: replaces any prior
block between the NEEDLE_DEPTH markers. Recomputes from results + backend.
"""
import importlib.util, statistics, collections, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results/_scripts"))
spec = importlib.util.spec_from_file_location("nd", ROOT / "results/_scripts/needle_depth_vs_accuracy.py")
nd = importlib.util.module_from_spec(spec); spec.loader.exec_module(nd)
from _htmllock import html_lock          # single-writer lock for results_tables.html

HTML = ROOT / "results/aggregate/html/results_tables.html"
START, END = "<!-- NEEDLE_DEPTH_START -->", "<!-- NEEDLE_DEPTH_END -->"

COLOR = nd.PALETTE          # shared curated, family-grouped palette


def _keep(r, tasks):
    if tasks is None:
        return True
    if tasks == "USE":
        return r.get("role") == "use"
    return r["task"] in tasks


def curve(rows, tasks, key, bins):
    by = collections.defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or not _keep(r, tasks):
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= v < hi:
                by[i].append(r["acc"]); break
    return [(statistics.mean(by[i]) if by.get(i) else None, len(by.get(i, []))) for i in range(len(bins))]


# Heatmap shade range (set per render from the actual data). The deepest cell
# keeps its existing max-saturation blue (rgb(176,209,243) reached at t=hi/100);
# everything below is stretched toward WHITE for contrast, instead of the old
# v/100 ramp that squeezed all cells into a narrow mid-blue band.
_SLO, _SHI = 0.0, 100.0


def shade(v):
    if v is None:
        return "#fff", "#aab4bb"
    lo, hi = _SLO, _SHI
    if hi <= lo:
        t = max(0.0, min(1.0, v / 100.0))
    else:
        t = max(0.0, min(1.0, (v - lo) / (hi - lo))) * (hi / 100.0)
    r = round(255 + (176 - 255) * t); g = round(255 + (209 - 255) * t); b = round(255 + (243 - 255) * t)
    return f"rgb({r},{g},{b})", "#243039"


def val_td(v, n, best=False):
    bg, fg = shade(v)
    cls = "val best" if best else "val"
    txt = "&ndash;" if v is None else f"{v:.0f}"      # no (n) clutter in cells
    return f"<td class='{cls}' style='background:{bg};color:{fg}'>{txt}</td>"


def data_table(per_model, tasks, key, bins, labels, models, head_lbl="Config"):
    cols = "".join(f"<col class='cval'>" for _ in labels) + "<col class='cval'>"
    head = "".join(f"<th class='model'><span class='m'>{lab}</span></th>" for lab in labels)
    head += "<th class='model'><span class='m'>all</span></th>"
    body = []
    for label, _ in models:
        cs = curve(per_model[label], tasks, key, bins)
        allv = [r["acc"] for r in per_model[label] if r.get(key) is not None and _keep(r, tasks)]
        best_i = max(range(len(cs)), key=lambda i: (cs[i][0] is not None, cs[i][0] or -1))
        dot = f"<span style='color:{COLOR.get(label,'#888')}'>&#9632;</span> "
        tds = "".join(val_td(v, n, best=(i == best_i)) for i, (v, n) in enumerate(cs))
        tds += val_td(statistics.mean(allv) if allv else None, len(allv))
        body.append(f"<tr class='catstart'><td class='task'>{dot}{label}</td>{tds}</tr>")
    return (f"<table><colgroup><col class='ctask'>{cols}</colgroup>"
            f"<thead><tr><th class='lbl'>{head_lbl}</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def svg(per_model, tasks, key, bins, labels, lo_lab, hi_lab, models, band=None, w=720, h=240):
    LEG = 184                         # reserved right column for the legend (no overlap)
    pl, pb, pt, pr = 34, 34, 16, 12 + LEG
    iw, ih = w - pl - pr, h - pt - pb
    # data-driven y-range: bracket the actual curve values (+ small pad) so the
    # plot doesn't waste vertical space far above/below every line.
    allv = [v for (label, _) in models for v, n in curve(per_model[label], tasks, key, bins) if v is not None]
    dlo, dhi = (min(allv), max(allv)) if allv else (0, 80)
    pad = max(1.5, (dhi - dlo) * 0.05)
    ylo = max(0, int((dlo - pad) // 5) * 5)
    yhi = min(100, int(-(-(dhi + pad) // 5)) * 5)   # ceil to a multiple of 5
    if yhi - ylo < 10:
        yhi = min(100, ylo + 10)
    gstep = 10 if (yhi - ylo) > 30 else 5
    grid = list(range(int(ylo), int(yhi) + 1, gstep))
    xs = [pl + iw * (i + 0.5) / len(labels) for i in range(len(labels))]
    edges = [pl + iw * i / len(labels) for i in range(len(labels) + 1)]
    yof = lambda v: pt + ih * (1 - (v - ylo) / (yhi - ylo))
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    if band:
        x0, x1 = edges[band[0]], edges[band[1] + 1]
        p.append(f"<rect x='{x0:.0f}' y='{pt}' width='{x1-x0:.0f}' height='{ih:.0f}' fill='#fbf3e2'/>")
        p.append(f"<text x='{(x0+x1)/2:.0f}' y='{pt+10}' font-size='8.5' fill='#c79a3a' text-anchor='middle'>middle</text>")
    for gv in grid:
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#eef1f3'/>")
        p.append(f"<text x='{pl-5}' y='{y+3:.0f}' font-size='9' fill='#aab4bb' text-anchor='end'>{gv}</text>")
    for i, lab in enumerate(labels):
        p.append(f"<text x='{xs[i]:.0f}' y='{h-pb+15}' font-size='8.5' fill='#33424b' text-anchor='middle'>{lab}</text>")
    p.append(f"<text x='{pl}' y='{h-4}' font-size='8.5' fill='#aab4bb'>{lo_lab}</text>")
    p.append(f"<text x='{pl+iw:.0f}' y='{h-4}' font-size='8.5' fill='#aab4bb' text-anchor='end'>{hi_lab}</text>")
    lx = pl + iw + 16
    for ci, (label, _) in enumerate(models):
        col = COLOR.get(label, "#888")
        cs = curve(per_model[label], tasks, key, bins)
        pts = [(xs[i], yof(v)) for i, (v, n) in enumerate(cs) if v is not None]
        d = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
        p.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='1.8'/>")
        for x, y in pts:
            p.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='2.4' fill='{col}'/>")
        ly = pt + 9 + ci * 14
        p.append(f"<line x1='{lx}' y1='{ly-3:.0f}' x2='{lx+14}' y2='{ly-3:.0f}' stroke='{col}' stroke-width='2.4'/>")
        p.append(f"<text x='{lx+19}' y='{ly:.0f}' font-size='9' fill='#33424b'>{label}</text>")
    p.append("</svg>")
    return "".join(p)


def norm_svg(per_model, tasks, models, w=540, h=215):
    """Accuracy minus each mode's own mean, over coarse day bins. Centres every
    mode at 0 so the SHAPE (depth of the middle dip) is comparable across modes
    of different overall skill. A deeper trough = stronger lost-in-the-middle."""
    labels = nd.BIN_LABELS
    pl, pb, pt, pr = 30, 30, 14, 12
    iw, ih = w - pl - pr, h - pt - pb
    lo, hi = -22, 16
    xs = [pl + iw * (i + 0.5) / len(labels) for i in range(len(labels))]
    yof = lambda v: pt + ih * (1 - (v - lo) / (hi - lo))
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for gv in (-20, -10, 0, 10):
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{w-pr}' y2='{y:.0f}' stroke='{'#c7cfd5' if gv==0 else '#eef1f3'}'/>")
        p.append(f"<text x='{pl-4}' y='{y+3:.0f}' font-size='9' fill='#aab4bb' text-anchor='end'>{gv:+d}</text>")
    for i, lab in enumerate(labels):
        p.append(f"<text x='{xs[i]:.0f}' y='{h-pb+15}' font-size='9.5' fill='#33424b' text-anchor='middle'>{lab}</text>")
    p.append(f"<text x='{pl}' y='{h-3}' font-size='8.5' fill='#aab4bb'>recent</text>")
    p.append(f"<text x='{w-pr}' y='{h-3}' font-size='8.5' fill='#aab4bb' text-anchor='end'>oldest</text>")
    for ci, (label, _) in enumerate(models):
        col = COLOR.get(label, "#888")
        cs = curve(per_model[label], tasks, "depth", nd.BINS)
        vals = [c[0] for c in cs]
        mean = sum(vals) / len(vals)
        pts = [(xs[i], yof(v - mean)) for i, v in enumerate(vals)]
        d = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
        wide = 2.4 if "Long Context" in label else 1.7
        p.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='{wide}'/>")
        for x, y in pts:
            p.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='2.4' fill='{col}'/>")
        p.append(f"<text x='{pl+6}' y='{pt+11+ci*12}' font-size='9' fill='{col}'>&#9632; {label}</text>")
    p.append("</svg>")
    return "".join(p)


def dip_row(per_model, label, tasks):
    cs = [c[0] for c in curve(per_model[label], tasks, "depth", nd.BINS)]
    mid = (cs[1] + cs[2]) / 2
    ends = (cs[0] + cs[3]) / 2
    return mid, ends, ends - mid


def main():
    per_model = {label: nd.collect(d) for label, d in nd.DAYS_MODELS}
    ALL = None  # all needle-bearing tasks (needed to populate the 10d+ distance bin)

    n_pool = sum(1 for r in per_model["GPT-5.5 Long Context"]
                 if r["depth"] is not None and _keep(r, ALL))
    # overall accuracy per mode (for the honest "LC wins overall" statement)
    import statistics as _st
    overall = {l: _st.mean([r["acc"] for r in per_model[l] if r["depth"] is not None and _keep(r, ALL)])
               for l, _ in nd.DAYS_MODELS}

    # fine token bins (≈balanced n over the Pers+Rec distribution)
    TBINS_F = [(0, 40_000), (40_000, 90_000), (90_000, 140_000), (140_000, 190_000),
               (190_000, 240_000), (240_000, 300_000), (300_000, 400_000), (400_000, 10**12)]
    TBL_F = ["<40k", "40–90k", "90–140k", "140–190k", "190–240k", "240–300k", "300–400k", "400k+"]

    # heatmap shade range from the ACTUAL cells of both tables, so low cells go
    # toward white (contrast) while the deepest cell keeps its max-saturation blue.
    global _SLO, _SHI
    _shv = []
    for label, _ in nd.DAYS_MODELS:
        _shv += [v for v, n in curve(per_model[label], ALL, "depth", nd.MIDBINS) if v is not None]
        _shv += [v for v, n in curve(per_model[label], ALL, "tok_offset", TBINS_F) if v is not None]
        _a1 = [r["acc"] for r in per_model[label] if r.get("depth") is not None and _keep(r, ALL)]
        _a2 = [r["acc"] for r in per_model[label] if r.get("tok_offset") is not None and _keep(r, ALL)]
        if _a1: _shv.append(statistics.mean(_a1))
        if _a2: _shv.append(statistics.mean(_a2))
    if _shv:
        _SLO, _SHI = min(_shv), max(_shv)

    sec = [START]
    sec.append('<section>')
    sec.append('<div class="cap"><h2>Lost in the middle &mdash; accuracy vs. query&rarr;source distance</h2>'
               '<span class="unit">accuracy %</span>'
               '<span class="note">all tasks &middot; distance from the query back to the FIRST appearance of its source information</span></div>')
    sec.append(f'<p class="lead" style="margin:0 0 18px">Each query is binned by the <b>distance from when it is asked back '
               'to the first appearance (first evidence) of its source information</b> in the context &mdash; fine near the '
               'query, wider through the middle, and a <b>10d+</b> bin for the late-tested probes. Across all 8 configs, '
               'accuracy is highest when the first evidence is recent or at the very start of the window and <b>dips '
               'through the middle</b>. The dip appears even for the memory and agentic modes &mdash; which never read the '
               'flat history &mdash; so it is largely <b>source recency / query difficulty</b>, not literal in-context '
               f'position. ({n_pool} queries/model, matched 10 personas; <b>bold</b> = each row’s peak bin.)</p>')

    # --- BY DAYS (fine near query, wider middle, 10d+) ---
    sec.append('<div class="cap"><h2 style="font-size:10.5px">By days &mdash; accuracy vs. query&rarr;source distance</h2></div>')
    sec.append(data_table(per_model, ALL, "depth", nd.MIDBINS, nd.MIDBIN_LABELS, nd.DAYS_MODELS, "Config (all modes)"))
    sec.append('<div style="margin:10px 0 6px">' +
               svg(per_model, ALL, "depth", nd.MIDBINS, nd.MIDBIN_LABELS,
                   "recent (next to query)", "10d+ (oldest)", nd.DAYS_MODELS, band=(3, 4)) + '</div>')

    # --- BY TOKENS ---
    sec.append('<div class="cap" style="margin-top:22px"><h2 style="font-size:10.5px">By tokens &mdash; history volume between the source and the query</h2>'
               '<span class="note">long-context modes: also the literal prompt position</span></div>')
    sec.append(data_table(per_model, ALL, "tok_offset", TBINS_F, TBL_F, nd.DAYS_MODELS, "Config (all modes)"))
    sec.append('<div style="margin:10px 0 6px">' +
               svg(per_model, ALL, "tok_offset", TBINS_F, TBL_F,
                   "near query", "deep / top of context", nd.DAYS_MODELS, band=(4, 5)) + '</div>')

    ov = ", ".join(f"{l.split(' ')[0] if 'Claude' not in l else l.split('-')[0]}&nbsp;{overall[l]:.0f}"
                   for l, _ in nd.DAYS_MODELS)
    sec.append('<div class="abldefs" style="max-width:820px;margin-top:14px">'
               '<b>Reading.</b> The dip in the shaded middle band (~3&ndash;7 days) is present in <b>every</b> config '
               '&mdash; including the memory and agentic modes that consolidate or retrieve rather than read the flat '
               'history. That universality means the curve is driven mostly by which source info is easy (recent, or '
               'long-standing whole-window interests) vs. hard (freshly emerged mid-window), i.e. <b>query difficulty</b>, '
               'more than literal in-context position. &nbsp;'
               f'<b>Long context is not rescued by memory:</b> overall accuracy (%): {ov}. &nbsp;'
               '<b>Note on the axis:</b> user activity spans ~9 days, so distance is &le;9d for ~95% of queries; the '
               '<b>10d+</b> bin holds the late-tested probes (over-personalization + sensitive-event-silence, ~26 queries) '
               'whose source sits 10&ndash;23 days back.</div>')
    sec.append('</section>')
    sec.append(END)
    block = "\n".join(sec)

    with html_lock():                       # single-writer: no concurrent overwrite
        html = HTML.read_text()
        if html.count(START) > 1 or html.count(END) > 1 or html.count("<!doctype") > 1:
            raise SystemExit("results_tables.html looks corrupted (duplicate markers/doctype) — "
                             "restore from /tmp/results_tables_backup.html before re-injecting")
        if START in html and END in html:
            pre = html.split(START)[0]
            post = html.split(END, 1)[1]
            html = pre + block + post
        else:
            # insert before the Ablation section (so it sits after Total tokens)
            anchor = '<section>\n<div class="cap"><h2>Ablation'
            html = html.replace(anchor, block + "\n\n" + anchor, 1)
        HTML.write_text(html)
    print("injected into", HTML)


if __name__ == "__main__":
    main()
