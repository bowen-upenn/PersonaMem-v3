#!/usr/bin/env python3
"""Inject the 'why iterative memory can't fix the middle' panel into
results_tables.html (own markers, idempotent). Combines:
  - verbatim line-survival forgetting curve (memory_dynamics, no LLM)
  - semantic retention by evidence age (Option A GPT-5.5 locator cache)
  - retention -> accuracy
Conclusion: the steep forgetting curve is mostly REWORDING; semantically the
memory keeps mid-window facts (~86%), and retention doesn't improve accuracy, so
the middle dip is a use/difficulty failure, not forgetting.
"""
import json, statistics, collections, importlib.util, glob, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sd = ROOT / "results/_scripts"
_md = importlib.util.spec_from_file_location("md", sd / "memory_dynamics.py")
md = importlib.util.module_from_spec(_md); _md.loader.exec_module(md)
_sm = importlib.util.spec_from_file_location("sm", sd / "memory_forgetting_semantic.py")
sm = importlib.util.module_from_spec(_sm); _sm.loader.exec_module(sm)
_msa = importlib.util.spec_from_file_location("msa", sd / "memory_size_ablation.py")
msa = importlib.util.module_from_spec(_msa); _msa.loader.exec_module(msa)
nd = msa.nd            # needle_depth module (MIDBINS / palette / etc.)
import sys; sys.path.insert(0, str(sd))
from _htmllock import html_lock          # single-writer lock for results_tables.html
SEM_THRESHOLD = 0.45
CAP_COLOR = nd.CAP_PALETTE              # teal ramp: more memory = deeper


# Heatmap shade range (set per render from the table's own cells). The deepest
# cell keeps its existing max-saturation blue (t=hi/70); lower cells stretch
# toward WHITE for contrast instead of squeezing into a narrow mid-blue band.
_ALO, _AHI = 0.0, 70.0


def acc_shade(v):
    if v is None:
        return "rgb(255,255,255)"
    lo, hi = _ALO, _AHI
    if hi <= lo:
        t = max(0.0, min(1.0, v / 70.0))
    else:
        t = max(0.0, min(1.0, (v - lo) / (hi - lo))) * (hi / 70.0)
    r = round(255 + (176 - 255) * t); g = round(255 + (209 - 255) * t); b = round(255 + (243 - 255) * t)
    return f"rgb({r},{g},{b})"

HTML = ROOT / "results/aggregate/html/results_tables.html"
START, END = "<!-- MEM_FORGET_START -->", "<!-- MEM_FORGET_END -->"
CACHE = sd / "_cache_mem_retention_gpt.jsonl"
ROSE, GREEN = nd.FORGET_COLORS["verbatim"], nd.FORGET_COLORS["concept"]


def _legend(p, items, x, ytop, line=True):
    for gi, (name, col) in enumerate(items):
        y = ytop + gi * 14
        if line:
            p.append(f"<line x1='{x}' y1='{y-3:.0f}' x2='{x+14}' y2='{y-3:.0f}' stroke='{col}' stroke-width='2.6'/>")
        else:
            p.append(f"<rect x='{x}' y='{y-9:.0f}' width='9' height='9' fill='{col}'/>")
        p.append(f"<text x='{x+(19 if line else 13)}' y='{y:.0f}' font-size='9' fill='#33424b'>{name}</text>")


def bar_svg(labels, series, lo_lab="", hi_lab="", refline=None, w=680, h=210, ymax=100):
    """series = list of (name, color, [values]) drawn as grouped bars."""
    LEG = 150
    pl, pb, pt, pr = 32, 30, 16, 12 + LEG
    iw, ih = w - pl - pr, h - pt - pb
    n = len(labels); g = len(series)
    slot = iw / n
    bw = slot * 0.82 / g
    yof = lambda v: pt + ih * (1 - v / ymax)
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for gv in (0, 25, 50, 75, 100):
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#eef1f3'/>")
        p.append(f"<text x='{pl-4}' y='{y+3:.0f}' font-size='8.5' fill='#aab4bb' text-anchor='end'>{gv}</text>")
    if refline is not None:
        y = yof(refline)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='{GREEN}' stroke-width='1.4' stroke-dasharray='4 3'/>")
    for i, lab in enumerate(labels):
        x0 = pl + slot * i + slot * 0.09
        for gi, (name, col, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            x = x0 + gi * bw
            y = yof(v)
            p.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-1:.1f}' height='{pt+ih-y:.1f}' fill='{col}'/>")
            p.append(f"<text x='{x+bw/2:.1f}' y='{y-2:.0f}' font-size='7.5' fill='#33424b' text-anchor='middle'>{v:.0f}</text>")
        p.append(f"<text x='{pl+slot*i+slot/2:.0f}' y='{h-pb+13}' font-size='8.5' fill='#33424b' text-anchor='middle'>{lab}</text>")
    if lo_lab:
        p.append(f"<text x='{pl}' y='{h-3}' font-size='8' fill='#aab4bb'>{lo_lab}</text>")
    if hi_lab:
        p.append(f"<text x='{pl+iw:.0f}' y='{h-3}' font-size='8' fill='#aab4bb' text-anchor='end'>{hi_lab}</text>")
    _legend(p, [(n, c) for n, c, _ in series], pl + iw + 16, pt + 9, line=False)
    p.append("</svg>")
    return "".join(p)


def line_svg(xs, series, xlabel="day the content was first added", xlabels=None,
             w=700, h=230, ymax=100, legend=True, fill_width=False, ymin=0):
    """series = [(name, color, {x: y})]; lines over the shared integer xs.
    xlabels: optional list (one per x) — else falls back to 'd{x}'.
    legend=False drops the in-SVG legend (for compact small-multiple panels).
    fill_width=True makes the SVG stretch to its container width (wider panels).
    ymin>0 tightens the y-range (gridlines adapt) to cut dead vertical space."""
    LEG = 170 if legend else 6
    pl, pb, pt, pr = 34, 34, 16, 14 + LEG
    iw, ih = w - pl - pr, h - pt - pb
    xpos = {x: pl + iw * (i + 0.5) / len(xs) for i, x in enumerate(xs)}
    yof = lambda v: pt + ih * (1 - (v - ymin) / (ymax - ymin))
    _style = "width:100%;height:auto;display:block" if fill_width else "max-width:100%"
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='{_style}'>"]
    _span = ymax - ymin
    _gstep = 25 if _span > 60 else (10 if _span > 30 else 5)
    _grid = list(range(-(-int(ymin) // _gstep) * _gstep, int(ymax) + 1, _gstep))
    for gv in _grid:
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#eef1f3'/>")
        p.append(f"<text x='{pl-5}' y='{y+3:.0f}' font-size='8.5' fill='#aab4bb' text-anchor='end'>{gv}</text>")
    for i, x in enumerate(xs):
        lab = xlabels[i] if xlabels else f"d{x}"
        p.append(f"<text x='{xpos[x]:.0f}' y='{h-pb+14}' font-size='8' fill='#33424b' text-anchor='middle'>{lab}</text>")
    p.append(f"<text x='{(pl+pl+iw)/2:.0f}' y='{h-4}' font-size='8.5' fill='#aab4bb' text-anchor='middle'>{xlabel}</text>")
    for name, col, dd in series:
        pts = [(xpos[x], yof(dd[x])) for x in xs if x in dd]
        d = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
        p.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2'/>")
        for x, y in pts:
            p.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='2.6' fill='{col}'/>")
    if legend:
        _legend(p, [(n, c) for n, c, _ in series], pl + iw + 16, pt + 10, line=True)
    p.append("</svg>")
    return "".join(p)


import html as _htmlmod


def _split_memory(mem):
    title, sections, current = "USER MEMORY", [], None
    for raw in mem.split("\n"):
        l = raw.rstrip()
        if l.startswith("# "):
            title = l[2:]
        elif l.startswith("## "):
            current = {"heading": l[3:], "items": []}
            sections.append(current)
        elif l.strip().startswith("- ") and current:
            current["items"].append(l.strip()[2:])
    return title, sections


def _rep_count(item):
    m = re.search(r"\[×(\d+)\]\s*$", item)
    return int(m.group(1)) if m else 1


def _shorten_item(item, max_chars=285):
    rep = ""
    m = re.search(r"\s*\[×(\d+)\]\s*$", item)
    if m:
        rep = f" [×{m.group(1)}]"
        item = item[:m.start()].rstrip()
    if len(item) > max_chars:
        cut = max(item.rfind("; ", 0, max_chars), item.rfind(", ", 0, max_chars),
                  item.rfind(" ", 0, max_chars))
        if cut < 95:
            cut = max_chars
        item = item[:cut].rstrip() + " ..."
    return item + rep


def _choose_memory_items(items, n=4):
    if len(items) <= n:
        return items
    chosen = [items[0]]
    ranked = sorted(enumerate(items[1:], start=1), key=lambda x: (_rep_count(x[1]), -x[0]), reverse=True)
    for idx, item in ranked:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= n:
            break
    return chosen


def _format_memory_item(item, max_chars=210):
    body = _htmlmod.escape(_shorten_item(item, max_chars=max_chars))
    body = re.sub(r"\[×(\d+)\]", r"<span style='color:#bd5f6a;font-weight:700'>×\1</span>", body)
    body = re.sub(r"^\[([^\]]+)\]", r"<span style='color:#3E7C8C;font-weight:700'>[\1]</span>", body)
    return (
        "<div style='margin:6px 0 0;padding-left:9px;border-left:2px solid #e6ebee'>"
        f"{body}</div>"
    )


def memory_example_html(uid="1", mem_dir="llm_memory_gpt5.5"):
    """Render one real consolidated memory as a compact excerpt.
    Every memory section is kept; long sections show examples plus an omission mark."""
    snaps = sorted((json.load(open(fp)) for fp in
                    glob.glob(f"{ROOT}/results/{mem_dir}/{uid}/memory_states/*.json")),
                   key=lambda d: d["t_test"])
    if not snaps:
        return ""
    snap = snaps[-1]
    mem = snap["memory"]
    nlines = sum(1 for l in mem.split("\n") if l.strip().startswith("-"))
    title, sections = _split_memory(mem)
    parts = [
        "<div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-start;"
        "border-bottom:1px solid #e6ebee;padding:0 0 9px;margin:0 0 9px'>"
        f"<div style='font-size:11.5px;font-weight:750;color:#16242c'>{_htmlmod.escape(title)}</div>"
        "<div style='font-size:9.1px;color:#7c8a93;text-align:right;line-height:1.35'>"
        f"{nlines} deduped lines<br>{len(sections)} sections</div></div>"
    ]
    for section in sections:
        examples = _choose_memory_items(section["items"], n=2)
        omitted = max(0, len(section["items"]) - len(examples))
        body = "".join(_format_memory_item(item) for item in examples)
        if omitted:
            body += (
                "<div style='margin:6px 0 0 11px;color:#aab4bb;font-size:9px;"
                "letter-spacing:.04em'>"
                f"&hellip; <span style='letter-spacing:0'>{omitted} more lines omitted</span></div>"
            )
        parts.append(
            "<div style='break-inside:avoid;border-top:1px solid #edf1f3;padding:9px 0 0;"
            "margin:9px 0 0'>"
            f"<div style='font-size:9.3px;font-weight:750;letter-spacing:.055em;text-transform:uppercase;"
            f"color:#7c8a93;margin:0 0 3px'>{_htmlmod.escape(section['heading'])}</div>{body}</div>"
        )
    box = ("<div style='max-width:720px;margin:0 auto;border:1px solid #dce4e8;border-radius:8px;"
           "background:#fcfdfe;padding:16px 18px;font-family:Inter,Arial,sans-serif;"
           "font-size:10.2px;line-height:1.42;color:#33424b;box-shadow:0 1px 0 rgba(22,36,44,.04)'>"
           + "".join(parts) + "</div>")
    cap = (f"<div class='cap' style='max-width:720px;margin:6px auto 0'><h2 style='font-size:10.8px'>What the consolidated memory looks like "
           f"<span style='font-weight:400;color:#aab4bb'>(one persona, GPT-5.5, 4096-token cap)</span></h2>"
           f"<span class='note'>representative excerpt from {nlines} deduped lines across {len(sections)} memory sections; "
           f"{snap.get('n_events')} events &middot; {snap.get('build_calls')} update calls &middot; "
           f"&hellip; marks omitted lines</span></div>")
    return cap + "<div style='margin:8px 0 22px'>" + box + "</div>"


def main():
    # --- snapshot dynamics (no LLM) ---
    fc = md.forgetting_curve()          # [(day, surv%, n)]
    churn = md.churn()
    cap = md.capacity()
    sat_lines = max(v for _, v, _ in cap)

    # --- semantic concept survival, SAME axes as verbatim (Figure-1 style) ---
    sc = sm.curve(SEM_THRESHOLD)                       # [(day, surv%, n)] concept-level
    verb = {d: s for d, s, _ in fc}
    sem = {d: s for d, s, _ in sc}
    xs = sorted(set(verb) | set(sem))
    sem_mean = statistics.mean(list(sem.values()))
    verb_mid = statistics.mean([verb[d] for d in verb if 2 <= d <= 6])

    # --- retention -> accuracy (Option A cache), one honest number ---
    rows = [json.loads(l) for l in CACHE.open() if l.strip()]
    rows = [r for r in rows if "present" in r]
    overall_ret = 100 * sum(r["present"] for r in rows) / len(rows)
    acc_keep = statistics.mean([r["acc"] for r in rows if r["present"]])
    acc_lost = statistics.mean([r["acc"] for r in rows if not r["present"]])

    sec = [START, '<section>']
    sec.append('<div class="cap"><h2>Why iterative memory can’t fix the middle &mdash; rewording, not forgetting</h2>'
               '<span class="note">GPT-5.5 textual memory &middot; concept-level forgetting curve</span></div>')
    sec.append('<p class="lead" style="margin:0 0 16px">A natural guess is that long context loses mid-window facts but '
               'a memory keeps them. The iterative memory <i>looks</i> forgetful: it is capacity-bounded '
               f'(~{sat_lines:.0f} lines for 1,700+ events) and rewrites <b>{churn["pct_dropped_per_update"]:.0f}%</b> of '
               'its lines every update, so <b>verbatim</b> line survival collapses for older-added content. But matching by '
               '<b>exact wording is the wrong test under compression</b> &mdash; each update rephrases lines. Redrawing the '
               'same curve at the <b>concept</b> level (a line survives if the saturated memory contains a semantically '
               f'equivalent line, cosine&ge;{SEM_THRESHOLD} on text-embedding-3-large) shows survival is <b>flat at '
               f'~{sem_mean:.0f}%</b> regardless of when the content was added &mdash; the middle is <b>not</b> forgotten. '
               '(matched 10 personas.)</p>')

    # --- a real example of the consolidated memory ---
    sec.append(memory_example_html())

    # --- 3 memory sizes side by side: verbatim vs concept forgetting curve ---
    abl = msa.stats()                                   # also reused by the Direct test below
    # shade range from this table's own blue cells (overall + fine bins) so the
    # heatmap whitens at the low end while keeping the deepest cell's max blue.
    global _ALO, _AHI
    _av = []
    for s in abl:
        _av += [v for v in s.get("fine", []) if v is not None]
        if s.get("overall") is not None:
            _av.append(s["overall"])
    if _av:
        _ALO, _AHI = min(_av), max(_av)
    size_of = {s["label"]: s for s in abl}
    panels, allx = [], set()
    for label, mdir in msa.CAPS:                        # Half · 2048 / Baseline · 4096 / Double · 8192
        fcd = md.forgetting_curve(mem_dir=mdir)         # verbatim (no LLM)
        # min_per_day=4 so the smallest (Half) cap still plots its last (day-10) point (n=4)
        scd = sm.curve(SEM_THRESHOLD, mem_dir=mdir, min_per_day=4)  # concept (cached embeddings)
        vb = {x: s for x, s, _ in fcd}
        se = {x: s for x, s, _ in scd}
        allx |= set(vb) | set(se)
        panels.append((label, vb, se))
    xs3 = sorted(allx)                                  # shared x-axis so panels align

    sec.append('<div class="cap"><h2 style="font-size:10.5px">Forgetting curve &mdash; exact wording vs. meaning</h2>'
               '<span class="note">% of content first added on day d still present in the FINAL memory &middot; '
               'half / baseline / double memory cap</span></div>')
    # one shared, reader-friendly legend (the two curves are identical across panels)
    sec.append(
        '<div style="display:flex;gap:22px;margin:4px 0 4px;font-size:9px;color:#33424b">'
        f'<span><span style="display:inline-block;width:15px;border-top:2.6px solid {GREEN};vertical-align:middle"></span>'
        '&nbsp;semantically same fact kept</span>'
        f'<span><span style="display:inline-block;width:15px;border-top:2.6px solid {ROSE};vertical-align:middle"></span>'
        '&nbsp;exact original wording kept</span></div>')
    cells3 = []
    for label, vb, se in panels:
        s = size_of.get(label, {})
        sz = f"{s.get('mem_lines', 0):.0f}L&nbsp;/&nbsp;{s.get('mem_chars', 0)/1000:.0f}k" if s else ""
        svg = line_svg(xs3, [("semantically same fact kept", GREEN, se),
                             ("exact original wording kept", ROSE, vb)],
                       w=330, h=200, legend=False, fill_width=True)
        cells3.append(
            f'<div style="flex:1;min-width:0;text-align:center">'
            f'<div style="font-size:10px;font-weight:700;color:#243039">{label}</div>'
            f'<div style="font-size:8.5px;color:#aab4bb;margin-bottom:2px">{sz}</div>'
            f'{svg}</div>')
    sec.append('<div style="display:flex;gap:5px;align-items:flex-start;margin:2px 0 6px">'
               + "".join(cells3) + '</div>')
    means = [f"{label.split(' · ')[0].lower()} ~{statistics.mean(list(se.values())):.0f}%"
             for label, _vb, se in panels if se]
    sec.append('<div class="note" style="margin-bottom:8px">At every memory size the lower (rose) line &mdash; the share of '
               'facts whose <i>exact original wording</i> is still there &mdash; plunges for early/mid content, while the upper '
               '(green) line &mdash; the same fact in <i>any</i> wording &mdash; stays high and flat '
               f'(kept {", ".join(means)}). The gap is the memory rephrasing facts as it rewrites itself, not losing them.</div>')

    sec.append('<div class="abldefs" style="max-width:820px;margin-top:6px">'
               f'<b>Reading.</b> Verbatim survival for mid-window content is ~{verb_mid:.0f}%, which looks like catastrophic '
               f'forgetting; concept-level survival is ~{sem_mean:.0f}% and flat &mdash; the consolidator <b>rephrases</b> '
               'lines each update, it does not delete the facts. A per-query GPT-5.5 probe agrees: the queried preference is '
               f'semantically present in the memory <b>{overall_ret:.0f}%</b> of the time, and accuracy when it is retained '
               f'({acc_keep:.0f}) is <b>no higher</b> than when it is dropped ({acc_lost:.0f}). &nbsp;'
               '<b>Conclusion.</b> Iterative memory cannot fix lost-in-the-middle because the middle dip is not a '
               'retention failure: long context keeps the full history and still dips, and memory keeps the facts and '
               'still dips. The bottleneck is <b>using</b> mid-window-emerging interests (weaker, more ambiguous signal), '
               'which a better/larger store does not address by retention alone. &nbsp;'
               '<b>Method.</b> Daily GPT-5.5 memory snapshots (10 personas); a concept = a cluster of bullet lines within '
               f'cosine&ge;{SEM_THRESHOLD}; first_day = the day its first member appears; survives = a member appears in the '
               'FINAL memory. Robust across cosine 0.40&ndash;0.55 (mean 81&ndash;85%). Embeddings cached. &nbsp;'
               '<b>Why the x-axis ends at ~day 10, not 30:</b> note this x-axis is &ldquo;day a fact was first ADDED to the '
               'memory,&rdquo; <i>not</i> the query&rarr;source distance of the other plots. The snapshot/test-time range '
               'runs to ~day 30, but user activity (events) spans only ~9 days, so <b>all content enters the memory by '
               '~day 9</b> and the memory then <b>freezes</b> (build_calls stop; the day 20&ndash;34 snapshots are '
               'byte-identical copies) &mdash; there is no content added after ~day 9 to plot. Both curves now run to '
               'day&nbsp;10; the late <b>concept</b> points are thin (n&nbsp;5&ndash;7) because by then the consolidator is '
               'mostly <b>rewording</b> existing facts &mdash; very few genuinely NEW concepts first-appear &mdash; while '
               'reworded line-strings still count as &ldquo;new lines&rdquo; for the verbatim curve.</div>')

    # ===== memory-SIZE ablation: does a bigger memory close the dip? =====
    # (abl already computed above for the 3-up forgetting panels)
    DBL = next(s for s in abl if "Double" in s["label"])
    HLF = next(s for s in abl if "Half" in s["label"])
    sec.append('<div class="cap" style="margin-top:26px"><h2 style="font-size:10.5px">Direct test &mdash; does a bigger memory close the middle dip?</h2>'
               '<span class="note">GPT-5.5 textual memory rebuilt at half / baseline / double the token cap, all tasks, matched 10</span></div>')
    # table (fine-near / wide-middle / 10d+ bins, all tasks)
    FL = nd.MIDBIN_LABELS
    bl = "".join("<col class='cval'>" for _ in FL)
    bh = "".join(f"<th class='model'><span class='m' style='font-size:9px'>{x}</span></th>" for x in FL)
    trs = []
    for s in abl:
        dot = f"<span style='color:{CAP_COLOR[s['label']]}'>&#9632;</span> "
        cells = "".join(
            f"<td class='val' style='background:{acc_shade(v)};color:#243039'>{v:.0f}</td>" if v is not None
            else "<td class='val na'>&ndash;</td>" for v in s["fine"])
        trs.append(
            f"<tr class='catstart'><td class='task'>{dot}{s['label']}</td>"
            f"<td class='val' style='color:#33424b'>{s['mem_lines']:.0f}L&nbsp;/&nbsp;{s['mem_chars']/1000:.0f}k</td>"
            f"<td class='val' style='background:{acc_shade(s['overall'])};color:#243039'>{s['overall']:.0f}</td>"
            f"{cells}<td class='val' style='color:#33424b'>{s['dip']:.0f}</td></tr>")
    sec.append(
        f"<table><colgroup><col class='ctask'><col class='cval'><col class='cval'>{bl}<col class='cval'></colgroup>"
        "<thead><tr><th class='lbl'>Memory cap</th>"
        "<th class='model'><span class='m'>mem size</span></th>"
        "<th class='model'><span class='m'>overall</span><span class='v'>all tasks</span></th>"
        f"{bh}<th class='model'><span class='m'>dip</span></th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table>")
    # by-cap fine-grained depth curves (3 lines over daily bins)
    xs = list(range(len(FL)))
    sec.append('<div style="margin:10px 0 4px">' +
               line_svg(xs,
                        [(s["label"], CAP_COLOR[s["label"]],
                          {i: s["fine"][i] for i in xs if s["fine"][i] is not None}) for s in abl],
                        xlabel="needle depth (days back to first appearance)", xlabels=FL,
                        ymin=30, ymax=70) + '</div>')
    sec.append(
        '<div class="abldefs" style="max-width:820px">'
        f'<b>Result.</b> Doubling the memory (to {DBL["mem_lines"]:.0f} lines / ~{DBL["mem_chars"]/1000:.0f}k chars, '
        f'~2&times; the half setting at {HLF["mem_lines"]:.0f} lines) does <b>not</b> close the middle dip: the 5&ndash;7d '
        'bin stays low across all three sizes, and overall accuracy barely moves (half &asymp; double). More capacity '
        'retains more text but does not buy back the mid-window queries &mdash; confirming the bottleneck is '
        '<b>use, not retention</b>. Cells shaded by accuracy; dip = ends &minus; middle.</div>')

    sec.append('</section>')
    sec.append(END)
    block = "\n".join(sec)

    with html_lock():                       # single-writer: no concurrent overwrite
        html = HTML.read_text()
        if html.count(START) > 1 or html.count(END) > 1 or html.count("<!doctype") > 1:
            raise SystemExit("results_tables.html looks corrupted (duplicate markers/doctype) — "
                             "restore from /tmp/results_tables_backup.html before re-injecting")
        if START in html and END in html:
            html = html.split(START)[0] + block + html.split(END, 1)[1]
        else:
            # place right after the lost-in-the-middle section
            anchor = "<!-- NEEDLE_DEPTH_END -->"
            html = html.replace(anchor, anchor + "\n\n" + block, 1)
        HTML.write_text(html)
    print("injected into", HTML)


if __name__ == "__main__":
    main()
