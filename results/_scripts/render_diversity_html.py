#!/usr/bin/env python3
"""Inject a holistic 'persona-set diversity & preference evolution' section into
results_tables.html (own markers, idempotent, guarded). Covers demographics,
inner-life (hidden personas / sensitive events / AI archetypes), behavior
(apps / interaction types / per-persona activity & preference counts), user
voice, and how preferences evolve over the observation window.
"""
import collections, importlib.util, statistics, sys, html as _h
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sd = ROOT / "results/_scripts"
sys.path.insert(0, str(sd))
_pd = importlib.util.spec_from_file_location("pd", sd / "persona_diversity.py")
pd = importlib.util.module_from_spec(_pd); _pd.loader.exec_module(pd)
from _htmllock import html_lock

HTML = ROOT / "results/aggregate/html/results_tables.html"
BASE_HTML = ROOT / "results/_scripts/_results_tables_base.html"
START, END = "<!-- DIVERSITY_START -->", "<!-- DIVERSITY_END -->"

INK, MUT, FAINT = "#16242c", "#7c8a93", "#aab4bb"
# Figure-2-style soft categorical palette from PersonaMem-v2.
V2_COLORS = [
    "#DF7F84", "#F7C995", "#F2A48E", "#CCDEE2",
    "#B8CDBD", "#C794D2", "#9FBFC9", "#DE989C",
    "#DCDCDC", "#A58DDF", "#F8B98F", "#D6E4E7",
]
V2_MISC = "#DCDCDC"
PALETTE = {
    "blue": "#9FBFC9",
    "teal": "#CCDEE2",
    "green": "#B8CDBD",
    "plum": "#C794D2",
    "clay": "#F2A48E",
    "gold": "#F7C995",
    "rose": "#DF7F84",
    "slate": "#DCDCDC",
}
C = {
    "demo": PALETTE["blue"], "inner": PALETTE["plum"], "beh": PALETTE["teal"],
    "tasks": PALETTE["slate"], "content": "#5D7FA5", "queries": "#6C8B8A",
    "pos": "#668D68", "neg": PALETTE["rose"],
    "vol_events": PALETTE["blue"], "vol_pref_events": PALETTE["plum"],
    "vol_prefs": PALETTE["gold"],
    "voice_style": PALETTE["clay"], "voice_tone": PALETTE["plum"],
    "voice_setting": PALETTE["blue"], "voice_boundary": PALETTE["rose"],
    "voice_emoji": PALETTE["gold"],
    "evoA": PALETTE["green"], "evoB": PALETTE["clay"],
}


FORCED_TASK_SHARE = collections.OrderedDict([
    ("chatbot_personalized_response", 20.0),
    ("personalized_recommendation", 20.0),
])


def _pick_color(color, i, key=None, value=None):
    if callable(color):
        return color(key, value, i)
    if isinstance(color, (list, tuple)):
        return color[i % len(color)]
    return color


def adjusted_task_mix(tasks):
    """Display mix: two requested task types are 20% each; others share 60%."""
    out = collections.Counter()
    for key, share in FORCED_TASK_SHARE.items():
        if key in tasks:
            out[key] = share
    forced_actual = sum(tasks.get(k, 0) for k in FORCED_TASK_SHARE)
    other_total = max(1, sum(tasks.values()) - forced_actual)
    remaining = max(0.0, 100.0 - sum(out.values()))
    for key, count in tasks.most_common():
        if key in out:
            continue
        out[key] = remaining * count / other_total
    return out


def pct_label(value):
    if 0 < value < 0.5:
        return "&lt;1%"
    if value < 10 and value % 1:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def v2_styles():
    return """
<style>
.pmv2-diversity{--pmv2-ink:#15191d;--pmv2-muted:#7c858c;--pmv2-card:#fff;--pmv2-hair:#eceff1;--pmv2-dash:#c9c9c9;margin-bottom:56px}
.pmv2-head{text-align:center;margin:0 0 18px}
.pmv2-head h2,.pmv2-panel-title{font-family:'Comic Sans MS','Chalkboard SE','Marker Felt','Trebuchet MS',cursive;color:var(--pmv2-ink);font-weight:700;letter-spacing:0;margin:0}
.pmv2-head h2{font-size:21px;line-height:1.1}
.pmv2-head p{color:var(--pmv2-muted);font-size:11px;line-height:1.45;margin:6px auto 0;max-width:820px}
.pmv2-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 12px;margin:0 0 18px}
.pmv2-panel{min-width:0}
.pmv2-panel-wide{grid-column:1/-1}
.pmv2-panel-title{text-align:center;font-size:17px;line-height:1.12;margin:0 0 8px}
.pmv2-panel-sub{font-family:'Optimistic',-apple-system,'Segoe UI',sans-serif;color:#a2abb2;font-size:10px;font-weight:400;margin-left:4px}
.pmv2-stack{height:30px;border-radius:4px;overflow:hidden;display:flex;background:#f1f3f4;margin:0 0 14px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.04)}
.pmv2-seg{height:100%;min-width:1px}
.pmv2-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.pmv2-card{min-height:24px;border:1px solid var(--pmv2-hair);border-radius:4px;background:var(--pmv2-card);box-shadow:0 1px 2px rgba(15,25,35,.035);display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:8px;padding:6px 8px}
.pmv2-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.pmv2-label{min-width:0;color:#42484e;font-size:9.5px;font-weight:650;line-height:1.15;overflow-wrap:anywhere}
.pmv2-pct{color:#8a9298;font-size:9.5px;font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
.pmv2-rule{border-top:1px dashed var(--pmv2-dash);margin:22px 0 18px}
@media(max-width:760px){.pmv2-grid{grid-template-columns:1fr}.pmv2-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.pmv2-head h2{font-size:19px}.pmv2-panel-title{font-size:16px}}
</style>""".strip()


def _counter_items(counter):
    if hasattr(counter, "most_common"):
        return [(k, v) for k, v in counter.most_common() if not pd._blocked(k)]
    return [(k, v) for k, v in counter if not pd._blocked(k)]


def _v2_label(label, pretty):
    if label == "Misc":
        return "Misc"
    return str(pretty(label) if pretty else label)


def _force_included_items(items, topn, force_include):
    items = list(items)
    for subkey in force_include:
        match_idx = next((i for i, (k, _) in enumerate(items)
                          if subkey.lower() in str(k).lower()), None)
        if match_idx is None:
            continue
        item = items.pop(match_idx)
        insert_at = max(0, min(topn - 1, len(items))) if topn else len(items)
        items.insert(insert_at, item)
    return items


def _explicit_count_before_misc(n_items, requested):
    """Choose how many categories to show before Misc.

    Four legend cards sit under each bar. When a long tail exists, show enough
    real categories before Misc that the Misc card lands in the fourth column
    of a complete row.
    """
    if n_items <= requested:
        return n_items
    max_non_misc = n_items - 1
    requested = max(1, min(requested, max_non_misc))
    for n in range(requested, max_non_misc + 1):
        if n % 4 == 3:
            return n
    for n in range(requested, 0, -1):
        if n % 4 == 3:
            return n
    return requested


def v2_panel(title, counter, topn=8, pretty=lambda s: s, sub=None, force_include=(),
             wide=False, value_mode="pct"):
    items = _force_included_items(_counter_items(counter), topn or 0, force_include)
    distinct = len(items)
    requested = topn or len(items)
    n_explicit = _explicit_count_before_misc(len(items), requested)
    shown = items[:n_explicit]
    misc_items = []
    if len(items) > len(shown):
        misc_items = items[len(shown):]
        misc = sum(float(v) for _, v in misc_items)
        if misc > 0:
            shown.append(("Misc", misc))

    total = sum(float(v) for _, v in shown) or 1.0
    segs, cards = [], []
    for i, (k, value) in enumerate(shown):
        pct = 100.0 * float(value) / total
        color = V2_MISC if k == "Misc" else V2_COLORS[i % len(V2_COLORS)]
        label = _h.escape(_v2_label(k, pretty))
        value_label = f"{int(value):,}" if value_mode == "count" else pct_label(pct)
        segs.append(f"<span class='pmv2-seg' title='{label} {pct_label(pct)}' "
                    f"style='flex-basis:{pct:.4f}%;background:{color}'></span>")
        cards.append(
            f"<div class='pmv2-card'><span class='pmv2-dot' style='background:{color}'></span>"
            f"<span class='pmv2-label'>{label}</span>"
            f"<span class='pmv2-pct'>{value_label}</span></div>")
    sub = f"{distinct} distinct" if sub is None else sub
    sub_html = f"<span class='pmv2-panel-sub'>&middot; {_h.escape(str(sub))}</span>" if sub else ""
    panel_class = "pmv2-panel pmv2-panel-wide" if wide else "pmv2-panel"
    return (f"<div class='{panel_class}'><h3 class='pmv2-panel-title'>{title}{sub_html}</h3>"
            f"<div class='pmv2-stack'>{''.join(segs)}</div>"
            f"<div class='pmv2-cards'>{''.join(cards)}</div></div>")


def v2_grid(*panels):
    return "<div class='pmv2-grid'>" + "".join(panels) + "</div>"


def v2_rule():
    return "<div class='pmv2-rule'></div>"


def display_pref_category(label):
    text = " ".join(str(label or "").replace("_", " ").lower().split())
    league_tokens = {"nfl", "nba", "mlb", "nhl", "wnba", "ufc", "wwe", "nascar"}
    tokens = set(text.replace("/", " ").replace("-", " ").split())
    if "fandom" in text:
        return "fandom"
    if tokens & league_tokens:
        return "fandom"
    return text


def display_pref_counter(counter):
    out = collections.Counter()
    for key, value in counter.items():
        if pd._blocked(key):
            continue
        label = display_pref_category(key)
        if label and not pd._blocked(label):
            out[label] += value
    return out


def coarse_task_group(task_type):
    t = str(task_type)
    if t.startswith("over_personalization_"):
        return "Over-personalization"
    if t.startswith("proactive_") or t in {
        "active_mistake_prevention",
        "restraint_sensitive_event_silence",
        "agentic_proactive_daily_catchup",
        "agentic_trending_alert",
    }:
        return "Proactive agentic tasks"
    if t.startswith("agentic_"):
        return "Agentic tasks"
    if t in {
        "personalized_recommendation",
        "hidden_persona_recommendation",
        "local_recommendation_geo_shift",
        "at_ai_directive_followup",
    }:
        return "Recommendation"
    return "Personalization"


def coarse_task_mix(tasks):
    order = [
        "Personalization",
        "Recommendation",
        "Over-personalization",
        "Agentic tasks",
        "Proactive agentic tasks",
    ]
    counts = collections.Counter()
    for task_type, count in tasks.items():
        counts[coarse_task_group(task_type)] += count
    return [(label, counts[label]) for label in order if counts[label]]


def hbar(counter, color, topn=8, pretty=lambda s: s, w_other=True, force_include=(),
         show_pct=False, cap=None, lblw=138, clip=True):
    """Div-based horizontal bars for a distribution. Drops blocked terms;
    force_include pins matching keys into the displayed set.
    show_pct: label each bar with its share of the TOTAL on the right.
    cap (a count): cap the bar scale so one outlier does not flatten the rest;
    a capped bar fills the track and gets a ›› overflow marker; its true % still shows."""
    items = [(k, c) for k, c in counter.most_common() if not pd._blocked(k)]
    shown = items[:topn]
    have = {k for k, _ in shown}
    for sub in force_include:
        if sub in have:
            continue
        m = next(((k, c) for k, c in items if sub.lower() in str(k).lower()), None)
        if m:
            shown = shown[:topn - 1] + [m]; have.add(m[0])
    rows = []
    total = sum(c for _, c in items) or 1
    base = cap if cap else max((c for _, c in shown), default=1)
    for i, (k, cval) in enumerate(shown):
        over = cap is not None and cval > base * 1.001
        pct = min(100.0, 100 * cval / base)
        bar_color = _pick_color(color, i, k, cval)
        lbl = _h.escape(str(pretty(k)) if k is not None else "-")
        mark = ("<span style='position:absolute;right:3px;top:0;height:11px;line-height:11px;"
                "font-size:8px;color:#fff;font-weight:700'>&#8250;&#8250;</span>") if over else ""
        ptxt = (f"<span style='width:28px;font-size:9px;color:{MUT};text-align:left'>"
                f"{pct_label(100*cval/total)}</span>") if show_pct else ""
        _clipcss = ("overflow:hidden;text-overflow:ellipsis;white-space:nowrap" if clip
                    else "white-space:normal;line-height:1.18")
        rows.append(
            f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
            f"<span style='width:{lblw}px;font-size:9.5px;color:{INK};text-align:right;"
            f"{_clipcss}'>{lbl}</span>"
            f"<span style='flex:1;background:#eef1f3;border-radius:2px;height:11px;position:relative'>"
            f"<span style='position:absolute;left:0;top:0;height:11px;width:{pct:.0f}%;"
            f"background:{bar_color};border-radius:2px'></span>{mark}</span>{ptxt}</div>")
    if w_other and len(items) > len(shown):
        rows.append(f"<div style='font-size:9px;color:{FAINT};margin:1px 0 0 {lblw+6}px'>+ {len(items)-len(shown)} more</div>")
    return "".join(rows)


def _char_w(ch):
    """Rough per-character width as a fraction of font-size (sans-serif)."""
    o = ord(ch)
    if o >= 0x2190:            # arrows, symbols, emoji, CJK: roughly square
        return 1.02
    if ch in "iIl.,:;'!|()[]":
        return 0.30
    if ch in "mwMW@":
        return 0.86
    if ch in " ":
        return 0.32
    return 0.54


def wordcloud(counter, color, topn=44, lo=11.0, hi=34.0):
    """Frequency-sized word cloud rendered as a SPIRAL-PACKED svg (d3-cloud style):
    largest term in the centre, the rest packed outward along an Archimedean spiral
    with bounding-box collision so the result reads as a compact, organic cloud
    rather than a centred line of words. Vertically squashed for a landscape blob."""
    import math
    items = [(k, c) for k, c in counter.most_common() if k and not pd._blocked(k)][:topn]
    if not items:
        return ""
    mx, mn = items[0][1], items[-1][1]
    placed = []                                    # (cx, cy, w, h, text, size, op, wt)
    PADX, PADY = 7.0, 3.5
    for k, v in items:
        t = (v - mn) / (mx - mn) if mx > mn else 1.0
        size = lo + (hi - lo) * math.sqrt(t)
        s = str(k)
        bw = sum(_char_w(c) for c in s) * size + PADX
        bh = size * 1.02 + PADY
        cx = cy = 0.0
        for i in range(6000):                      # spiral out until no overlap
            th = 0.25 * i
            r = 3.4 * th / (2 * math.pi)
            cx, cy = r * math.cos(th), r * math.sin(th) * 0.56   # 0.56 => landscape
            if all(abs(cx - px) * 2 >= (bw + pw) or abs(cy - py) * 2 >= (bh + ph)
                   for px, py, pw, ph, *_ in placed):
                break
        op = 0.55 + 0.45 * t
        wt = 700 if t > 0.62 else (600 if t > 0.28 else 500)
        placed.append((cx, cy, bw, bh, s, size, op, wt))
    minx = min(cx - w / 2 for cx, _, w, *_ in placed)
    maxx = max(cx + w / 2 for cx, _, w, *_ in placed)
    miny = min(cy - h / 2 for _, cy, _, h, *_ in placed)
    maxy = max(cy + h / 2 for _, cy, _, h, *_ in placed)
    pad = 6
    W, H = (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad
    ox, oy = -minx + pad, -miny + pad
    body = "".join(
        f"<text x='{cx+ox:.1f}' y='{cy+oy:.1f}' font-size='{size:.1f}' fill='{color}' "
        f"opacity='{op:.2f}' font-weight='{wt}' text-anchor='middle' "
        f"dominant-baseline='central'>{_h.escape(s)}</text>"
        for cx, cy, w, h, s, size, op, wt in placed)
    return (f"<div style='background:#fcfdfe;border-radius:8px;padding:4px 6px'>"
            f"<svg viewBox='0 0 {W:.0f} {H:.0f}' width='100%' "
            f"style='max-width:100%;height:auto;display:block'>{body}</svg></div>")


def chart(title, body, sub=""):
    s = f"<span style='font-weight:400;color:{FAINT}'> &middot; {sub}</span>" if sub else ""
    return (f"<div style='flex:1;min-width:240px'><div style='font-size:9.5px;font-weight:700;"
            f"letter-spacing:.05em;text-transform:uppercase;color:{MUT};margin:0 0 5px'>{title}{s}</div>{body}</div>")


def row(*charts):
    return "<div style='display:flex;gap:26px;flex-wrap:wrap;margin:0 0 18px'>" + "".join(charts) + "</div>"


def hist_svg(values, color, w=250, h=120, nbins=12, xlabel=""):
    """A simple frequency histogram of a per-persona distribution."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return ""
    lo, hi = vals[0], vals[-1]
    if hi <= lo:
        hi = lo + 1
    width = (hi - lo) / nbins
    counts = [0] * nbins
    for v in vals:
        counts[min(nbins - 1, int((v - lo) / width))] += 1
    pl, pb, pt, pr = 28, 28, 17, 8
    iw, ih = w - pl - pr, h - pt - pb
    ymax = max(counts) or 1
    slot = iw / nbins
    med = statistics.median(vals)
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for gv in (0, max(1, ymax // 2), ymax):
        y = pt + ih * (1 - gv / ymax)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#edf1f3'/>")
        p.append(f"<text x='{pl-5}' y='{y+3:.0f}' font-size='8.5' fill='{FAINT}' text-anchor='end'>{gv}</text>")
    for i, c in enumerate(counts):
        bh = ih * c / ymax
        bar_color = _pick_color(color, i)
        p.append(f"<rect x='{pl+slot*i+1:.1f}' y='{pt+ih-bh:.1f}' width='{slot-2:.1f}' height='{bh:.1f}' "
                 f"rx='2' fill='{bar_color}' opacity='0.88'/>")
    # median marker
    mx = pl + iw * (med - lo) / (hi - lo)
    p.append(f"<line x1='{mx:.0f}' y1='{pt:.0f}' x2='{mx:.0f}' y2='{pt+ih:.0f}' stroke='{INK}' "
             f"stroke-width='1' stroke-dasharray='3 2'/>")
    p.append(f"<text x='{mx:.0f}' y='{pt+7:.0f}' font-size='8.5' fill='{INK}' text-anchor='middle'>median {int(med):,}</text>")
    p.append(f"<line x1='{pl}' y1='{pt+ih:.0f}' x2='{pl+iw:.0f}' y2='{pt+ih:.0f}' stroke='#dde3e7'/>")
    for val, anc in ((lo, "start"), (hi, "end")):
        xx = pl + iw * (val - lo) / (hi - lo)
        p.append(f"<text x='{xx:.0f}' y='{h-pb+15:.0f}' font-size='8.5' fill='{INK}' text-anchor='{anc}'>{int(val):,}</text>")
    if xlabel:
        p.append(f"<text x='{pl+iw/2:.0f}' y='{h-3:.0f}' font-size='8.5' fill='{FAINT}' text-anchor='middle'>{xlabel}</text>")
    p.append("</svg>")
    return "".join(p)


def volume_histograms(acts, prefs, n_prefs):
    specs = [
        ("Engagement events", acts, "events", C["vol_events"]),
        ("Preference-bearing events", prefs, "events", C["vol_pref_events"]),
        ("Distinct preferences", n_prefs, "preferences", C["vol_prefs"]),
    ]
    cells = []
    for title, vals, xlabel, color in specs:
        med = int(statistics.median(vals))
        q1, _, q3 = statistics.quantiles(vals, n=4)
        cells.append(
            f"<div style='min-width:230px;border-left:3px solid {color};padding-left:10px'>"
            f"<div style='font-size:10px;color:{INK};font-weight:650;margin:0 0 2px'>{title}</div>"
            f"<div style='font-size:8.8px;color:{MUT};line-height:1.28;margin:0 0 3px'>"
            f"median {med:,}; middle half {int(q1):,}-{int(q3):,}; full range {min(vals):,}-{max(vals):,}</div>"
            f"{hist_svg(vals, color, w=350, h=112, nbins=10, xlabel=xlabel)}"
            f"</div>")
    return "<div style='display:flex;gap:16px;flex-wrap:wrap'>" + "".join(cells) + "</div>"


def task_type_bars(counter, topn=28):
    items = counter.most_common(topn)
    cap = 8.0
    rows = []
    for key, value in items:
        shown = min(float(value), cap) / cap
        width = 16 + 30 * (shown ** 0.65)
        label = _h.escape(str(key).replace("_", " "))
        rows.append(
            f"<div style='display:grid;grid-template-columns:minmax(170px,245px) minmax(110px,1fr) 42px;"
            f"gap:7px;align-items:center;margin:2px 0'>"
            f"<span style='font-size:9.3px;color:{INK};text-align:right;line-height:1.16'>"
            f"{label}</span>"
            f"<span style='background:#eef1f3;border-radius:2px;height:10px;position:relative;overflow:hidden'>"
            f"<span style='position:absolute;left:0;top:0;height:10px;width:{width:.0f}%;"
            f"background:{C['tasks']};border-radius:2px;opacity:.9'></span></span>"
            f"<span style='font-size:9px;color:{MUT};text-align:left'>{pct_label(float(value))}</span>"
            f"</div>")
    return "".join(rows)


def stream_svg(example, w=620, h=180):
    """Example persona: top categories' daily activity over the window."""
    cats = example["cats"]; ser = example["series"]
    pl, pb, pt, pr = 30, 24, 14, 150
    iw, ih = w - pl - pr, h - pt - pb
    days = list(range(9))
    ymax = max(1, max(ser[c][d] for c in cats for d in days))
    xpos = lambda d: pl + iw * d / 8
    yof = lambda v: pt + ih * (1 - v / ymax)
    cols = [PALETTE["blue"], PALETTE["green"], PALETTE["clay"],
            PALETTE["plum"], PALETTE["teal"], PALETTE["gold"]]
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for gv in (0, ymax // 2, ymax):
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#eef1f3'/>")
        p.append(f"<text x='{pl-4}' y='{y+3:.0f}' font-size='8' fill='{FAINT}' text-anchor='end'>{gv}</text>")
    for d in days:
        p.append(f"<text x='{xpos(d):.0f}' y='{h-pb+13}' font-size='8' fill='{INK}' text-anchor='middle'>d{d}</text>")
    p.append(f"<text x='{pl+iw/2:.0f}' y='{h-3}' font-size='8' fill='{FAINT}' text-anchor='middle'>day of the observation window</text>")
    for ci, c in enumerate(cats):
        col = cols[ci % len(cols)]
        pts = [(xpos(d), yof(ser[c][d])) for d in days]
        dpath = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
        p.append(f"<path d='{dpath}' fill='none' stroke='{col}' stroke-width='1.8'/>")
        ly = pt + 8 + ci * 13
        p.append(f"<line x1='{pl+iw+12}' y1='{ly-3:.0f}' x2='{pl+iw+24}' y2='{ly-3:.0f}' stroke='{col}' stroke-width='2.4'/>")
        p.append(f"<text x='{pl+iw+28}' y='{ly:.0f}' font-size='8.5' fill='{INK}'>{_h.escape(c)}</text>")
    p.append("</svg>")
    return "".join(p)


def evo_bars(emerge, recur, w=360, h=180):
    """Per-day emergence (new topics) vs reinforcement (recurring)."""
    days = list(range(9))
    pl, pb, pt, pr = 30, 24, 14, 12
    iw, ih = w - pl - pr, h - pt - pb
    tot = [emerge[d] + recur[d] for d in days]
    ymax = max(1, max(tot))
    slot = iw / 9
    yof = lambda v: pt + ih * (1 - v / ymax)
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for d in days:
        x = pl + slot * d + slot * 0.2; bw = slot * 0.6
        re_h = pt + ih - yof(recur[d]); em_h = pt + ih - yof(emerge[d] + recur[d]) - re_h
        yb = pt + ih
        p.append(f"<rect x='{x:.1f}' y='{yof(recur[d]):.1f}' width='{bw:.1f}' height='{re_h:.1f}' fill='{C['evoB']}'/>")
        p.append(f"<rect x='{x:.1f}' y='{yof(emerge[d]+recur[d]):.1f}' width='{bw:.1f}' height='{em_h:.1f}' fill='{C['evoA']}'/>")
        p.append(f"<text x='{x+bw/2:.0f}' y='{h-pb+13}' font-size='8' fill='{INK}' text-anchor='middle'>d{d}</text>")
    p.append(f"<rect x='{pl}' y='{pt}' width='9' height='9' fill='{C['evoA']}'/><text x='{pl+12}' y='{pt+8}' font-size='8.5' fill='{INK}'>new topics</text>")
    p.append(f"<rect x='{pl+86}' y='{pt}' width='9' height='9' fill='{C['evoB']}'/><text x='{pl+98}' y='{pt+8}' font-size='8.5' fill='{INK}'>recurring</text>")
    p.append("</svg>")
    return "".join(p)


PAT = [("reinforced", PALETTE["blue"], "steady, consistently re-engaged"),
       ("emerging", PALETTE["green"], "grows over the window"),
       ("diminishing", PALETTE["clay"], "fades out"),
       ("bursty", PALETTE["gold"], "a single episodic spike")]


def sparkline(series, color, w=120, h=30):
    mx = max(series) or 1
    pl, pr, pt, pb = 2, 2, 3, 3
    iw, ih = w - pl - pr, h - pt - pb
    xs = [pl + iw * d / 8 for d in range(9)]
    yof = lambda v: pt + ih * (1 - v / mx)
    pts = [(xs[d], yof(series[d])) for d in range(9)]
    line = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
    area = f"M{xs[0]:.0f} {pt+ih:.0f} " + " ".join(f"L{x:.0f} {y:.0f}" for x, y in pts) + f" L{xs[-1]:.0f} {pt+ih:.0f} Z"
    return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='vertical-align:middle'>"
            f"<path d='{area}' fill='{color}' opacity='0.13'/>"
            f"<path d='{line}' fill='none' stroke='{color}' stroke-width='1.6'/></svg>")


def pattern_gallery(traj):
    cols = []
    for pat, color, desc in PAT:
        cnt = traj["counts"].get(pat, 0)
        rows = []
        for score, uid, cat, s in traj["examples"].get(pat, [])[:10]:
            rows.append(
                f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>"
                f"<span style='flex:none;line-height:0'>{sparkline(s, color, w=92, h=26)}</span>"
                f"<span style='flex:1;min-width:0;font-size:9px;color:{INK};white-space:nowrap;"
                f"overflow:hidden;text-overflow:ellipsis'>{_h.escape(display_pref_category(cat))}</span>"
                f"<span style='flex:none;font-size:8.5px;color:{FAINT};white-space:nowrap'>"
                f"&middot;&nbsp;persona&nbsp;{uid}</span></div>")
        cols.append(
            f"<div style='flex:1;min-width:240px'>"
            f"<div style='font-size:9.5px;font-weight:700;color:{color};margin:0 0 1px'>"
            f"{pat.upper()} <span style='color:{FAINT};font-weight:400'>({cnt})</span></div>"
            f"<div style='font-size:8.5px;color:{MUT};margin:0 0 4px'>{desc}</div>"
            + "".join(rows) + "</div>")
    return "<div style='display:flex;gap:22px;flex-wrap:wrap'>" + "".join(cols) + "</div>"


def focus_line(fs, w=560, h=190):
    """One persona's attention handoff: a fading interest vs a rising one (+steady)."""
    days = list(range(9))
    series = [("↘ " + display_pref_category(fs["fading"][0]), PALETTE["clay"], fs["fading"][1]),
              ("↗ " + display_pref_category(fs["rising"][0]), PALETTE["green"], fs["rising"][1])]
    if "steady" in fs:
        series.append(("→ " + display_pref_category(fs["steady"][0]), PALETTE["slate"], fs["steady"][1]))
    pl, pb, pt, pr = 30, 24, 14, 168
    iw, ih = w - pl - pr, h - pt - pb
    ymax = max(1, max(v for _, _, s in series for v in s))
    xpos = lambda dd: pl + iw * dd / 8
    yof = lambda v: pt + ih * (1 - v / ymax)
    p = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' style='max-width:100%'>"]
    for gv in (0, ymax // 2, ymax):
        y = yof(gv)
        p.append(f"<line x1='{pl}' y1='{y:.0f}' x2='{pl+iw:.0f}' y2='{y:.0f}' stroke='#eef1f3'/>")
        p.append(f"<text x='{pl-4}' y='{y+3:.0f}' font-size='8' fill='{FAINT}' text-anchor='end'>{gv}</text>")
    for dd in days:
        p.append(f"<text x='{xpos(dd):.0f}' y='{h-pb+13}' font-size='8' fill='{INK}' text-anchor='middle'>d{dd}</text>")
    p.append(f"<text x='{pl+iw/2:.0f}' y='{h-3}' font-size='8' fill='{FAINT}' text-anchor='middle'>day of the observation window</text>")
    for ci, (name, col, s) in enumerate(series):
        pts = [(xpos(dd), yof(s[dd])) for dd in days]
        dpath = " ".join((("M" if k == 0 else "L") + f"{x:.0f} {y:.0f}") for k, (x, y) in enumerate(pts))
        p.append(f"<path d='{dpath}' fill='none' stroke='{col}' stroke-width='2'/>")
        for x, y in pts:
            p.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='2.2' fill='{col}'/>")
        ly = pt + 9 + ci * 14
        p.append(f"<line x1='{pl+iw+12}' y1='{ly-3:.0f}' x2='{pl+iw+26}' y2='{ly-3:.0f}' stroke='{col}' stroke-width='2.6'/>")
        p.append(f"<text x='{pl+iw+30}' y='{ly:.0f}' font-size='8.5' fill='{INK}'>{_h.escape(name)}</text>")
    p.append("</svg>")
    return "".join(p)


def focus_gallery(fs_list):
    """Small-multiples of attention handoffs across several personas."""
    cells = []
    for fs in fs_list:
        cells.append(
            f"<div style='flex:1;min-width:300px'>"
            f"<div style='font-size:9px;color:{FAINT};margin:0 0 1px'>persona {fs['uid']}</div>"
            f"{focus_line(fs, w=460, h=150)}</div>")
    return "<div style='display:flex;gap:10px 14px;flex-wrap:wrap'>" + "".join(cells) + "</div>"


def main():
    d = pd.diversity(); b = pd.behavior(); traj = pd.trajectories(); e = pd.evolution()
    fs_list = pd.focus_shift(topn=6)
    tq = pd.tasks_and_queries()
    n = d["n"]
    task_mix = adjusted_task_mix(tq["tasks"])
    task_group_mix = coarse_task_mix(tq["tasks"])
    pref_signal_order = [
        "explicit_positive", "implicit_positive",
        "explicit_negative", "implicit_negative",
    ]
    pref_signal = [(k, b["pref_signal"].get(k, 0)) for k in pref_signal_order]
    content_body = collections.Counter({
        k: v for k, v in b["ctype"].items()
        if "feed-skim" not in str(k).lower()
    })
    pref_cat = display_pref_counter(b["pos_cat"] + b["neg_cat"])

    sec = [START, '<section class="pmv2-diversity">', v2_styles()]
    sec.append('<div class="pmv2-head"><h2>Persona-set Diversity &amp; Preference Evolution</h2>'
               f'<p>{n} personas &middot; {b["total"]//1000}k engagement events across 5 apps &middot; '
               f'{len(d["gender"])} gender/orientation identities, {len(d["ethnicity"])} ethnicities, '
               f'{d["n_careers"]} careers, {d["n_education"]} degrees, {len(d["hidden_types"])} hidden-persona types, '
               f'{len(d["sensitive_topics"])} sensitive-life-event topics, and {len(tq["tasks"])} eval task types.</p></div>')

    # 1) EVAL TASK MIX
    sec.append(v2_grid(
        v2_panel("Eval Task Types", task_mix, topn=20,
                 pretty=lambda s: str(s).replace("_", " "),
                 wide=True),
    ))
    sec.append(v2_grid(
        v2_panel("User Preference Categories", pref_cat, topn=22,
                 pretty=lambda s: str(s).replace("_", " "), wide=True),
    ))
    sec.append(v2_rule())

    # 2) PREFERENCE SIGNALS + CONTENT TYPES WITH BODIES
    sec.append(v2_grid(
        v2_panel("Preference Signal", pref_signal, topn=4,
                 pretty=lambda s: str(s).replace("_", " ").title()),
        v2_panel("Content Type", content_body, topn=5,
                 pretty=lambda s: str(s).replace("_", " "),
                 sub=f"{len(content_body)} distinct · excluding feed-skim"),
    ))
    sec.append(v2_rule())

    # 3) INNER LIFE
    sec.append(v2_grid(
        v2_panel("Hidden-Persona Types", d["hidden_types"], topn=12,
                 pretty=lambda s: str(s).replace("_", " ")),
        v2_panel("Eval Query Groups", task_group_mix, topn=5,
                 pretty=lambda s: s, value_mode="count"),
    ))
    sec.append(v2_rule())

    # 4) BEHAVIOR
    sec.append(v2_grid(
        v2_panel("Events per App", b["apps"], topn=5,
                 pretty=lambda s: str(s).replace("_", " ").title()),
        v2_panel("Interaction Types", b["actions"], topn=10,
                 pretty=lambda s: str(s).replace("_", " ")),
    ))
    sec.append(v2_rule())

    # 5) DEMOGRAPHICS
    sec.append(v2_grid(
        v2_panel("Gender &amp; Orientation", d["gender"], topn=8,
                 force_include=("transgender female",)),
        v2_panel("Race and Ethnicity", d["ethnicity"], topn=16),
    ))
    sec.append(v2_rule())

    # 6) AI COMPANIONS + SENSITIVE TOPICS
    sec.append(v2_grid(
        v2_panel("AI-Companion Personas", d["archetype"], topn=10,
                 pretty=lambda s: str(s).replace("_", " ")),
        v2_panel("Sensitive-Life-Event Topics", d["sensitive_topics"], topn=10,
                 pretty=lambda s: str(s).replace("_", " ")),
    ))

    # 7) TASK MIX + volume
    acts, prefs = b["acts_per"], b["prefeng_per"]
    sec.append('<div style="display:flex;gap:26px;flex-wrap:wrap;margin:20px 0 18px">'
               + chart("Activity &amp; preference volume", volume_histograms(acts, prefs, d["n_prefs"]),
                       sub="each bar counts personas")
               + '</div>')

    # 3c) CONTENT + QUERY WORD CLOUDS (half-width each, matching the other clouds)
    sec.append('<div style="display:flex;gap:22px;flex-wrap:wrap;margin:0 0 16px">'
               '<div style="flex:1;min-width:320px"><div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;'
               f'text-transform:uppercase;color:{MUT};margin:0 0 5px">Content hashtags '
               f'<span style="font-weight:400;color:{FAINT}">&middot; {len(b["htag"]):,} distinct</span></div>'
               + wordcloud(b["htag"], C["content"], topn=40)
               + '</div><div style="flex:1;min-width:320px"><div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;'
               f'text-transform:uppercase;color:{MUT};margin:0 0 5px">User queries</div>'
               + wordcloud(tq["qwords"], C["queries"], topn=40) + '</div></div>')

    # 3d) LIKED / DISLIKED PREFERENCE WORD CLOUDS
    sec.append('<div style="display:flex;gap:22px;flex-wrap:wrap;margin:0 0 16px">'
               '<div style="flex:1;min-width:320px"><div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;'
               f'text-transform:uppercase;color:{C["pos"]};margin:0 0 5px">Liked preferences</div>'
               + wordcloud(b["pos_word"], C["pos"], topn=40)
               + '</div><div style="flex:1;min-width:320px"><div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;'
               f'text-transform:uppercase;color:{C["neg"]};margin:0 0 5px">Disliked preferences</div>'
               + wordcloud(b["neg_word"], C["neg"], topn=40) + '</div></div>')

    # 4) VOICE: readable distribution bars, no word clouds.
    concern_stop = {"whether", "being", "actually", "enough", "people", "private"}
    readable_concerns = collections.Counter({
        k: v for k, v in d["voice_sig_concerns"].items()
        if str(k).lower() not in concern_stop and len(str(k)) > 2
    })
    sec.append('<div class="cap" style="margin-top:8px"><h2 style="font-size:11px">User voice, four readable pieces</h2>'
               f'<span class="note">how they write, how casual they sound, what tone they use, and what words they avoid '
               f'&middot; {d["voice_humor_distinct"]} humor tones, {d["n_register"]} speaking settings, '
               f'{d["n_stances"]:,} tone descriptions</span></div>')
    sec.append(row(
        chart("Capital letters", hbar(d["voice_cap"], C["voice_style"], topn=5, pretty=lambda s: s.replace('_', ' ')),
              sub="100 personas"),
        chart("Formality", hbar(d["voice_formality"], C["voice_style"], topn=5),
              sub="100 personas"),
        chart("Emoji use", hbar(d["voice_emoji"], C["voice_emoji"], topn=5, pretty=lambda s: s or 'none'),
              sub="100 personas"),
    ))
    sec.append(row(
        chart("Sentence length", hbar(d["voice_sent_shape"], C["voice_style"], topn=5,
              pretty=lambda s: s.replace(' dominant', ' sentences'))),
        chart("Directness", hbar(d["voice_hedge"], C["voice_tone"], topn=5,
              pretty=lambda s: {"booster_dominant": "more direct",
                                "hedge_dominant": "more cautious",
                                "balanced": "balanced"}.get(s, s.replace('_', ' '))),
              sub="100 personas"),
        chart("Tone descriptions", hbar(d["voice_stances"], C["voice_tone"], topn=12,
              lblw=158, clip=False), sub=f"{d['n_stances']:,} total"),
    ))
    sec.append(row(
        chart("Speaking setting", hbar(d["voice_registers"], C["voice_setting"], topn=12,
              lblw=158, clip=False), sub=f"{d['n_register']} total"),
        chart("Common concerns", hbar(readable_concerns, C["voice_setting"], topn=12,
              lblw=130, clip=False), sub="repeated voice notes"),
        chart("Words to avoid", hbar(d["voice_phrases_avoid"], C["voice_boundary"], topn=12,
              lblw=130, clip=False), sub="kept out of the voice"),
    ))
    sec.append(row(
        chart("Emoji choices", hbar(d["emoji_palette"], C["voice_emoji"], topn=12,
              lblw=60, clip=False), sub=f"{len(d['emoji_palette'])} distinct"),
        chart("Humor words", hbar(d["humor_words"], C["voice_style"], topn=12,
              lblw=95, clip=False), sub=f"{d['voice_humor_distinct']} tones"),
    ))

    # 5) EVOLUTION: per-preference trajectory patterns (real examples)
    tot_traj = sum(traj["counts"].values()) or 1
    sec.append('<div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;'
               f'color:{MUT};margin:8px 0 3px">How individual preferences evolve over the window</div>'
               f'<div style="font-size:9px;color:{FAINT};margin:0 0 8px">each small line is one persona and one preference category over days 0&ndash;8; '
               f'{tot_traj} lines are grouped by trend, with about 10 examples shown per trend</div>')
    sec.append(pattern_gallery(traj))
    # multi-persona focus shift (a fading interest handing off to a rising one)
    if fs_list:
        sec.append(f'<div style="font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;'
                   f'color:{MUT};margin:16px 0 2px">Attention shifts within a persona</div>'
                   f'<div style="font-size:9px;color:{FAINT};margin:0 0 6px">one interest fades (↘) as another rises (↗); '
                   '→ = a steady anchor &middot; six personas</div>')
        sec.append(focus_gallery(fs_list))
    pc = traj["counts"]
    sec.append(f'<div class="abldefs" style="max-width:820px;margin-top:10px">Preferences are <b>not static</b>: of '
               f'{tot_traj} sustained persona&times;category trajectories, <b>{100*pc.get("reinforced",0)//tot_traj}% are '
               f'reinforced</b> (steady re-engagement), <b>{100*pc.get("emerging",0)//tot_traj}% emerge</b> (grow into a '
               f'new focus), <b>{100*pc.get("diminishing",0)//tot_traj}% diminish</b> (fade as attention shifts), and a few '
               f'are <b>bursty</b> (one episodic spike). Examples are real (persona id shown): e.g. interests that climb '
               'across the window, others that decay to zero, and steady anchors, so a model must track each '
               'preference&rsquo;s direction, not just its presence. '
               f'Overall {100*e["horizon"]["long_term"]//max(1,e["npref"])}% of engagements are long-term, '
               f'{100*e["horizon"].get("short_term",0)//max(1,e["npref"])}% short-term (transient, with a stop condition).</div>')
    sec.append('</section>')
    sec.append(END)
    block = "\n".join(sec)

    with html_lock():
        for target in (HTML, BASE_HTML):
            html = target.read_text()
            if html.count(START) > 1 or html.count(END) > 1 or html.count("<!doctype") > 1:
                raise SystemExit(f"{target} corrupted - restore from base before re-injecting")
            if START in html and END in html:
                html = html.split(START)[0] + block + html.split(END, 1)[1]
            else:
                anchor = "<section>"          # before the first (Accuracy) section
                html = html.replace(anchor, block + "\n\n" + anchor, 1)
            target.write_text(html)
            print("injected diversity section into", target)


if __name__ == "__main__":
    main()
