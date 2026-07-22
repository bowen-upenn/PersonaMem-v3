#!/usr/bin/env python3
"""Render the two ablation figures (QA auto-verification + cross-judge
agreement) as inline SVG into results/aggregate/html/results_tables.html.

Pure-SVG, Optimistic font, matched to the existing report palette. Reads:
  results/audit/qa_audit_p1/audit_summary.json
  results/audit/qa_agree/agreement.json
  results/audit/judge_agreement_p1/all_scores.jsonl
  results/audit/judge_agreement_p3/accuracy_agreement.json

Idempotent: replaces content between the QA_AUDIT / JUDGE_AGREE markers,
inserting before <footer> on first run. Holds the single-writer lock for
the whole read-modify-write (CLAUDE.md).
"""
from __future__ import annotations

import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _ablation_stats import pearson, spearman, krippendorff_alpha_interval  # noqa: E402
from _htmllock import html_lock  # noqa: E402

ROOT = Path(_HERE).resolve().parents[1]
HTML = ROOT / "results/aggregate/html/results_tables.html"
QA_JSON = ROOT / "results/audit/qa_audit_p1/audit_summary.json"
QA_AGREE_JSON = ROOT / "results/audit/qa_agree/agreement.json"
JUDGE_JSONL = ROOT / "results/audit/judge_agreement_p1/all_scores.jsonl"
ACCURACY_AGREE_JSON = ROOT / "results/audit/judge_agreement_p3/accuracy_agreement.json"

# ---- palette (harmonised with the report's existing ink/accent tones) ----
INK, INK2, MUTED, FAINT, HAIR, VRULE = "#16242c", "#33424b", "#7c8a93", "#aab4bb", "#e6ebee", "#c7cfd5"
GREEN = "#4E9C72"          # pass bars
GREEN_HI = "#9ED4B4"
AMBERFAIL = "#C98A4A"      # below-100 accent
JUDGE_COLOR = {
    "gpt-5.5":          "#2F6DB4",   # slate blue
    "gpt-5.4-mini":     "#C99A3C",   # ochre / gold (cheaper same-family judge)
    "claude-opus-4.8":  "#3E948A",   # teal-green
}
JUDGE_LABEL = {
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-opus-4.8": "Opus-4.8",
}
JUDGE_SHORT = {  # tight labels for the heatmap column headers
    "gpt-5.5": "5.5", "gpt-5.4-mini": "5.4m", "claude-opus-4.8": "Opus",
}
JUDGES = ("gpt-5.5", "claude-opus-4.8")
DIMS = ("preference_alignment", "helpfulness", "appropriate_restraint", "no_hallucinated_preference")
DIM_LABEL = {
    "preference_alignment": "Pref. alignment",
    "helpfulness": "Helpfulness",
    "appropriate_restraint": "Restraint",
    "no_hallucinated_preference": "No hallucination",
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _lerp(c1, c2, t):
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{int(a[i] + (b[i]-a[i])*t):02x}" for i in range(3))


# ===========================================================================
# Figure 1 — QA auto-verification
# ===========================================================================

# (key, display title, plain-English description, tier).
# Each benchmark item is a test question with a "good" answer and a deliberately
# "weaker" alternative; these checks vet every item before it ships. Titles and
# wording are written for a reader unfamiliar with the project.
CRITERIA = [
    ("__g", "Is the test item complete?", "", ""),
    ("completeness", "Nothing missing",
     "All required fields are present.", "content"),
    ("schema_sanity", "Valid format",
     "Candidates, tools, and history links are valid.", "content"),
    ("__g", "Does it read like a real conversation?", "", ""),
    ("naturalness", "Sounds like a real person",
     "The user query sounds natural.", "content"),
    ("telegraph_avoidance", "Doesn’t give itself away",
     "The answer does not name private user traits.", "content"),
    ("no_refusal", "Actually answers",
     "The answer attempts the task.", "content"),
    ("no_rubric_leak", "No behind-the-scenes text",
     "No grading or system text leaks.", "content"),
    ("__g", "Is the personalization real?", "", ""),
    ("context_required", "Truly needs this user",
     "The good answer needs this user’s history.", "content"),
    ("context_restraint", "Fair “don’t overshare” test",
     "The restraint item can be answered safely.", "content"),
    ("gt_alignment", "Tests the right thing",
     "It tests the intended preference.", "content"),
    ("privacy_leak", "Keeps private things private",
     "No off-limits preference is revealed.", "content"),
    ("__g", "Is it a fair comparison?", "", ""),
    ("inferior_axis_check", "Weaker answer fails the right way",
     "The weaker answer fails on the tested skill.", "content"),
    ("sensitive_probe_placement", "Clue comes before the question",
     "The evidence appears before the test.", "content"),
]


def build_qa_section() -> str:
    s = json.loads(QA_JSON.read_text())
    bydim = s["by_dimension"]
    nq = s["n_queries"]
    qa_agree = json.loads(QA_AGREE_JSON.read_text()) if QA_AGREE_JSON.exists() else None

    # content-tier headline (gates a static re-audit can fairly assess)
    cp = cf = 0
    for k, *_rest in CRITERIA:
        if k in ("__g",):
            continue
        tier = _rest[2]
        d = bydim.get(k)
        if d and tier == "content":
            cp += d["passed"]
            cf += d["failed"]
    head_rate = cp / (cp + cf) * 100 if (cp + cf) else 0

    # layout
    W = 1100
    LX = 8                 # label x
    LW = 560               # label block width
    BX = 588               # bar track x
    BW = 372               # bar track width
    VX = BX + BW + 14      # value x
    rh, gh = 30, 26        # row / group-header height
    y = 14
    parts = [f"<svg viewBox='0 0 {W} 0' width='100%' style='max-width:100%;font-family:Optimistic,sans-serif' xmlns='http://www.w3.org/2000/svg' id='qasvg'>"]
    # axis guides at 0/50/100 over the bar track
    def track_guides(y0, y1):
        g = []
        for frac in (0.0, 0.5, 1.0):
            gx = BX + BW * frac
            g.append(f"<line x1='{gx:.1f}' y1='{y0}' x2='{gx:.1f}' y2='{y1}' stroke='{HAIR}' stroke-width='1'/>")
        return "".join(g)

    rows_svg = []
    for entry in CRITERIA:
        k = entry[0]
        if k == "__g":                         # category header
            title = entry[1]
            y += 16
            rows_svg.append(
                f"<text x='{LX}' y='{y+11}' font-size='10.5' font-weight='700' "
                f"letter-spacing='0.06em' fill='{INK}' "
                f"text-transform='uppercase'>{esc(title.upper())}</text>")
            y += 28
            continue
        _, title, desc, tier = entry
        d = bydim.get(k, {})
        ev = d.get("evaluated", 0)
        rate = d.get("pass_rate")
        lines = _wrap_lines(desc, LW, 9.3, max_lines=1)
        # check name (title) + bar/value sit on the first line; description flows
        # below across as many lines as it needs, then the row advances past it.
        rows_svg.append(
            f"<text x='{LX}' y='{y+12}' font-size='11.5' font-weight='700' fill='{INK}'>{esc(title)}</text>")
        for i, ln in enumerate(lines):
            rows_svg.append(
                f"<text x='{LX}' y='{y+26+i*11}' font-size='9.3' fill='{MUTED}'>{esc(ln)}</text>")
        if ev == 0 or rate is None:
            rows_svg.append(track_guides(y, y + 20))
            rows_svg.append(f"<text x='{BX+BW/2:.0f}' y='{y+14}' font-size='9.5' fill='{FAINT}' text-anchor='middle'>not exercised by this persona</text>")
        else:
            fillw = BW * rate
            rows_svg.append(track_guides(y, y + 20))
            rows_svg.append(f"<rect x='{BX}' y='{y+3}' width='{BW}' height='13' rx='6.5' fill='{HAIR}'/>")
            rows_svg.append(
                f"<rect x='{BX}' y='{y+3}' width='{max(fillw,6):.1f}' height='13' rx='6.5' "
                f"fill='{GREEN}' opacity='0.92'/>")
            pct = rate * 100
            chk = "  ✓" if pct >= 99.95 else ""
            rows_svg.append(
                f"<text x='{VX}' y='{y+13}' font-size='11.5' font-weight='700' fill='{INK}' "
                f"font-variant-numeric='tabular-nums'>{pct:.0f}%{chk}</text>")
            rows_svg.append(
                f"<text x='{VX+58}' y='{y+13}' font-size='9' fill='{FAINT}' "
                f"font-variant-numeric='tabular-nums'>{d['passed']}/{ev}</text>")
        y += 30 + len(lines) * 11          # adaptive: room for every wrapped line
    track_bottom = y

    # 50%/100% tick labels at bottom of the chart
    rows_svg.append(f"<text x='{BX:.0f}' y='{track_bottom+12}' font-size='8.5' fill='{FAINT}' text-anchor='middle'>0%</text>")
    rows_svg.append(f"<text x='{BX+BW*0.5:.0f}' y='{track_bottom+12}' font-size='8.5' fill='{FAINT}' text-anchor='middle'>50%</text>")
    rows_svg.append(f"<text x='{BX+BW:.0f}' y='{track_bottom+12}' font-size='8.5' fill='{FAINT}' text-anchor='middle'>100% pass</text>")

    H = track_bottom + 24
    svg = parts[0].replace("viewBox='0 0 1100 0'", f"viewBox='0 0 {W} {H}'") + "".join(rows_svg) + "</svg>"

    def inline_stat(big, label, col=INK):
        return (
            f"<div style='display:flex;flex-direction:column;gap:1px;min-width:210px'>"
            f"<div style='font-size:24px;font-weight:700;color:{col};letter-spacing:-0.02em;"
            f"font-variant-numeric:tabular-nums'>{big}</div>"
            f"<div style='font-size:10px;font-weight:400;color:{MUTED};line-height:1.35;max-width:250px'>{label}</div>"
            f"</div>"
        )

    stat_items = [inline_stat(f"{head_rate:.1f}%", "quality-check pass rate", GREEN)]
    if qa_agree:
        pooled = qa_agree.get("pooled", {})
        stat_items.append(inline_stat(f"{pooled.get('agreement_pct', 0):.1f}%",
                                      "GPT-5.5/Opus-4.8 pass-fail agreement", GREEN))
        stat_items.append(inline_stat(f"{len(qa_agree.get('by_dim', {}))}",
                                      "quality audit dimensions", INK))
    stat_row = (
        "<div style='display:flex;align-items:flex-start;gap:34px;margin:2px 2px 16px;flex-wrap:nowrap'>"
        + "".join(stat_items) + "</div>"
    )
    note = ""
    return (
        "<!-- QA_AUDIT_START -->\n"
        "<section>\n"
        "<div class=\"cap\"><h2>Benchmark quality control: automatic checks</h2>"
        f"<span class=\"note\">single-persona QC audit + GPT-5.5/Opus overlap &middot; {nq} questions</span></div>\n"
        f"{stat_row}\n{svg}\n{note}\n"
        "</section>\n"
        "<!-- QA_AUDIT_END -->"
    )


def _wrap_lines(text, maxw_px, fs, max_lines=3):
    """Greedy word-wrap into <= max_lines lines (char-width approx)."""
    cpl = max(8, int(maxw_px / (fs * 0.52)))
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= cpl:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and cur and lines[-1] != cur:
        lines[-1] = lines[-1].rstrip(".,") + "…"
    return lines


def _wrap_text(text, x, y, maxw_px, fs, fill, lh=11, max_lines=3):
    out = []
    for i, ln in enumerate(_wrap_lines(text, maxw_px, fs, max_lines)):
        out.append(f"<text x='{x}' y='{y + i*lh}' font-size='{fs}' fill='{fill}'>{esc(ln)}</text>")
    return "".join(out)


# ===========================================================================
# Figure 2 — cross-judge agreement
# ===========================================================================

def _load_judge():
    rows = [json.loads(l) for l in JUDGE_JSONL.read_text().splitlines() if l.strip()]
    cell = defaultdict(dict)   # (item,dim)->judge->score
    resp = defaultdict(dict)   # item->judge->mean
    for r in rows:
        vals = [r[d] for d in DIMS if r.get(d) is not None]
        if vals:
            resp[r["item_key"]][r["judge"]] = float(np.mean(vals))
        for d in DIMS:
            if r.get(d) is not None:
                cell[(r["item_key"], d)][r["judge"]] = r[d]
    return rows, cell, resp


def _axis(x0, y0, w, h, ymax, fill_label, ticks=None):
    g = [f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none'/>"]
    g.append(f"<line x1='{x0}' y1='{y0+h}' x2='{x0+w}' y2='{y0+h}' stroke='{VRULE}' stroke-width='1'/>")
    g.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0+h}' stroke='{VRULE}' stroke-width='1'/>")
    return "".join(g)


def build_judge_section() -> str:
    rows, cell, resp = _load_judge()
    keys = list(cell)
    score_cells = sum(1 for k in keys for j in JUDGES if cell[k].get(j) is not None)
    complete_cells = sum(1 for k in keys if all(cell[k].get(j) is not None for j in JUDGES))
    # agreement stats
    mat = [[cell[k].get(j, float("nan")) for k in keys] for j in JUDGES]
    alpha = krippendorff_alpha_interval(mat)
    pair_r = {}
    for a in range(len(JUDGES)):
        for b in range(a + 1, len(JUDGES)):
            xa = [cell[k].get(JUDGES[a]) for k in keys]
            xb = [cell[k].get(JUDGES[b]) for k in keys]
            pair_r[(JUDGES[a], JUDGES[b])] = (pearson(xa, xb), spearman(xa, xb))
    pearsons = [v[0] for v in pair_r.values() if v[0] is not None]
    mean_r = float(np.mean(pearsons)) if pearsons else None

    # per-judge: histogram(0..10), overall mean, real/foil means
    hist, jmean, rfmean = {}, {}, {}
    for j in JUDGES:
        allv = [r[d] for r in rows if r["judge"] == j for d in DIMS if r.get(d) is not None]
        h, _ = np.histogram(allv, bins=range(0, 12))
        hist[j] = h
        jmean[j] = float(np.mean(allv))
        rr = [r[d] for r in rows if r["judge"] == j and r["population"] == "real" for d in DIMS if r.get(d) is not None]
        ff = [r[d] for r in rows if r["judge"] == j and r["population"] == "foil" for d in DIMS if r.get(d) is not None]
        rfmean[j] = (float(np.mean(rr)), float(np.mean(ff)))

    n_items = len({k[0] for k in keys})
    W = 1100
    P = []  # svg parts
    # ---------------- Row 1 ----------------
    R1H = 250
    # Panel A: grouped histogram --------------------------------------------
    ax0, ay0, aw, ah = 36, 26, 380, 168
    A = [f"<text x='8' y='14' font-size='10.5' font-weight='700' letter-spacing='0.04em' fill='{INK}'>SCORE DISTRIBUTION</text>",
         f"<text x='8' y='{ay0+ah+34}' font-size='9' fill='{MUTED}'>rubric score (0–10), all four dimensions pooled</text>"]
    A.append(_axis(ax0, ay0, aw, ah, None, ""))
    hmax = max(1, max(int(hist[j].max()) for j in JUDGES))
    for gy in range(0, hmax + 1, max(1, round(hmax / 4))):
        yy = ay0 + ah - ah * gy / hmax
        A.append(f"<line x1='{ax0}' y1='{yy:.1f}' x2='{ax0+aw}' y2='{yy:.1f}' stroke='{HAIR}' stroke-width='1'/>")
        A.append(f"<text x='{ax0-4}' y='{yy+3:.1f}' font-size='8' fill='{FAINT}' text-anchor='end'>{gy}</text>")
    binw = aw / 11.0
    nb = len(JUDGES)
    bw = binw / (nb + 1.4)
    for sc in range(11):
        bx0 = ax0 + binw * sc + (binw - bw * nb) / 2
        for ji, j in enumerate(JUDGES):
            v = int(hist[j][sc])
            bh = ah * v / hmax
            A.append(f"<rect x='{bx0 + ji*bw:.1f}' y='{ay0+ah-bh:.1f}' width='{bw-1:.1f}' height='{bh:.1f}' fill='{JUDGE_COLOR[j]}' opacity='0.9'/>")
        A.append(f"<text x='{ax0 + binw*sc + binw/2:.1f}' y='{ay0+ah+12}' font-size='8' fill='{FAINT}' text-anchor='middle'>{sc}</text>")
    # legend (top-right of the panel)
    lx = ax0 + aw - 132
    for li, j in enumerate(JUDGES):
        ly = ay0 + 4 + li * 14
        A.append(f"<rect x='{lx}' y='{ly}' width='9' height='9' rx='2' fill='{JUDGE_COLOR[j]}'/>")
        A.append(f"<text x='{lx+13}' y='{ly+8}' font-size='8.5' fill='{INK2}'>{JUDGE_LABEL[j]}</text>")
    P.append(f"<g transform='translate(0,0)'>{''.join(A)}</g>")

    # Panel B: 3x3 Pearson heatmap ------------------------------------------
    bx, by = 470, 26
    cellpx = 50
    B = [f"<text x='{bx}' y='14' font-size='10.5' font-weight='700' letter-spacing='0.04em' fill='{INK}'>PAIRWISE AGREEMENT</text>",
         f"<text x='{bx}' y='{by+len(JUDGES)*cellpx+52}' font-size='9' fill='{MUTED}'>Pearson r on each item×dimension score</text>"]
    gx0 = bx + 96
    gy0 = by + 14
    for i, ji in enumerate(JUDGES):
        B.append(f"<text x='{gx0-6}' y='{gy0 + i*cellpx + cellpx/2 + 3:.0f}' font-size='9' font-weight='700' fill='{JUDGE_COLOR[ji]}' text-anchor='end'>{JUDGE_LABEL[ji]}</text>")
        B.append(f"<text x='{gx0 + i*cellpx + cellpx/2:.0f}' y='{gy0-5}' font-size='9' font-weight='700' fill='{JUDGE_COLOR[ji]}' text-anchor='middle'>{JUDGE_SHORT[ji]}</text>")
    for i, ji in enumerate(JUDGES):
        for jx, jj in enumerate(JUDGES):
            if i == jx:
                r = 1.0
            else:
                r = pair_r.get((ji, jj), pair_r.get((jj, ji)))[0]
            t = 0.0 if r is None else max(0.0, min(1.0, (r - 0.5) / 0.5))
            col = _lerp("#eef2f5", "#2F6DB4", t)
            cx, cy = gx0 + jx * cellpx, gy0 + i * cellpx
            B.append(f"<rect x='{cx}' y='{cy}' width='{cellpx-3}' height='{cellpx-3}' rx='4' fill='{col}'/>")
            tc = "#fff" if t > 0.55 else INK2
            label = "n/a" if r is None else f"{r:.2f}"
            B.append(f"<text x='{cx+(cellpx-3)/2:.0f}' y='{cy+(cellpx-3)/2+4:.0f}' font-size='11' font-weight='700' fill='{tc}' text-anchor='middle' font-variant-numeric='tabular-nums'>{label}</text>")
    P.append("".join(B))

    # Panel C: real-reply mean by judge (final score; foils not shown) -------
    cx0, cy0, cw, ch = 760, 26, 300, 168
    C = [f"<text x='{cx0}' y='14' font-size='10.5' font-weight='700' letter-spacing='0.04em' fill='{INK}'>REAL-REPLY MEAN</text>",
         f"<text x='{cx0}' y='{cy0+ch+34}' font-size='9' fill='{MUTED}'>mean rubric score on the real model replies (0–10)</text>"]
    C.append(_axis(cx0, cy0, cw, ch, 10, ""))
    for gy in range(0, 11, 2):
        yy = cy0 + ch - ch * gy / 10
        C.append(f"<line x1='{cx0}' y1='{yy:.1f}' x2='{cx0+cw}' y2='{yy:.1f}' stroke='{HAIR}' stroke-width='1'/>")
        C.append(f"<text x='{cx0-4}' y='{yy+3:.1f}' font-size='8' fill='{FAINT}' text-anchor='end'>{gy}</text>")
    grpw = cw / len(JUDGES)
    for ji, j in enumerate(JUDGES):
        rmean = rfmean[j][0]
        bxx = cx0 + grpw * ji + grpw / 2 - 24
        bh = ch * rmean / 10
        C.append(f"<rect x='{bxx:.0f}' y='{cy0+ch-bh:.1f}' width='48' height='{bh:.1f}' rx='2' fill='{JUDGE_COLOR[j]}' opacity='0.92'/>")
        C.append(f"<text x='{bxx+24:.0f}' y='{cy0+ch-bh-4:.1f}' font-size='12' font-weight='700' fill='{INK}' text-anchor='middle' font-variant-numeric='tabular-nums'>{rmean:.1f}</text>")
        C.append(f"<text x='{bxx+24:.0f}' y='{cy0+ch+12}' font-size='9' font-weight='700' fill='{JUDGE_COLOR[j]}' text-anchor='middle'>{JUDGE_LABEL[j]}</text>")
    P.append("".join(C))

    # ---------------- Row 2: pairwise scatter ----------------
    R2Y = R1H + 14
    S = [f"<text x='8' y='{R2Y+2}' font-size='10.5' font-weight='700' letter-spacing='0.04em' fill='{INK}'>PER-ITEM SCORES, JUDGE vs JUDGE</text>"]
    pairs = [(JUDGES[a], JUDGES[b]) for a in range(len(JUDGES)) for b in range(a + 1, len(JUDGES))]
    S.append(
        f"<text x='320' y='{R2Y+2}' font-size='9' fill='{MUTED}'>each dot = one response×dimension "
        f"(up to {len(keys)} per pair); jittered; dashed = perfect agreement</text>"
    )
    sp_w = 285 if len(pairs) >= 3 else 300
    sp_h = 210
    sp_gap = 45 if len(pairs) >= 3 else 60
    sy0 = R2Y + 16
    for pi, (ja, jb) in enumerate(pairs):
        sx0 = 36 + pi * (sp_w + sp_gap)
        S.append(_axis(sx0, sy0, sp_w, sp_h, 10, ""))
        # gridlines
        for gv in range(0, 11, 2):
            xx = sx0 + sp_w * gv / 10
            yy = sy0 + sp_h - sp_h * gv / 10
            S.append(f"<line x1='{sx0}' y1='{yy:.1f}' x2='{sx0+sp_w}' y2='{yy:.1f}' stroke='{HAIR}' stroke-width='0.7'/>")
            S.append(f"<text x='{sx0-4}' y='{yy+3:.1f}' font-size='7.5' fill='{FAINT}' text-anchor='end'>{gv}</text>")
            S.append(f"<text x='{xx:.1f}' y='{sy0+sp_h+11}' font-size='7.5' fill='{FAINT}' text-anchor='middle'>{gv}</text>")
        # y=x line
        S.append(f"<line x1='{sx0}' y1='{sy0+sp_h}' x2='{sx0+sp_w}' y2='{sy0}' stroke='{FAINT}' stroke-width='1' stroke-dasharray='4 3'/>")
        # points (deterministic jitter from index)
        rng = np.random.RandomState(7)
        for k in keys:
            va, vb = cell[k].get(ja), cell[k].get(jb)
            if va is None or vb is None:
                continue
            jx = (rng.rand() - 0.5) * 0.5
            jy = (rng.rand() - 0.5) * 0.5
            px = sx0 + sp_w * (va + jx) / 10
            py = sy0 + sp_h - sp_h * (vb + jy) / 10
            S.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='2.0' fill='{JUDGE_COLOR[ja]}' opacity='0.38'/>")
        r = pair_r.get((ja, jb), pair_r.get((jb, ja)))[0]
        r_label = "n/a" if r is None else f"{r:.2f}"
        S.append(f"<text x='{sx0+8}' y='{sy0+14}' font-size='10' font-weight='700' fill='{INK}'>r = {r_label}</text>")
        S.append(f"<text x='{sx0+sp_w/2:.0f}' y='{sy0+sp_h+26}' font-size='9' font-weight='700' fill='{JUDGE_COLOR[ja]}' text-anchor='middle'>{JUDGE_LABEL[ja]} <tspan fill='{MUTED}' font-weight='400'>(x)</tspan> &#8594; <tspan fill='{JUDGE_COLOR[jb]}'>{JUDGE_LABEL[jb]}</tspan> <tspan fill='{MUTED}' font-weight='400'>(y)</tspan></text>")
    P.append("".join(S))

    H = sy0 + sp_h + 36
    svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:100%;font-family:Optimistic,sans-serif' "
           f"xmlns='http://www.w3.org/2000/svg'>{''.join(P)}</svg>")

    # stat callouts
    def callout(big, small, col=INK):
        return (f"<div style='display:flex;flex-direction:column;gap:1px'>"
                f"<div style='font-size:26px;font-weight:700;color:{col};letter-spacing:-0.02em;font-variant-numeric:tabular-nums'>{big}</div>"
                f"<div style='font-size:10px;color:{MUTED};line-height:1.35;max-width:230px'>{small}</div></div>")
    nj = len(JUDGES)
    njw = {2: "two", 3: "three", 4: "four"}.get(nj, str(nj))
    rs = sorted(v[0] for v in pair_r.values() if v[0] is not None)
    rmeans = " / ".join(f"{rfmean[j][0]:.1f}" for j in JUDGES)
    rlabels = " / ".join(JUDGE_LABEL[j] for j in JUDGES)
    across = "between" if nj == 2 else "across"
    if not rs:
        r_callout = callout("n/a", "not enough overlapping scores for a pairwise Pearson r")
    elif len(rs) == 1:
        r_callout = callout(f"{rs[0]:.2f}", f"Pearson <i>r</i> ({JUDGE_LABEL[JUDGES[0]]} &harr; {JUDGE_LABEL[JUDGES[1]]}) on each item&times;dimension score")
    else:
        r_callout = callout(f"{mean_r:.2f}", f"mean pairwise Pearson <i>r</i> ({rs[0]:.2f}&ndash;{rs[-1]:.2f} across the {len(rs)} pairs)")
    callouts = (
        "<div style='display:flex;gap:40px;margin:2px 2px 16px;flex-wrap:wrap'>"
        + callout(f"{alpha:.2f}", f"Krippendorff&rsquo;s &alpha; (interval) {across} the {njw} judges on current collected score cells", GREEN)
        + r_callout
        + callout(rmeans, f"real-reply mean ({rlabels}); the headline barely moves with the judge")
        + "</div>"
    )
    gpt_opus_r = pair_r.get(("gpt-5.5", "claude-opus-4.8"), (None, None))[0]
    accuracy_note = (
        f"<p style='font-size:10px;color:{MUTED};margin:0 2px 12px;line-height:1.45;max-width:940px'>"
        f"The GPT-5.5 &harr; Opus table-accuracy agreement is the pairwise score agreement shown here: "
        f"Pearson <i>r</i>={gpt_opus_r:.2f} across <b style='color:{INK2}'>{complete_cells}</b> shared "
        f"response&times;dimension scores, with real-reply means {rfmean['gpt-5.5'][0]:.2f} vs "
        f"{rfmean['claude-opus-4.8'][0]:.2f}. Broader p1&ndash;p3 Opus aggregate score files exist, "
        f"but their per-item sidecars are not present, so this fixed-prompt study is the current item-level "
        f"GPT-5.5/Opus agreement evidence.</p>"
    )
    lead = (
        f"<p style='font-size:11px;color:{MUTED};margin:0 2px 12px;line-height:1.5;max-width:940px'>"
        f"A held-out persona&rsquo;s real GPT-5.5 long-context replies and matched foils are re-scored on the "
        f"<b style='color:{INK2}'>identical</b> chatbot rubric by "
        f"<b style='color:{JUDGE_COLOR['gpt-5.5']}'>GPT-5.5</b> and "
        f"<b style='color:{JUDGE_COLOR['claude-opus-4.8']}'>Claude&nbsp;Opus&nbsp;4.8</b>. "
        f"The fixed-prompt panel uses <b style='color:{INK2}'>{score_cells}</b> collected score cells over "
        f"<b style='color:{INK2}'>{len(keys)}</b> response&times;dimension units "
        f"(<b style='color:{INK2}'>{complete_cells}</b> have both judges). "
        f"Agreement is measured over real replies plus foils to supply score spread; the real-reply means show how much the reported table score moves with judge choice.</p>"
    )
    return (
        "<!-- JUDGE_AGREE_START -->\n"
        "<section>\n"
        "<div class=\"cap\"><h2>Table accuracy: agreement between judge models</h2>"
        "<span class=\"note\">current collected scores &middot; identical rubric &middot; GPT-5.5 vs Opus-4.8</span></div>\n"
        f"{lead}\n{callouts}\n{accuracy_note}\n{svg}\n"
        "</section>\n"
        "<!-- JUDGE_AGREE_END -->"
    )


# ===========================================================================
def _inject(html_doc: str, marker_start: str, marker_end: str, section: str) -> str:
    if marker_start in html_doc and marker_end in html_doc:
        pre = html_doc.split(marker_start)[0]
        post = html_doc.split(marker_end, 1)[1]
        return pre + section + post
    # first run: insert before <footer>
    anchor = "<footer>"
    assert anchor in html_doc, "no <footer> anchor in results_tables.html"
    return html_doc.replace(anchor, section + "\n" + anchor, 1)


def main():
    qa = build_qa_section()
    jg = build_judge_section()
    with html_lock():
        doc = HTML.read_text()
        doc = _inject(doc, "<!-- QA_AUDIT_START -->", "<!-- QA_AUDIT_END -->", qa)
        doc = _inject(doc, "<!-- JUDGE_AGREE_START -->", "<!-- JUDGE_AGREE_END -->", jg)
        HTML.write_text(doc)
    print("[render] injected QA_AUDIT + JUDGE_AGREE sections into", HTML)


if __name__ == "__main__":
    main()
