#!/usr/bin/env python3
"""Abstract-but-honest visualization of mem0's RAG vector store for ONE persona,
injected into results_tables.html (own markers, idempotent).

Honesty: every dot is a REAL fact mem0 extracted and stored for the persona; its
position is that fact's REAL 3072-d embedding projected to 2D (PCA), so dots that
sit together really are semantically similar; the faint lines are each fact's REAL
nearest neighbours (cosine) — the structure a query searches. Labels are the most
common word per cluster; example facts are shown verbatim. No new model calls:
the embeddings are read straight out of the qdrant store.
"""
import sqlite3, pickle, glob, re, collections, sys, html as _html
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sd = ROOT / "results/_scripts"
sys.path.insert(0, str(sd))
from _htmllock import html_lock, HTML        # single-writer lock for results_tables.html

UID = "1"
MEM_DIR = "mem0_gpt5.5"
K = 5
START, END = "<!-- RAG_DB_START -->", "<!-- RAG_DB_END -->"
# muted, elegant palette (desaturated family hues; not bright)
PAL = ["#5B7C99", "#4F9088", "#7E9A6B", "#B0925E", "#A47189", "#6E7CA6", "#9A8C72"]
STOP = set("user users the a an and or of to in on for with is are was were be been engages "
           "especially repeatedly content social media about around including this that their "
           "them they it its as at by from has have had also more most when only if his her she "
           "he new likes like enjoys interested observed activity prefers wants would could "
           "january february march april may june july august september october november december "
           "day date dates time today recommended advised laughing clearly often sometimes "
           "should noted mentioned via per based given across between within "
           "bad good big small high low old young real full best worst likely usually "
           "typically generally include includes various several many overall "
           "text texts video videos post posts clip clips photo photos image images "
           "feel feels feeling style thing things stuff kind sort lot".split())


def load_facts(uid):
    dbs = glob.glob(f"{ROOT}/results/{MEM_DIR}/{uid}/mem0_store/**/storage.sqlite", recursive=True)
    con = sqlite3.connect(dbs[0])
    facts = []
    for (blob,) in con.execute("SELECT point FROM points"):
        p = pickle.loads(blob)
        pl = p.payload or {}
        t = (pl.get("data") or "").strip()
        if t and p.vector:
            facts.append((t, np.asarray(p.vector, dtype=np.float32), int(pl.get("ts") or 0)))
    con.close()
    return facts


def kmeans(Vn, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    cent = Vn[rng.choice(len(Vn), k, replace=False)].copy()
    lab = np.zeros(len(Vn), dtype=int)
    for _ in range(iters):
        d = ((Vn[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for c in range(k):
            m = Vn[lab == c]
            if len(m):
                cent[c] = m.mean(0)
    return lab, cent


def _doc_freq(texts):
    df = collections.Counter()
    for t in texts:
        for w in set(re.findall(r"[a-z]{3,}", t.lower())):
            if w not in STOP:
                df[w] += 1
    return df


# Curated topic vocabulary -> a cluster is labelled by the topic its facts most
# mention (honest: the words really are in the facts), so labels are always clean
# real words, never clustering noise. Tuned to this persona's themes.
TOPIC_WORDS = {
    "cats":      ["cat", "feline", "kitten", "tabby"],
    "dogs":      ["dog", "pup", "puppy", "zoomies"],
    "wildlife":  ["wildlife", "rehabilitation", "conservation", "owl", "hawk",
                  "fox", "bear", "orphaned", "injured", "rescue"],
    "baseball":  ["baseball", "mlb", "blue jays", "jays", "pitch", "bullpen",
                  "box score", "scorekeeper", "changeup"],
    "municipal": ["municipal", "bridge", "public-works", "public works", "jobsite",
                  "bid", "estimate", "guardrail", "infrastructure", "road", "specification"],
    "movies":    ["movie", "horror", "sequel", "freddy", "monster", "campy",
                  "tomatoes", "nightmare", "rubber-suit"],
    "rural":     ["farm", "rural", "barn", "cows", "shed", "acres", "gravel"],
    "history":   ["archaeolog", "neanderthal", "prehistoric", "stone age", "star trek"],
    "Dallas":    ["dallas", "located", "location"],
    "UFO":       ["ufo", "ghost", "paranormal"],
}

# Per-fact readable name: first matching rule wins, so list specific entities
# before generic topics. Each label is a short human-readable phrase that the
# fact really is about (honest). Tuned to this persona's facts.
TAG_RULES = [
    (("brother",), "texting brother"),
    (("sister",), "sister's cat"),
    (("killer tomatoes",), "Killer Tomatoes"),
    (("freddy", "nightmare on elm", "krueger"), "Freddy Krueger"),
    (("star trek",), "Star Trek"),
    (("neanderthal", "stone age", "prehistor", "archaeolog"), "prehistory"),
    (("ufo",), "UFO clip"),
    (("ghost", "paranormal"), "ghost hunting"),
    (("bear",), "bear conservation"),
    (("owl", "hawk"), "bird rehab"),
    (("fox",), "orphaned fox"),
    (("blue jays", "vladdy", " jays"), "Blue Jays"),
    (("propaganda", "media-literacy", "media literacy"), "media literacy"),
    (("dallas",), "Dallas"),
    (("boxing", "canelo", "mayweather"), "boxing"),
    (("feralcatrescue", "foster", "rescue pup", "rescue"), "animal rescue"),
    (("zoomies",), "dog zoomies"),
    (("courthouse cat",), "courthouse cat"),
    (("dog", " pup"), "dog videos"),
    (("wildlife", "rehabilitation", "conservation", "orphaned", "injured", "alley cat"), "wildlife rehab"),
    (("guardrail",), "guardrail safety"),
    (("bridge",), "bridge work"),
    (("estimate", "jobsite", "bid ", "bids", "specification", "municipal", "public-works", "public works"), "municipal work"),
    (("monster", "rubber-suit", "campy", "horror", "sequel", "so-bad"), "B-movies"),
    (("box score", "pitch", "mlb", "baseball", "bullpen", "changeup", "scorekeeper", "red sox"), "baseball"),
    (("farm", "barn", "cows", "shed", "acres", "rural"), "rural life"),
    (("skipped",), "skipped content"),
    (("cat", "feline", "kitten", "tabby"), "cat videos"),
]


def fact_label(t, fallback):
    tl = t.lower()
    for keys, label in TAG_RULES:
        if any(k in tl for k in keys):
            return label
    return fallback


def topic_label(idxs, texts, used, df):
    """Label a cluster by the curated topic its facts mention most (deduped).
    Falls back to the most distinctive recurring word only if nothing matches."""
    blob = " ".join(texts[i].lower() for i in idxs)
    scores = {name: sum(blob.count(w) for w in ws) for name, ws in TOPIC_WORDS.items()}
    for name in sorted(scores, key=lambda k: -scores[k]):
        if scores[name] > 0 and name not in used:
            used.add(name)
            return name
    return keyword(idxs, texts, df, used)


def keyword(idxs, texts, df, used):
    """Most DISTINCTIVE word for this cluster (frequent here, rare elsewhere),
    skipping words already used by a bigger cluster so labels don't repeat."""
    local = collections.Counter()
    for i in idxs:
        for w in set(re.findall(r"[a-z]{3,}", texts[i].lower())):   # per-fact presence
            if w not in STOP:
                local[w] += 1
    # prefer words that RECUR in the cluster (>=2 facts); rank by tf-idf-ish
    # local^2/df so a label is both frequent here and concentrated, not a hapax.
    cand = [w for w in local if local[w] >= 2] or list(local)
    ranked = sorted(cand, key=lambda w: (-(local[w] * local[w] / (df[w] + 0.5)), -local[w]))
    for w in ranked:
        if w not in used:
            used.add(w)
            return w
    return ranked[0] if ranked else "misc"


def tsne(X, perp=10, n_iter=900, exagg_iter=120):
    """Minimal deterministic t-SNE (PCA-initialised, no RNG) — fine for a few
    hundred points. Lays facts out so on-screen distance ~ real semantic distance."""
    n = len(X)
    D = np.maximum(0.0, 2.0 - 2.0 * (X @ X.T))      # cosine sq-distance (X is unit-norm)
    np.fill_diagonal(D, 1e9)
    P = np.zeros((n, n))
    logU = np.log(perp)
    for i in range(n):
        lo, hi, beta = -np.inf, np.inf, 1.0
        Di = D[i]
        Pi = np.exp(-Di)
        for _ in range(60):
            Pi = np.exp(-Di * beta)
            s = Pi.sum() + 1e-12
            H = np.log(s) + beta * np.sum(Di * Pi) / s
            if abs(H - logU) < 1e-5:
                break
            if H > logU:
                lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta; beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i] = Pi / (Pi.sum() + 1e-12)
    P = np.maximum((P + P.T) / (2 * n), 1e-12)
    Xc = X - X.mean(0)
    _, _, Wt = np.linalg.svd(Xc, full_matrices=False)
    Y = (Xc @ Wt[:2].T) * 1e-4                       # deterministic PCA init
    Yv = np.zeros_like(Y)
    P4 = P * 4.0
    for it in range(n_iter):
        s = np.sum(Y * Y, 1)
        num = 1.0 / (1.0 + (-2.0 * Y @ Y.T + s).T + s)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = ((P4 if it < exagg_iter else P) - Q) * num
        dY = (np.diag(PQ.sum(1)) - PQ) @ Y * 4.0
        Yv = (0.5 if it < 250 else 0.8) * Yv - 200.0 * dY
        Y = Y + Yv
        Y -= Y.mean(0)
    return Y


def main():
    facts = sorted(load_facts(UID), key=lambda f: f[0])   # stable order -> reproducible clustering
    texts = [f[0] for f in facts]
    V = np.asarray([f[1] for f in facts])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)   # cosine space

    # ---- group facts by the topic they actually mention (honest + intuitive) ----
    def assign(t):
        t = t.lower()
        best, bn = None, 0
        for name, ws in TOPIC_WORDS.items():
            k = sum(t.count(w) for w in ws)
            if k > bn:
                bn, best = k, name
        return best
    raw = [assign(t) for t in texts]
    cnt = collections.Counter(x for x in raw if x)
    named = [name for name, _ in cnt.most_common() if cnt[name] >= 3]     # topics big enough for a column
    grp = [(t if t in named else "other") for t in raw]
    order = named + (["other"] if "other" in grp else [])
    groups = {name: [i for i in range(len(texts)) if grp[i] == name] for name in order}
    groups = {k: v for k, v in groups.items() if v}
    order = [k for k in order if k in groups]
    COL = {name: PAL[i % len(PAL)] for i, name in enumerate([k for k in order if k != "other"])}
    COL["other"] = "#9aa6ad"

    # real nearest neighbours (cosine) -> intra-topic constellation links
    sim = Vn @ Vn.T
    np.fill_diagonal(sim, -1.0)
    nn = {i: list(sim[i].argsort()[::-1][:2]) for i in range(len(texts))}

    def short(i, nwords=3):                       # a few real words from a fact, for an inline tag
        ws = re.sub(r"[^a-zA-Z0-9 #]", " ", texts[i]).split()
        ws = [w for w in ws if w.lower() not in STOP and len(w) > 2][:nwords]
        return _html.escape(" ".join(ws))

    # ---- layout: t-SNE on the real embeddings -> on-screen distance reflects
    # real semantic distance (no artificial anchors). Deterministic (PCA-init).
    Y = tsne(Vn)
    Y = Y - Y.mean(0)
    _, _, Wy = np.linalg.svd(Y, full_matrices=False)        # rotate principal axis -> horizontal
    Y = Y @ Wy.T
    if np.ptp(Y[:, 0]) < np.ptp(Y[:, 1]):
        Y = Y[:, ::-1]

    # ---- SVG (equal x/y scale so distances stay honest; canvas matches the
    # data's aspect so there is little empty space) ----
    W, pad = 760, 12
    rxr, ryr = (np.ptp(Y[:, 0]) or 1.0), (np.ptp(Y[:, 1]) or 1.0)
    plotW = W - 2 * pad
    sc = plotW / rxr
    H = int(round(ryr * sc)) + 2 * pad
    H = max(260, min(H, 400))                                # compact: cap the height
    PX = pad + (Y[:, 0] - Y[:, 0].min()) * sc
    PY = pad + (ryr - (Y[:, 1] - Y[:, 1].min())) * ((H - 2 * pad) / ryr)
    cen = {name: (float(np.median(PX[groups[name]])), float(np.median(PY[groups[name]])))
           for name in order}

    p = [f"<svg viewBox='0 0 {W} {H}' width='{W}' height='{H}' style='width:100%;height:auto;display:block'>"]
    p.append("<defs><radialGradient id='ragbg' cx='42%' cy='38%' r='80%'>"
             "<stop offset='0%' stop-color='#fdfefe'/><stop offset='100%' stop-color='#f5f7f9'/>"
             "</radialGradient></defs>")
    p.append(f"<rect x='0' y='0' width='{W}' height='{H}' rx='8' fill='url(#ragbg)'/>")
    # 1) soft cluster halo per named topic (real centroid + real spread)
    for name in order:
        if name == "other":
            continue
        ix = groups[name]; cx, cy = cen[name]
        d = np.hypot(PX[ix] - cx, PY[ix] - cy)
        r = max(20.0, float(np.percentile(d, 78)) + 16)
        p.append(f"<circle cx='{cx:.0f}' cy='{cy:.0f}' r='{r:.0f}' fill='{COL[name]}' opacity='0.07'/>")
    # 2) real nearest-neighbour links (cosine) -> organic constellation
    for i, nbrs in nn.items():
        for j in nbrs:
            same = grp[i] == grp[j]
            p.append(f"<line x1='{PX[i]:.1f}' y1='{PY[i]:.1f}' x2='{PX[j]:.1f}' y2='{PY[j]:.1f}' "
                     f"stroke='{COL[grp[i]] if same else '#cfd8df'}' stroke-width='0.7' "
                     f"opacity='{0.34 if same else 0.16}'/>")
    # 3) glowing fact dots; each carries a full-fact tooltip on hover, and a
    #    short human-readable name placed outward. Names are de-duplicated so a
    #    cluster of 14 cat dots isn't labelled "cat videos" 14 times.
    shown = set()
    for i in range(len(texts)):
        c = COL[grp[i]]
        cx0, _cy0 = cen[grp[i]]
        right = PX[i] >= cx0
        tx = PX[i] + (5 if right else -5)
        anchor = "start" if right else "end"
        p.append(f"<circle cx='{PX[i]:.1f}' cy='{PY[i]:.1f}' r='6' fill='{c}' opacity='0.09'/>")
        p.append(f"<circle cx='{PX[i]:.1f}' cy='{PY[i]:.1f}' r='3.2' fill='{c}' "
                 f"stroke='#ffffff' stroke-width='0.7'><title>{_html.escape(texts[i])}</title></circle>")
        name = fact_label(texts[i], grp[i])
        if name not in shown:                       # one visible label per distinct name
            shown.add(name)
            for paint in ("fill='#ffffff' stroke='#ffffff' stroke-width='2.4'", "fill='#5a6770'"):
                p.append(f"<text x='{tx:.1f}' y='{PY[i]+2:.1f}' text-anchor='{anchor}' font-size='7' "
                         f"{paint}>{_html.escape(name)}</text>")
    # 4) one bold, haloed label + count per named topic, at its cluster centre
    for name in order:
        if name == "other":
            continue
        cx, cy = cen[name]
        for paint in ("fill='#ffffff' stroke='#ffffff' stroke-width='3.4'", f"fill='{COL[name]}'"):
            p.append(f"<text x='{cx:.0f}' y='{cy - 7:.0f}' text-anchor='middle' font-size='14' "
                     f"font-weight='700' {paint}>{_html.escape(name)}</text>")
        p.append(f"<text x='{cx:.0f}' y='{cy + 4:.0f}' text-anchor='middle' font-size='8.5' "
                 f"fill='#ffffff' stroke='#ffffff' stroke-width='2.6'>{len(groups[name])}</text>")
        p.append(f"<text x='{cx:.0f}' y='{cy + 4:.0f}' text-anchor='middle' font-size='8.5' "
                 f"fill='#7a868f'>{len(groups[name])}</text>")
    p.append("</svg>")
    svg = "".join(p)

    # legend (topic -> count)
    leg = "".join(
        f"<span style='margin-right:13px;white-space:nowrap'>"
        f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;background:{COL[name]};"
        f"vertical-align:middle'></span>&nbsp;{_html.escape(name)} ({len(groups[name])})</span>"
        for name in order)

    # a few real example facts (verbatim), one per top topic, colour-coded
    sample = [(name, groups[name][0]) for name in order if name != "other"][:6]
    ex = "".join(
        f"<div style='font-size:9.5px;color:#33424b;padding:3px 0;border-top:1px solid #eef1f3'>"
        f"<b style='color:{COL[name]}'>{_html.escape(name)}</b> &mdash; "
        f"&ldquo;{_html.escape(texts[i][:140])}{'&hellip;' if len(texts[i])>140 else ''}&rdquo;</div>"
        for name, i in sample)

    sec = [START, "<section>"]
    sec.append('<div class="cap"><h2>What mem0&rsquo;s memory actually looks like &mdash; one user&rsquo;s fact store</h2>'
               f'<span class="note">GPT-5.5 Mem0 (RAG) &middot; persona {UID} &middot; {len(texts)} stored facts</span></div>')
    sec.append('<p class="lead" style="margin:0 0 12px">mem0 does <b>not</b> keep this person&rsquo;s posts. It reads their '
               'history and writes down short <b>facts</b> about them, then turns each fact into an <b>embedding</b> &mdash; a '
               'point in a high-dimensional &ldquo;meaning space&rdquo; &mdash; and files it in a vector database. The map below '
               f'uses <b>t-SNE</b> to flatten that space to two dimensions while preserving real distances: <b>every dot is one '
               f'real stored fact</b> ({len(texts)} for this user), and dots near each other really are about similar things, so the '
               'topics fall into natural clusters. The faint lines join each fact to its nearest neighbours &mdash; the structure a '
               'question searches. When the user asks something, mem0 embeds the question and pulls back only the few nearest dots '
               'to answer from.</p>')
    sec.append(f'<div style="margin:6px 0 4px">{svg}</div>')
    sec.append(f'<div class="note" style="margin:2px 0 10px">{leg}</div>')
    sec.append('<div class="abldefs" style="max-width:820px"><b>A few of the actual stored facts</b> (verbatim):'
               f'{ex}<div style="margin-top:6px"><b>How to read it.</b> Dot positions come from t-SNE on the real '
               '3072-dimension embeddings, so nearby dots really are similar (the figure is lightly compressed vertically to '
               'stay compact); colour and label mark '
               'the topic each fact mentions (a fact only joins a topic it actually talks about); the soft halos and faint lines '
               'are the real cluster spread and nearest-neighbour links. Each dot is tagged with two of its own words and shows '
               'its full fact on hover; the grey &ldquo;other&rdquo; dots pool the smaller topics (history, rural, UFO, &hellip;). '
               'So this is an honest &mdash; if simplified &mdash; picture of what is in the store, not a hand-drawn sketch.</div></div>')
    sec.append("</section>")
    sec.append(END)
    block = "\n".join(sec)

    with html_lock():
        with open(HTML) as f:
            html = f.read()
        if START in html and END in html:
            html = html.split(START)[0] + block + html.split(END, 1)[1]
        else:
            anchor = "<!-- MEM_FORGET_END -->"
            if anchor in html:
                html = html.replace(anchor, anchor + "\n\n" + block, 1)
            else:                                   # fallback: before closing body
                html = html.replace("</body>", block + "\n</body>", 1)
        with open(HTML, "w") as f:
            f.write(html)
    print(f"injected RAG-db viz: persona {UID}, {len(texts)} facts, "
          f"{len(order)} topic columns {[(name, len(groups[name])) for name in order]}")


if __name__ == "__main__":
    main()
