#!/usr/bin/env python3
"""Diversity of the persona set + how user preferences evolve over time.

diversity(): distributions over the processed profiles (demographics, hidden-
persona types, sensitive-life-event topics, mobility, MBTI, interest breadth).

evolution(): from the timestamped app events — per relative day of the
observation window, how many preference engagements are NEW topics (emergence)
vs. RECURRING (reinforcement), plus short/long-term split and stance-flip count.
Also a per-persona example: top categories' daily activity.
"""
import json, glob, collections, statistics, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPS = ["instagram", "facebook", "threads", "chatbot", "ai_studio"]
PROFILES = sorted(glob.glob(f"{ROOT}/backend/*/profile.json"))

# structural / generic words to drop from preference + humor word clouds
_STOP = set("""the a an and or to of with for is are be that this his her their them they not very more about
treats uses talks through logic when into like as on in it dry interested follows supports including featuring
such around from related posts style framed generic clips centered focused engages identifies avoids dislikes
content social media enjoys enjoy likes prefers values especially strongly own using over most their there
based real life when than usually rather without voice little but also some who while across both via per""".split())


# never surface these in any diversity display (per request)
_STOP |= {"christian", "faith", "church", "jesus", "biblical", "scripture", "christianity", "prayer", "religious"}
# eval-scaffold words to drop from the user-query cloud
_STOP |= {"system", "prompt", "assistant", "instructions", "please", "respond", "reply", "given", "below", "task"}
# substring blocklist for whole (often multi-word) terms — stances, registers,
# hashtags. Catches e.g. "faith-and-family warm", "sunday-faith plainspoken",
# "faithinaction" that a single-token stoplist misses.
_BLOCK_SUBSTR = ("christian", "faith", "church", "jesus", "biblical",
                 "scripture", "christianity", "prayer", "gospel", "worship")


def _blocked(term):
    t = (term or "").lower()
    return any(b in t for b in _BLOCK_SUBSTR)


def _drop_blocked(counter):
    return collections.Counter({k: v for k, v in counter.items() if not _blocked(k)})


def _words(s):
    return [w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 3 and w not in _STOP]


def tasks_and_queries():
    """Task-type histogram (Accuracy-table tasks) + a user-query word cloud, from
    backend/{uid}/test.json across all personas."""
    tasks = collections.Counter()
    qwords = collections.Counter()
    for f in PROFILES:
        uid = Path(f).parent.name
        tj = ROOT / "backend" / uid / "test.json"
        if not tj.exists():
            continue
        for it in json.load(open(tj)):
            tt = it.get("task_type") or it.get("task")
            if tt:
                tasks[tt] += 1
            full = it.get("instance_full") or {}
            q = it.get("query_text") or it.get("user_query") or full.get("query_text") or full.get("user_query") or ""
            for w in _words(q):
                qwords[w] += 1
    return {"tasks": tasks, "qwords": qwords}


def _profiles():
    return [(Path(f).parent.name, json.load(open(f))) for f in PROFILES]


def diversity():
    P = [p for _, p in _profiles()]
    n = len(P)
    out = {"n": n}

    def cnt(fn):
        return collections.Counter(filter(None, (fn(p) for p in P)))

    # sexual-orientation / gender grouping
    out["gender"] = cnt(lambda p: p.get("gender"))
    out["ethnicity"] = cnt(lambda p: p.get("race_ethnicity"))
    out["mobility"] = cnt(lambda p: p.get("mobility_class"))
    out["mbti"] = cnt(lambda p: (p.get("mbti") or {}).get("type"))
    out["archetype"] = cnt(lambda p: (p.get("ai_studio_persona") or {}).get("persona_archetype"))
    out["n_careers"] = len(set(p.get("career") for p in P))
    out["n_education"] = len(set(p.get("education") for p in P))
    # hidden-persona types
    ht = collections.Counter()
    for p in P:
        for hp in (p.get("hidden_personas") or []):
            ht[hp.get("type")] += 1
    out["hidden_types"] = ht
    # sensitive-life-event topics
    sle = collections.Counter()
    for p in P:
        for hp in (p.get("hidden_personas") or []):
            if hp.get("type") == "sensitive_life_event":
                for ev in (hp.get("events") or []):
                    t = ev.get("topic") or (ev.get("label_fragment") or "?")
                    sle[t] += 1
    out["sensitive_topics"] = sle
    # interest breadth: distinct preference categories per persona
    cats_per = []
    for p in P:
        c = set()
        # profile preferences are strings; categories live on app-event prefs -> use hidden persona evidence_hashtags as a proxy of breadth
        for hp in (p.get("hidden_personas") or []):
            c.update(hp.get("evidence_hashtags") or [])
        cats_per.append(len(c))
    out["breadth_median"] = statistics.median(cats_per) if cats_per else 0
    out["n_prefs"] = [len(p.get("preferences") or []) for p in P]

    # --- user voice diversity ---
    out["voice_cap"] = cnt(lambda p: (p.get("user_voice") or {}).get("default_capitalization"))
    fb = collections.Counter()
    palette = []
    emoji = collections.Counter()
    for p in P:
        uv = p.get("user_voice") or {}
        f = uv.get("formality_baseline")
        if isinstance(f, (int, float)):
            fb["formal (≥0.6)" if f >= 0.6 else ("casual (≤0.35)" if f <= 0.35 else "mid (0.35–0.6)")] += 1
        emoji[uv.get("emoji_intensity_default")] += 1
        palette.append(len(uv.get("emoji_palette") or []))
    out["voice_formality"] = fb
    out["voice_emoji"] = emoji
    out["voice_palette_med"] = statistics.median(palette) if palette else 0
    out["voice_humor_distinct"] = len(set((p.get("user_voice") or {}).get("humor_tone") for p in P))
    # emoji palette (actual emojis) + humor-tone words for word clouds
    emo = collections.Counter(); hum = collections.Counter()
    for p in P:
        uv = p.get("user_voice") or {}
        for em in (uv.get("emoji_palette") or []):
            emo[em] += 1
        for w in _words(uv.get("humor_tone")):
            hum[w] += 1
    out["emoji_palette"] = emo
    out["humor_words"] = hum
    out["n_register"] = len(set((p.get("user_voice") or {}).get("natural_register") for p in P))

    # --- the FOUR voice layers (identity spine / idiolect / repertoire / holdovers) ---
    stances = collections.Counter(); registers = collections.Counter()
    phrases_avoid = collections.Counter(); sig = collections.Counter()
    sent_shape = collections.Counter(); hedge = collections.Counter()
    idiolect_w = collections.Counter(); concerns_phr = collections.Counter()
    for p in P:
        uv = p.get("user_voice") or {}
        rep = uv.get("repertoire") or {}
        for s in (rep.get("stances") or []):
            stances[str(s).strip()] += 1
        for r in (rep.get("registers") or []):
            registers[str(r).strip()] += 1
        for ph in (uv.get("phrases_to_avoid") or []):
            phrases_avoid[str(ph).strip().strip('“”"')] += 1
        spine = uv.get("identity_spine") or {}
        for c in (spine.get("signature_concerns") or []):
            for w in _words(c):
                sig[w] += 1
        idi = uv.get("idiolect") or {}
        syn = idi.get("syntactic_preferences") or {}
        if syn.get("sentence_length_shape"):
            sent_shape[str(syn["sentence_length_shape"]).replace("_", " ")] += 1
        if idi.get("hedge_booster_ratio"):
            hedge[str(idi["hedge_booster_ratio"])] += 1
        for w in _words(idi.get("function_word_profile")):
            idiolect_w[w] += 1
    out["voice_stances"] = stances
    out["voice_registers"] = registers
    out["voice_phrases_avoid"] = phrases_avoid
    out["voice_sig_concerns"] = sig
    out["voice_idiolect_words"] = idiolect_w
    out["voice_sent_shape"] = sent_shape
    out["voice_hedge"] = hedge
    out["n_stances"] = len(stances)
    out["n_registers_distinct"] = len(registers)
    return out


def behavior():
    """Events per app, interaction-action distribution, and per-persona counts of
    activities (events) and preference-engagements across all personas."""
    apps = collections.Counter()
    actions = collections.Counter()
    pos_cat = collections.Counter()      # positive-polarity preference categories
    neg_cat = collections.Counter()      # negative-polarity preference categories
    pref_signal = collections.Counter()  # raw explicit/implicit positive/negative event types
    pos_word = collections.Counter()     # positive preference TERMS (tokenized persona_item)
    neg_word = collections.Counter()     # negative preference TERMS
    ctype = collections.Counter()        # content type
    htag = collections.Counter()         # content hashtags
    dm = self_auth = total = 0
    acts_per = []          # activities (events) per persona
    prefeng_per = []       # preference-engagement events per persona
    for f in PROFILES:
        uid = Path(f).parent.name
        n_ev = n_prefeng = 0
        for a in APPS:
            fp = ROOT / "backend" / uid / f"{a}.json"
            if not fp.exists():
                continue
            for e in json.load(open(fp)):
                apps[a] += 1; total += 1; n_ev += 1
                act = (e.get("interaction_format") or {}).get("action")
                if act:
                    actions[act] += 1
                ct = e.get("content_type") or "feed-skim (no body)"
                ctype[ct] += 1
                for ht in (e.get("source_hashtags") or []):
                    htag[ht.lower().lstrip("#")] += 1
                itype = e.get("source_interaction_type") or "unknown"
                pref_signal[itype] += 1
                neg = "negative" in itype
                prefs = e.get("preferences") or []
                if prefs:
                    n_prefeng += 1
                for pr in prefs:
                    cat = pr.get("category") or "?"
                    (neg_cat if neg else pos_cat)[cat] += 1
                    wc = neg_word if neg else pos_word
                    for w in _words(pr.get("persona_item")):
                        wc[w] += 1
                dm += int(bool(e.get("is_dm")))
                self_auth += int(bool(e.get("is_self_authored")))
        acts_per.append(n_ev); prefeng_per.append(n_prefeng)
    return {"apps": apps, "actions": actions, "n_actions": len(actions),
            "pos_cat": pos_cat, "neg_cat": neg_cat, "pos_word": pos_word, "neg_word": neg_word,
            "pref_signal": pref_signal, "ctype": ctype, "htag": htag,
            "dm": dm, "self_auth": self_auth, "total": total,
            "acts_per": acts_per, "prefeng_per": prefeng_per}


def _cat_series(uid):
    """{category: [9 daily counts]} for one persona's preference engagements."""
    ev = _events(uid)
    if not ev:
        return {}
    t0 = ev[0][0]
    series = collections.defaultdict(lambda: [0] * 9)
    for ts, e in ev:
        d = min(8, int((ts - t0) // 86400))
        for pr in (e.get("preferences") or []):
            series[pr.get("category") or "?"][d] += 1
    return series


def focus_shift(min_total=14, topn=6):
    """Personas with a clear attention HANDOFF: a category that fades (early-heavy)
    while another rises (late-heavy). Returns the top-N (by handoff clarity), each
    with the fading category, the rising category, and a steady anchor."""
    cands = []
    for f in PROFILES:
        uid = Path(f).parent.name
        s = _cat_series(uid)
        dims, emgs, steady = [], [], []
        for cat, ser in s.items():
            tot = sum(ser)
            if tot < min_total or cat == "?":
                continue
            early, late = sum(ser[:4]), sum(ser[5:])
            spike = max(ser) / tot                      # exclude single-day bursts
            if spike >= 0.5:
                continue
            if early >= 2 * max(1, late) and early >= 8:
                dims.append((early - late, cat, ser))
            elif late >= 2 * max(1, early) and late >= 8:
                emgs.append((late - early, cat, ser))
            elif min(ser) >= 1:
                steady.append((tot, cat, ser))
        if dims and emgs:
            d0 = max(dims); e0 = max(emgs)
            if _blocked(d0[1]) or _blocked(e0[1]):
                continue
            # reward a clean, comparable-magnitude handoff (both lines visible)
            score = min(sum(d0[2]), sum(e0[2])) + 0.3 * (d0[0] + e0[0])
            anchor = next((a for a in sorted(steady, reverse=True) if not _blocked(a[1])), None)
            cands.append((score, uid, d0, e0, anchor))
    cands.sort(key=lambda x: -x[0])
    out = []
    for _, uid, d0, e0, anchor in cands[:topn]:
        item = {"uid": uid, "fading": (d0[1], d0[2]), "rising": (e0[1], e0[2])}
        if anchor:
            item["steady"] = (anchor[1], anchor[2])
        out.append(item)
    return out


def trajectories(min_total=14, min_days=3, per_pattern=10):
    """Classify each (persona, category) preference trajectory over the ~9-day
    window into reinforced / emerging / diminishing / bursty, and return pattern
    counts + representative examples (persona, category, 9-day series). Examples
    are deduped by category so the gallery surfaces distinct sub-aspects."""
    pat_count = collections.Counter()
    examples = collections.defaultdict(list)   # pattern -> [(score, uid, cat, series)]
    for f in PROFILES:
        uid = Path(f).parent.name
        ev = _events(uid)
        if not ev:
            continue
        t0 = ev[0][0]
        series = collections.defaultdict(lambda: [0] * 9)
        for ts, e in ev:
            d = min(8, int((ts - t0) // 86400))
            for pr in (e.get("preferences") or []):
                series[pr.get("category") or "?"][d] += 1
        for cat, s in series.items():
            tot = sum(s)
            active = sum(1 for x in s if x)
            if tot < min_total or active < min_days or cat == "?":
                continue
            first = sum(s[:4]); last = sum(s[5:]); peak = max(s); pk = s.index(peak)
            if peak / tot >= 0.42 and 1 <= pk <= 7 and active <= 5:
                pat = "bursty"; score = peak / tot
            elif last - first >= max(4, 0.30 * tot):
                pat = "emerging"; score = (last - first) / tot
            elif first - last >= max(4, 0.30 * tot):
                pat = "diminishing"; score = (first - last) / tot
            else:
                pat = "reinforced"; score = min(s) * active  # steady + broad
            pat_count[pat] += 1
            if not _blocked(cat):
                examples[pat].append((score + tot * 1e-4, uid, cat, s))
    # dedupe by category (keep the clearest instance of each distinct sub-aspect)
    ex = {}
    for pat, v in examples.items():
        seen, picked = set(), []
        for item in sorted(v, reverse=True):
            cat = item[2]
            if cat in seen:
                continue
            seen.add(cat); picked.append(item)
            if len(picked) >= per_pattern:
                break
        ex[pat] = picked
    return {"counts": pat_count, "examples": ex}


def _events(uid):
    ev = []
    for a in APPS:
        fp = ROOT / "backend" / uid / f"{a}.json"
        if not fp.exists():
            continue
        for e in json.load(open(fp)):
            ts = e.get("source_timestamp")
            prefs = e.get("preferences") or []
            if ts and prefs:
                ev.append((ts, e))
    ev.sort(key=lambda x: x[0])
    return ev


def evolution():
    """Per relative day: new-topic emergence vs recurrence (across all personas),
    short/long-term split, stance flips, and an example persona stream."""
    emerge = collections.Counter()      # rel_day -> count of (persona, category) first appearances
    recur = collections.Counter()
    horizon = collections.Counter()
    flips = 0
    npref = 0
    for f in PROFILES:
        uid = Path(f).parent.name
        ev = _events(uid)
        if not ev:
            continue
        t0 = ev[0][0]
        seen = set()
        for ts, e in ev:
            d = int((ts - t0) // 86400)
            if d > 8:
                d = 8
            for pr in (e.get("preferences") or []):
                cat = pr.get("category") or pr.get("persona_item", "")[:30]
                npref += 1
                horizon[pr.get("time_horizon") or "long_term"] += 1
                uh = pr.get("update_history") or []
                if any((isinstance(u, dict) and u.get("type") == "contradicted") for u in uh):
                    flips += 1
                if cat in seen:
                    recur[d] += 1
                else:
                    seen.add(cat); emerge[d] += 1
    # example persona stream: top categories by day for one persona
    ex_uid = "1"
    ev = _events(ex_uid)
    t0 = ev[0][0] if ev else 0
    bycat = collections.defaultdict(lambda: collections.Counter())  # cat -> day -> count
    catetot = collections.Counter()
    for ts, e in ev:
        d = min(8, int((ts - t0) // 86400))
        for pr in (e.get("preferences") or []):
            cat = pr.get("category") or "?"
            bycat[cat][d] += 1
            catetot[cat] += 1
    top_cats = [c for c, _ in catetot.most_common(6)]
    example = {"uid": ex_uid, "cats": top_cats,
               "series": {c: {d: bycat[c][d] for d in range(9)} for c in top_cats}}
    return {"emerge": emerge, "recur": recur, "horizon": horizon,
            "flips": flips, "npref": npref, "example": example}


if __name__ == "__main__":
    d = diversity()
    print(f"=== DIVERSITY (n={d['n']} personas) ===")
    print(f"genders: {len(d['gender'])}  ethnicities: {len(d['ethnicity'])}  careers: {d['n_careers']}  "
          f"education: {d['n_education']}  MBTI: {len(d['mbti'])}  mobility: {dict(d['mobility'])}")
    print("hidden types:", dict(d["hidden_types"].most_common()))
    print("sensitive topics:", dict(d["sensitive_topics"].most_common()))
    print("archetype key found:", len(d["archetype"]), "distinct")
    e = evolution()
    print(f"\n=== EVOLUTION (n_pref_engagements={e['npref']}, stance flips={e['flips']}) ===")
    print("horizon:", dict(e["horizon"]))
    print("emerge by day:", [e["emerge"][i] for i in range(9)])
    print("recur   by day:", [e["recur"][i] for i in range(9)])
    print("example persona top cats:", e["example"]["cats"])
