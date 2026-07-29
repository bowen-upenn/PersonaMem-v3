#!/usr/bin/env python3
"""Reduce identity-level Christian prevalence to a fixed keeper set.

Policy (2026-07-24): at most 5 of the 100 published personas carry a Christian
identity (Christian canonical preferences). The real-world seeds skew heavily
Christian (2% of 4.2M rows, 973/998 users touched), so the pipeline faithfully
produced 37 identity-level Christian personas; this tool converts all but the
keepers to varied secular themes while preserving narrative coherence.

What converts, per non-keeper persona with Christian canonical preferences:
  - every Christian canonical preference text (profile, event attachments,
    update histories, test.json ground-truth copies) -> a secular equivalent
    aligned with the persona's other interests (themes rotate: community
    volunteering, mindfulness & nature, gratitude & positivity, soul & music
    heritage, philosophy & stoicism, family heritage & traditions)
  - prose on events that carry those preferences or are positive Christian
    engagements (titles, captions, media descriptions, transcripts)
  - Christian hashtags on those events
  - golden/inferior/GT texts in test.json still faith-flavored after the
    string mapping get a minimal LLM rewrite
What stays: implicit-negative scroll-pasts of faith content with no attached
preference (content-in-the-world, not identity), and the 5 keepers.

Usage:
    python scripts/reduce_christian_prevalence.py                # dry-run scope
    python scripts/reduce_christian_prevalence.py --apply        # convert
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

KEEPERS = {"14", "105", "461", "9", "655"}
EVAL = set('1 2 3 5 6 8 9 10 13 14 26 105 115 209 229 282 461 655 760 835'.split())
DELETE_MAX = 10   # engagement-only personas with <= this many flagged events: delete them
CH = re.compile(r'christ|jesus|bible|gospel|church|worship|devotional|psalm|'
                r'scripture|sermon|faith|prayer|pray\b|praying|blessed|blessing|'
                r'\bgod\b|\blord\b|hallelujah|amen\b|salvation|holy\b', re.I)
CH_STRONG = re.compile(r'christ|jesus|bible|gospel|church|worship|devotional|'
                       r'psalm|scripture|sermon', re.I)
APPS = ["instagram", "facebook", "threads", "chatbot", "ai_studio"]

THEMES = [
    ("community volunteering and local mutual aid",
     ["#communitylove", "#volunteerwork", "#givingback", "#neighborshelpingneighbors"]),
    ("mindfulness, quiet mornings, and time in nature",
     ["#mindfulmoments", "#quietmornings", "#naturewalks", "#stillness"]),
    ("gratitude journaling and everyday positivity",
     ["#gratitudedaily", "#smalljoys", "#brighterdays", "#thankfulheart"]),
    ("classic soul music and musical heritage",
     ["#classicsoul", "#soulmusic", "#musicheritage", "#timelesstunes"]),
    ("practical philosophy and reflective reading",
     ["#dailyreflection", "#lifelessons", "#goodreads", "#perspective"]),
    ("family heritage, recipes, and traditions",
     ["#familytraditions", "#heritagecooking", "#generations", "#homestories"]),
]


def _mini_client(rate_limit=50):
    from query_llm import QueryLLM
    return QueryLLM({"models": {"llm_model": "gpt-5.4-mini"}}, rate_limit_per_min=rate_limit)


def _ask(client, prompt):
    from data_preparation import utils
    resp = client.query_llm(prompt)
    return utils.extract_json_from_response(resp)


def flagged_event_oids(uid):
    """(app, source_object_id) of pref-attached or positive Christian events."""
    out = []
    for app in APPS:
        for e in json.load(open(REPO_ROOT / f'backend/{uid}/{app}.json')):
            plist = [p.get('persona_item') or '' for p in (e.get('preferences') or [])]
            positive = 'positive' in (e.get('source_interaction_type') or '')
            if any(CH_STRONG.search(p) for p in plist) or \
               (positive and CH_STRONG.search(json.dumps(e))):
                out.append((app, e.get('source_object_id')))
    return out


def delete_events_and_prefs(uid):
    """Delete the few flagged events + drop Christian pref lines from profile."""
    removed = 0
    targets = set(flagged_event_oids(uid))
    for app in APPS:
        fp = REPO_ROOT / f'backend/{uid}/{app}.json'
        evs = json.load(open(fp))
        keep = [e for e in evs if (app, e.get('source_object_id')) not in targets]
        removed += len(evs) - len(keep)
        if len(keep) != len(evs):
            json.dump(keep, open(fp, 'w'), ensure_ascii=False, indent=2)
    fp = REPO_ROOT / f'backend/{uid}/profile.json'
    prof = json.load(open(fp))
    prefs = prof.get('preferences') or []
    kept = [x for x in prefs
            if not CH_STRONG.search(x if isinstance(x, str) else json.dumps(x))]
    if len(kept) != len(prefs):
        prof['preferences'] = kept
        json.dump(prof, open(fp, 'w'), ensure_ascii=False, indent=2)
    return removed, len(prefs) - len(kept)


def collect_scope(uid):
    """Christian canonical pref texts + flagged event prose + hashtags."""
    prefs = set()
    prof = json.load(open(REPO_ROOT / f'backend/{uid}/profile.json'))
    for x in (prof.get('preferences') or []):
        t = x if isinstance(x, str) else json.dumps(x)
        if CH_STRONG.search(t):
            # profile prefs carry a "HH:MM, date : text" prefix — strip to text
            core = t.split(' : ', 1)[-1].strip().strip('"')
            prefs.add(core)
    prose = set()
    tags = set()
    top_interests = []
    for app in APPS:
        for e in json.load(open(REPO_ROOT / f'backend/{uid}/{app}.json')):
            plist = [p.get('persona_item') or '' for p in (e.get('preferences') or [])]
            christian_pref_event = any(CH_STRONG.search(p) for p in plist)
            for p in plist:
                if CH_STRONG.search(p):
                    prefs.add(p)
                elif p and len(top_interests) < 12 and p not in top_interests:
                    top_interests.append(p)
            positive = 'positive' in (e.get('source_interaction_type') or '')
            if christian_pref_event or (positive and CH_STRONG.search(json.dumps(e))):
                c = e.get('content') or {}
                for k in ('title', 'caption', 'overall_description', 'audio_transcript'):
                    v = c.get(k) or ''
                    if CH.search(v):
                        prose.add(v)
                for t in (e.get('source_hashtags') or []):
                    if CH.search(t):
                        tags.add(t)
    return sorted(prefs), sorted(prose, key=len, reverse=True), sorted(tags), top_interests


def build_mapping(client, uid, theme, prefs, prose, tags, interests):
    """One LLM call for prefs+tags mapping; batched calls for prose."""
    mapping = {}
    if prefs or tags:
        prompt = f"""You are rewriting a synthetic social-media persona's preferences away from religious themes toward: {theme[0]}.
The persona's OTHER interests (keep consistent with these): {json.dumps(interests[:8])}
Rewrite each PREFERENCE below into a natural secular preference on the new theme (same sentence style, similar length, no religious words). Rewrite each HASHTAG into a natural secular hashtag on the new theme (reuse from {json.dumps(theme[1])} plus natural variants; match the original's CamelCase/lowercase style).
Return JSON with the SAME COUNTS and SAME ORDER as the inputs:
{{"preferences": ["<new pref 1>", ...], "hashtags": ["<new tag 1>", ...]}}
PREFERENCES ({len(prefs)}): {json.dumps(prefs)}
HASHTAGS ({len(tags)}): {json.dumps(tags)}"""
        out = _ask(client, prompt) or {}
        for old, new in zip(prefs, out.get('preferences') or []):
            if isinstance(new, dict):
                new = new.get('new') or new.get('text') or (next(iter(new.values()), '') if new else '')
            if isinstance(new, str) and new and not CH.search(new):
                mapping[old] = new
        for old, new in zip(tags, out.get('hashtags') or []):
            if isinstance(new, dict):
                new = new.get('new') or new.get('text') or (next(iter(new.values()), '') if new else '')
            if isinstance(new, str) and new and not CH.search(new):
                mapping[old] = new if new.startswith('#') or not old.startswith('#') else '#' + new.lstrip('#')
    # batched prose rewrites (8 per call)
    for i in range(0, len(prose), 8):
        batch = prose[i:i + 8]
        prompt = f"""Rewrite each text below to remove ALL religious content, retargeting it to: {theme[0]}. Keep length, tone, format and any non-religious details; the result must read as natural social-media content of the same kind (caption stays a caption, transcript a transcript). No religious words.
Return JSON: {{"rewrites": ["<new text 1>", ...]}} in the same order.
TEXTS: {json.dumps(batch)}"""
        out = _ask(client, prompt) or {}
        rew = out.get('rewrites') or []
        for old, new in zip(batch, rew):
            if isinstance(new, str) and new and not CH_STRONG.search(new):
                mapping[old] = new
    return mapping


def apply_mapping(uid, mapping):
    """Exact-string replacement across every file (JSON-escape aware)."""
    total = 0
    files = APPS + ['calendar', 'profile', 'test']
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    for f in files:
        fp = REPO_ROOT / f'backend/{uid}/{f}.json'
        if not fp.exists():
            continue
        raw = fp.read_text()
        n_f = 0
        for old, new in items:
            o = json.dumps(old, ensure_ascii=False)[1:-1]
            n = json.dumps(new, ensure_ascii=False)[1:-1]
            if o in raw:
                n_f += raw.count(o)
                raw = raw.replace(o, n)
        if n_f:
            json.loads(raw)
            fp.write_text(raw)
            total += n_f
    return total


def remaining_christian_prefs(uid):
    prof = json.load(open(REPO_ROOT / f'backend/{uid}/profile.json'))
    n = sum(1 for x in (prof.get('preferences') or [])
            if CH_STRONG.search(x if isinstance(x, str) else json.dumps(x)))
    for app in APPS:
        for e in json.load(open(REPO_ROOT / f'backend/{uid}/{app}.json')):
            for p in (e.get('preferences') or []):
                if CH_STRONG.search(p.get('persona_item') or ''):
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--rate_limit', type=int, default=50)
    args = ap.parse_args()

    uids = sorted([d for d in os.listdir(REPO_ROOT / 'backend')
                   if re.fullmatch(r'[0-9]+', d)], key=int)
    convert = []
    for u in uids:
        if u in KEEPERS:
            continue
        prefs, prose, tags, interests = collect_scope(u)
        if prefs:
            convert.append((u, prefs, prose, tags, interests))
    print(f'personas to convert: {len(convert)} (keepers: {sorted(KEEPERS, key=int)})')
    print(f'total: {sum(len(c[1]) for c in convert)} prefs, '
          f'{sum(len(c[2]) for c in convert)} prose strings, '
          f'{sum(len(c[3]) for c in convert)} hashtags')
    if not args.apply:
        for u, prefs, prose, tags, _ in convert:
            print(f'  p{u:<4} prefs={len(prefs):<3} prose={len(prose):<4} tags={len(tags)}')
        return

    client = _mini_client(args.rate_limit)
    for i, (u, prefs, prose, tags, interests) in enumerate(convert):
        n_flagged = len(flagged_event_oids(u))
        if u not in EVAL and n_flagged <= DELETE_MAX:
            n_ev, n_pref = delete_events_and_prefs(u)
            left = remaining_christian_prefs(u)
            print(f'[{i+1}/{len(convert)}] p{u}: DELETE mode -- removed {n_ev} events, '
                  f'{n_pref} profile prefs; remaining_christian_prefs={left}', flush=True)
            continue
        theme = THEMES[i % len(THEMES)]
        mapping = build_mapping(client, u, theme, prefs, prose, tags, interests)
        n = apply_mapping(u, mapping)
        left = remaining_christian_prefs(u)
        print(f'[{i+1}/{len(convert)}] p{u}: REWRITE theme="{theme[0][:30]}" '
              f'mapped={len(mapping)} replaced={n} remaining_christian_prefs={left}',
              flush=True)


if __name__ == '__main__':
    main()
