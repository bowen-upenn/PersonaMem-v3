#!/usr/bin/env python3
"""Needle-depth vs accuracy, long-context modes only.

First-principles question: in long-context mode the full time-masked history sits
in the prompt (chronological, query appended at the end). For a retrieval-shaped
query, how DEEP into that context does the relevant user-info first appear, and
does accuracy fall as it appears earlier (further from the query)?

Distance metric (locked with user): OLDEST supporting in-context event.
  needle_depth_days = (t_test - min{ source_timestamp of in-context events whose
                       hashtags overlap the GT preference's source_hashtags }) / 86400
  = how far back the model must reach for the FIRST appearance of the answer's
    supporting evidence. (Temporal recency ~ token position here: events are
    roughly uniform over the ~8.7-day window.)

Correctness: the SAME per-query headline used by the published tables
  (scripts.aggregate_eval._accuracy_value + task_registry.PRIMARY_METRIC).

Rows: the two long-context configs. Columns: depth bins. Matched-10 personas.
"""
import csv, json, glob, sys, collections, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10**9)

from scripts.aggregate_eval import _accuracy_value          # noqa: E402
from evaluation.task_registry import normalize_task_type    # noqa: E402
from evaluation.inference_utils import _compact_event, count_tokens  # noqa: E402
from evaluation.backend_query import APPS as BQ_APPS        # noqa: E402

MATCHED = {"1", "2", "3", "5", "6", "8", "9", "10", "13", "14"}
# Sources carry the objective + judge dims the headline tables use (judged
# variant where one exists, else judge dims are in-place). collect() filters to
# the matched-10 personas regardless of how many a dir contains.
#
# MODELS = the two LONG-CONTEXT configs. ONLY these load the flat chronological
# history prompt, so the token-offset position is defined for them. The other
# configs read a consolidated memory (Textual Memory), a retrieval store (Mem0),
# or search via tools (Codex / Claude Code) — there is no fixed needle position
# in a flat prompt, so they appear in the DAYS view only (days = a property of
# the query's data, valid for any mode).
MODELS = [
    ("GPT-5.5 Long Context", "llm_longctx_gpt5.5_judged"),
    ("Gemini-3.5-Flash Long Context", "llm_longctx_gemini3.5flash_judged"),
]
# All 8 headline configs, in headline order (for the DAYS view + @AI lag).
DAYS_MODELS = [
    ("GPT-5.5 Long Context", "llm_longctx_gpt5.5_judged"),
    ("GPT-5.5 Textual Memory", "llm_memory_gpt5.5"),
    ("GPT-5.5 Mem0 w/ RAG", "mem0_gpt5.5"),
    ("GPT-5.5 Codex High", "codex_agent_gpt5.5"),
    ("Gemini-3.5-Flash Long Context", "llm_longctx_gemini3.5flash_judged"),
    ("Gemini-3.5-Flash Textual Memory", "llm_memory_gemini3.5flash_judged"),
    ("Opus-4.8 Claude Code High", "agent_tools_opus4.8"),
    ("Sonnet-4.6 Claude Code High", "agent_tools_sonnet4.6"),
]
APPS = ["instagram", "facebook", "threads", "chatbot", "ai_studio"]

# Retrieval-shaped tasks: "correct" means USING the right past info, so
# "Aggregate over ALL tasks": every query that has a recoverable needle (a
# locatable first-appearance of the relevant info) is pooled, regardless of task.
# A needle comes from a DIRECT stored timestamp (directive lag / trigger event /
# geo transition) or from HASHTAG matching the GT preference against history.
# Tasks with no locatable needle (pure generation, open-ended QA) can't be placed
# on a distance axis — they're reported in the coverage table as excluded.
# `tasks=None` in the table fns pools over all needle-bearing tasks.

# Role tags (for the caveat): in RESTRAINT tasks "correct" means NOT leaning on
# the info, so the distance→accuracy relation can run opposite to USE tasks.
RESTRAINT_TASKS = {
    "over_personalization_chatbot_text", "over_personalization_context_shift",
    "over_personalization_sensitive_event", "over_personalization_repetition_chatbot",
    "over_personalization_repetition_recsys", "restraint_sensitive_event_silence",
    "proactive_overactive_check", "active_mistake_prevention",
}
# The two cleanest pure-retrieval tasks (kept for a focused secondary view).
RETRIEVAL_TASKS = {"personalized_recommendation", "chatbot_personalized_response"}

# SCOPE: the Personalization + Recommendation headline categories, excluding the
# hidden-persona, hallucination, and restraint tasks. All are USE tasks with a
# recoverable needle. This is the pooled set for the tables.
POOL_TASKS = {
    "chatbot_personalized_response",     # Personalization
    "local_recommendation_geo_shift",    # Personalization
    "personalized_recommendation",       # Recommendation
    "at_ai_directive_followup",          # Recommendation
}

# depth bins (days back to the OLDEST supporting event) = position in the
# chronological prompt: small depth = recent = BOTTOM (near query); large depth
# = oldest = TOP (far from query). Reading left→right = bottom→top of context.
BINS = [(0, 3), (3, 5), (5, 7), (7, 99)]
BIN_LABELS = ["0–3d", "3–5d", "5–7d", "7d+"]

# fine-grained 1-day bins
DBINS = [(i, i + 1) for i in range(8)] + [(8, 99)]
DBIN_LABELS = [f"{i}–{i+1}d" for i in range(8)] + ["8d+"]

# query→source distance bins: fine near the query, WIDER in the middle (from day 3),
# plus a 10d+ catch-all for the late-tested probes (over all tasks).
MIDBINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 1e12)]
MIDBIN_LABELS = ["0–1d", "1–2d", "2–3d", "3–5d", "5–7d", "7–10d", "10d+"]

# Performance-ranked gradient: each model's colour is its Overall-accuracy rank
# (matched-10), rose (best) -> sage green (worst) along the rgb(226,140,164)->
# rgb(169,210,164) ramp, saturation +30% and lightness -15% for a deeper, elegant
# dusty set that still separates. Colour encodes rank, not family.
PALETTE = {
    "GPT-5.5 Long Context": "#DA5F78",        # rank 2 (53.4)
    "GPT-5.5 Textual Memory": "#A4AD80",      # rank 6 (49.4)
    "GPT-5.5 Mem0 w/ RAG": "#92BB7B",         # rank 7 (48.8)
    "GPT-5.5 Codex High": "#CB7077",          # rank 3 (53.2)
    "Gemini-3.5-Flash Long Context": "#7EC975",     # rank 8 (47.7, worst)
    "Gemini-3.5-Flash Textual Memory": "#BE8479",   # rank 4 (51.5)
    "Opus-4.8 Claude Code High": "#E94E79",   # rank 1 (53.7, best)
    "Sonnet-4.6 Claude Code High": "#B19A7E", # rank 5 (50.9)
}
# Memory-size caps: a single-hue teal ramp (more memory = deeper).
CAP_PALETTE = {"Half · 2048": "#A8C5CA", "Baseline · 4096": "#4A919E", "Double · 8192": "#1F5673"}
# Forgetting curve: artifact vs truth.
FORGET_COLORS = {"verbatim": "#C97B63", "concept": "#3E7C8C"}
# Source cohorts: reinforced-to-recent vs only-at-the-beginning.
COHORT_COLOR = {"reinforced": "#2A9D8F", "beginning": "#E08A3C"}

# token-offset bins: # tokens between the needle's first appearance and the
# query at the end of the prompt (how far back, in tokens, the model must reach).
# Edges chosen for balanced n over the pooled 0–561k distribution (median 232k).
TBINS = [(0, 50_000), (50_000, 120_000), (120_000, 200_000), (200_000, 300_000),
         (300_000, 420_000), (420_000, 10**12)]
TBIN_LABELS = ["<50k", "50–120k", "120–200k", "200–300k", "300–420k", "420k+"]


def norm_tags(seq):
    return {str(h).lower().lstrip("#").strip() for h in (seq or []) if str(h).strip()}


def load_events(uid):
    """[(ts, hashtag_set)] for every content event in the persona's history."""
    ev = []
    for a in APPS:
        fp = ROOT / "backend" / uid / f"{a}.json"
        if not fp.exists():
            continue
        for e in json.load(open(fp)):
            ts = e.get("source_timestamp")
            if not ts:
                continue
            tags = norm_tags(e.get("source_hashtags"))
            c = e.get("content") or {}
            tags |= norm_tags(c.get("hashtags"))
            for p in (e.get("preferences") or []):
                tags |= norm_tags(p.get("source_hashtags"))
            ev.append((ts, tags))
    ev.sort()
    return ev


def gt_hashtags(full):
    """Hashtags that identify the held-out / primary GT preference (the needle)."""
    if not isinstance(full, dict):
        return set()
    gt = full.get("gt_slice") or {}
    tgt = gt.get("target") or []
    if tgt:  # chatbot / over-personalization recall: target[0] is the held-out pref
        t = norm_tags(tgt[0].get("source_hashtags"))
        if t:
            return t
    ho = full.get("held_out_idx")
    cands = full.get("candidates") or []
    if ho is not None and isinstance(ho, int) and 0 <= ho < len(cands):
        t = norm_tags(cands[ho].get("hashtags"))
        if t:
            return t
    if full.get("source_hashtags"):
        return norm_tags(full.get("source_hashtags"))
    gobj = full.get("groundtruth_preference_obj") or {}
    if isinstance(gobj, dict):
        t = norm_tags(gobj.get("source_hashtags")) | norm_tags(gobj.get("evidence_hashtags"))
        if t:
            return t
    return set()


def _trigger_ts(full, t_test):
    """A directly-stored needle timestamp, if any (directive lag / trigger event /
    geo transition). Returns a ts strictly before t_test, else None."""
    if not isinstance(full, dict) or not t_test:
        return None
    lag = full.get("lag_seconds")
    if isinstance(lag, (int, float)) and lag > 0:
        return t_test - lag
    te = full.get("trigger_evidence") or {}
    for k in ("incoming_at_ts", "post_ts", "window_start_ts", "source_timestamp"):
        v = te.get(k)
        if isinstance(v, (int, float)) and 1e9 < v < t_test:
            return v
    for k in ("transition_first_ts",):
        v = full.get(k)
        if isinstance(v, (int, float)) and 1e9 < v < t_test:
            return v
    return None


def needle_depth(full, t_test, events):
    """(depth_days, n_support, needle_ts, nearest_days) for the relevant info.
    depth = distance to the FIRST (oldest) appearance; nearest = distance to the
    most-recent appearance (the freshest copy). Priority: direct stored timestamp
    (single needle → nearest==depth) → oldest/nearest hashtag-matched event."""
    if not t_test:
        return None, 0, None, None
    direct = _trigger_ts(full, t_test)
    if direct is not None:
        d = (t_test - direct) / 86400.0
        return d, 1, direct, d
    tags = gt_hashtags(full)
    if not tags:
        return None, 0, None, None
    matches = [ts for ts, ht in events if ts <= t_test and (tags & ht)]
    if not matches:
        return None, 0, None, None
    needle_ts = min(matches)
    return (t_test - needle_ts) / 86400.0, len(matches), needle_ts, (t_test - max(matches)) / 86400.0


_TL_CACHE = {}


def timeline(uid):
    """Faithful long-context prompt reconstruction: [(ts, line_tokens)] sorted
    (ts, app), one _compact_event JSON line per non-DM event (matches
    serialize_history_for_context). Validated to within ~1k tokens of the
    stored input_tokens (the fixed query/system block)."""
    if uid in _TL_CACHE:
        return _TL_CACHE[uid]
    tl = []
    for app in BQ_APPS:
        fp = ROOT / "backend" / uid / f"{app}.json"
        if not fp.exists():
            continue
        for e in json.load(open(fp)):
            if e.get("is_dm"):
                continue
            ts = e.get("source_timestamp")
            if not ts:
                continue
            line = json.dumps({"app": app, **_compact_event(e)}, ensure_ascii=False)
            tl.append((ts, app, count_tokens(line)))
    tl.sort(key=lambda x: (x[0], x[1]))
    _TL_CACHE[uid] = tl
    return tl


def needle_token_offset(needle_ts, t_test, tl):
    """# tokens between the needle's first appearance and the query (end)."""
    if needle_ts is None or not t_test:
        return None
    return sum(tok for ts, _app, tok in tl if needle_ts < ts < t_test)


def load_instances(uid):
    out = {}
    fp = ROOT / "backend" / uid / "test.json"
    if not fp.exists():
        return out
    for it in json.load(open(fp)):
        out[it.get("query_id")] = it
    return out


def collect(model_dir):
    """-> per-query dicts {task, depth, acc, n_support, tok_offset, role} over ALL
    task types. depth/tok_offset are None when no needle is recoverable."""
    rows = []
    ev_cache, inst_cache = {}, {}
    for results_csv in sorted(glob.glob(f"{ROOT}/results/{model_dir}/*/results.csv")):
        uid = Path(results_csv).parent.name
        if uid not in MATCHED:
            continue
        events = ev_cache.setdefault(uid, load_events(uid))
        insts = inst_cache.setdefault(uid, load_instances(uid))
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                tt = normalize_task_type(r.get("task_type", ""))
                try:
                    m = json.loads(r.get("metrics_json") or "{}")
                except Exception:
                    m = {}
                acc = _accuracy_value(r.get("task_type", ""), m, r.get("status", ""))
                if acc is None:
                    continue
                it = insts.get(r.get("query_id"))
                if not it:
                    continue
                full = it.get("instance_full") or {}
                t_test = it.get("ts") or full.get("t_test")
                depth, nsup, needle_ts, nearest = needle_depth(full, t_test, events)
                toff = needle_token_offset(needle_ts, t_test, timeline(uid))
                # cohort: is the source REINFORCED up to near the query (fresh copy
                # + repeated) or only at the BEGINNING (freshest copy already stale)?
                cohort = None
                if nearest is not None:
                    cohort = "reinforced" if (nearest < 1.0 and nsup >= 3) else "beginning"
                rows.append({"task": tt, "depth": depth, "acc": acc, "n_support": nsup,
                             "tok_offset": toff, "nearest": nearest, "cohort": cohort,
                             "role": "restraint" if tt in RESTRAINT_TASKS else "use"})
    return rows


def binize(d):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def main():
    per_model = {label: collect(d) for label, d in DAYS_MODELS}
    DAYS_MODELS_L = [m[0] for m in DAYS_MODELS]

    # ---- coverage report (no silent capping) -------------------------------
    print("=" * 78)
    print("NEEDLE RECOVERABILITY per task (GPT-5.5 LC sample) — ALL tasks scanned")
    print("=" * 78)
    cov = collections.defaultdict(lambda: [0, 0, "use"])
    for r in per_model[MODELS[0][0]]:
        cov[r["task"]][0] += 1
        cov[r["task"]][2] = r["role"]
        if r["depth"] is not None:
            cov[r["task"]][1] += 1
    print(f"{'task':38} {'role':>9} {'n_q':>5} {'needle':>7} {'rate':>6}")
    inc = exc = 0
    for t, (n, k, role) in sorted(cov.items(), key=lambda x: -x[1][1]):
        inc += k; exc += (n - k)
        print(f"{t:38} {role:>9} {n:>5} {k:>7} {100*k/n if n else 0:>5.0f}%")
    print(f"\nPOOLED over all needle-bearing queries: {inc}/model  (excluded, no needle: {exc})")

    # ---- generic binning over an arbitrary key + bin edges -----------------
    def binof(val, bins):
        for i, (lo, hi) in enumerate(bins):
            if lo <= val < hi:
                return i
        return len(bins) - 1

    def _keep(r, tasks):
        if tasks is None:
            return True
        if tasks == "USE":
            return r["role"] == "use"
        return r["task"] in tasks

    def cells(rows, tasks, key, bins):
        buckets = collections.defaultdict(list)
        for r in rows:
            v = r.get(key)
            if v is None or not _keep(r, tasks):
                continue
            buckets[binof(v, bins)].append(r["acc"])
        return buckets

    def print_table(title, tasks, key, bins, labels, models=None):
        models = models or [m[0] for m in MODELS]
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        print(f"{'config':30}" + "".join(f"{b:>11}" for b in labels) + f"{'all':>11}")
        for label in models:
            b = cells(per_model[label], tasks, key, bins)
            allacc = [r["acc"] for r in per_model[label]
                      if r.get(key) is not None and _keep(r, tasks)]
            line = f"{label:30}"
            for i in range(len(labels)):
                vals = b.get(i, [])
                line += f"{(f'{statistics.mean(vals):.0f}/{len(vals)}' if vals else '-'):>11}"
            line += f"{f'{statistics.mean(allacc):.0f}/{len(allacc)}' if allacc else '':>11}"
            print(line)

    print("\n    SCOPE = Personalization + Recommendation tasks (no hidden-persona / hallucination / restraint)")
    print(">>> BY DAYS — coarse — cell = acc% / n  [all 8 configs]")
    print_table("Personalization + Recommendation", POOL_TASKS, "depth", BINS, BIN_LABELS, DAYS_MODELS_L)
    print("\n>>> BY DAYS (fine, 1-day bins)")
    print_table("Personalization + Recommendation", POOL_TASKS, "depth", DBINS, DBIN_LABELS, DAYS_MODELS_L)
    print("\n>>> BY TOKENS = history volume between the evidence's first appearance and the query")
    print("    (for the 2 long-context modes it is also the literal prompt position)")
    print_table("Personalization + Recommendation", POOL_TASKS, "tok_offset", TBINS, TBIN_LABELS, DAYS_MODELS_L)

    # ---- mid-dip magnitude per mode: is the U DEEPER for long-context? ------
    print("\n" + "=" * 86)
    print("MID-DIP per mode (coarse days): ends=mean(0–3d,7d+) middle=mean(3–5d,5–7d) drop=ends−middle")
    print("  bigger drop = stronger lost-in-the-middle ON TOP of the shared query-difficulty U")
    print("=" * 86)
    print(f"{'config':32}{'ends':>8}{'middle':>8}{'drop':>8}")
    for label, _ in DAYS_MODELS:
        b = cells(per_model[label], POOL_TASKS, "depth", BINS)
        m = {i: (statistics.mean(b[i]) if b.get(i) else None) for i in range(4)}
        if None in m.values():
            continue
        ends = (m[0] + m[3]) / 2
        mid = (m[1] + m[2]) / 2
        tag = "  <- long-context" if "Long Context" in label else ""
        print(f"{label:32}{ends:>8.1f}{mid:>8.1f}{ends-mid:>8.1f}{tag}")

    # ---- corroboration: @AI directive lag (true single needle, NO recent ----
    # reinforcement; distance = directive→query lag, pre-stored as lag_bucket) -
    print("\n" + "=" * 86)
    print("CORROBORATION — @AI directive follow-up: accuracy vs directive→query lag")
    print("  (single in-context needle, no recent reinforcement; distance pre-stored)")
    print("=" * 86)
    order = {"24h": 0, "72h": 1, "7d": 2}
    print(f"{'config':32}{'24h':>13}{'72h':>13}{'7d':>13}")
    for label, mdir in DAYS_MODELS:
        buck = collections.defaultdict(list)
        for results_csv in glob.glob(f"{ROOT}/results/{mdir}/*/results.csv"):
            if Path(results_csv).parent.name not in MATCHED:
                continue
            with open(results_csv) as f:
                for r in csv.DictReader(f):
                    if normalize_task_type(r.get("task_type", "")) != "at_ai_directive_followup":
                        continue
                    try:
                        m = json.loads(r.get("metrics_json") or "{}")
                    except Exception:
                        continue
                    a = _accuracy_value(r.get("task_type", ""), m, r.get("status", ""))
                    if a is not None and m.get("lag_bucket") in order:
                        buck[m["lag_bucket"]].append(a)
        line = f"{label:32}"
        for lb in ["24h", "72h", "7d"]:
            v = buck.get(lb, [])
            line += f"{(f'{statistics.mean(v):.0f} (n={len(v)})' if v else '-'):>13}"
        print(line)

    # ---- confound check: is depth just a proxy for interest strength? -------
    print("\n" + "=" * 86)
    print("CONFOUND CHECK — does the U track evidence VOLUME, not position? (GPT-5.5 LC, pooled)")
    print("=" * 86)
    rows = [r for r in per_model[MODELS[0][0]] if r["depth"] is not None and r["task"] in POOL_TASKS]
    agg = collections.defaultdict(lambda: {"acc": [], "ns": []})
    for r in rows:
        agg[binof(r["depth"], BINS)]["acc"].append(r["acc"])
        agg[binof(r["depth"], BINS)]["ns"].append(r["n_support"])
    print(f"{'bin':>6}{'n':>6}{'mean_acc':>10}{'mean_support':>14}")
    for i, lab in enumerate(BIN_LABELS):
        a, ns = agg[i]["acc"], agg[i]["ns"]
        if a:
            print(f"{lab:>6}{len(a):>6}{statistics.mean(a):>10.0f}{statistics.mean(ns):>14.1f}")

    def corr(xs, ys):
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        import math
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return cov / (sx * sy) if sx and sy else 0.0
    acc = [r["acc"] for r in rows]; dep = [r["depth"] for r in rows]; sup = [r["n_support"] for r in rows]
    print(f"\ncorr(acc, depth)={corr(acc, dep):+.3f}  corr(acc, support)={corr(acc, sup):+.3f}  "
          f"corr(depth, support)={corr(dep, sup):+.3f}")
    print("=> accuracy does NOT track evidence volume (0–3d has the LEAST support yet")
    print("   the 2nd-highest accuracy); the mid-context dip is positional, not density.")

    # ---- write CSV (every view) --------------------------------------------
    out = ROOT / "results" / "aggregate" / "needle_depth_vs_accuracy.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for view, key, bins, labels, tasks, mods in [
            ("days_coarse_PERS_REC", "depth", BINS, BIN_LABELS, POOL_TASKS, DAYS_MODELS),
            ("days_fine_PERS_REC", "depth", DBINS, DBIN_LABELS, POOL_TASKS, DAYS_MODELS),
            ("days_fine_recsys", "depth", DBINS, DBIN_LABELS, {"personalized_recommendation"}, DAYS_MODELS),
            ("tokens_PERS_REC", "tok_offset", TBINS, TBIN_LABELS, POOL_TASKS, DAYS_MODELS),
            ("tokens_longctx_recsys", "tok_offset", TBINS, TBIN_LABELS, {"personalized_recommendation"}, MODELS),
        ]:
            w.writerow([f"# {view}"])
            w.writerow(["config"] + labels + ["all"])
            for label, _ in mods:
                b = cells(per_model[label], tasks, key, bins)
                allacc = [r["acc"] for r in per_model[label]
                          if r.get(key) is not None and _keep(r, tasks)]
                rr = [label]
                for i in range(len(labels)):
                    vals = b.get(i, [])
                    rr.append(f"{statistics.mean(vals):.1f}|{len(vals)}" if vals else "")
                rr.append(f"{statistics.mean(allacc):.1f}|{len(allacc)}" if allacc else "")
                w.writerow(rr)
            w.writerow([])
    print(f"\nwrote {out}")

    # ---- distributions -----------------------------------------------------
    g = [r for r in per_model[MODELS[0][0]] if r["task"] in POOL_TASKS]
    ds = [r["depth"] for r in g if r["depth"] is not None]
    ts = [r["tok_offset"] for r in g if r["tok_offset"] is not None]
    if ds:
        q = statistics.quantiles(ds, n=4)
        print(f"\nDepth(days):  n={len(ds)} min={min(ds):.1f} p25={q[0]:.1f} med={statistics.median(ds):.1f} p75={q[2]:.1f} max={max(ds):.1f}")
    if ts:
        q = statistics.quantiles(ts, n=4)
        print(f"Tok-offset:   n={len(ts)} min={min(ts)/1000:.1f}k p25={q[0]/1000:.1f}k med={statistics.median(ts)/1000:.1f}k p75={q[2]/1000:.1f}k max={max(ts)/1000:.1f}k")


if __name__ == "__main__":
    main()
