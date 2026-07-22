#!/usr/bin/env python3
"""Approach A — position in the mode's ACTUAL context (consolidated memory).

For the memory modes the model never reads the raw repeated events: it reads a
CONSOLIDATED memory where each preference is deduped to ~1 line. So "position"
is well-defined (no multiplicity). For each Personalization+Recommendation query
we locate the GT preference in the memory snapshot (<= t_test) and measure:
  - presence: did the preference survive consolidation?
  - position: which third of the memory doc (top / middle / bottom)
then relate to accuracy. Matching is keyword-overlap (approximate — paraphrase
causes misses, esp. recsys which has only hashtags); presence is a LOWER bound.
"""
import csv, json, glob, re, collections, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10**9)
from scripts.aggregate_eval import _accuracy_value          # noqa: E402
from evaluation.task_registry import normalize_task_type    # noqa: E402

MATCHED = {"1", "2", "3", "5", "6", "8", "9", "10", "13", "14"}
# (label, accuracy_dir, snapshot_dir) — each model builds its OWN memory, so the
# snapshots differ; Gemini accuracy lives in the judged dir.
MEM_MODES = [
    ("GPT-5.5 Textual Memory", "llm_memory_gpt5.5", "llm_memory_gpt5.5"),
    ("Gemini-3.5-Flash Textual Memory", "llm_memory_gemini3.5flash_judged", "llm_memory_gemini3.5flash"),
]
TASKS = {"chatbot_personalized_response", "local_recommendation_geo_shift",
         "personalized_recommendation", "at_ai_directive_followup"}
STOP = {"enjoy", "enjoys", "content", "social", "media", "like", "likes", "user",
        "really", "featuring", "video", "videos", "stuff", "things", "about", "with"}


def words(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return {w for w in re.findall(r"[a-zA-Z]{4,}", s.lower())}


def gt_words(full):
    w = set()
    for p in (full.get("gt_slice", {}) or {}).get("target", [])[:1]:
        w |= words(p.get("persona_item", "")) | words(p.get("category", ""))
        w |= words(" ".join(p.get("source_hashtags", [])))
    ho = full.get("held_out_idx")
    cands = full.get("candidates") or []
    if ho is not None and isinstance(ho, int) and 0 <= ho < len(cands):
        # recsys: hashtags are the only preference signal (NOT the title — that's
        # the future item's wording, not the user's stored interest)
        w |= words(" ".join(cands[ho].get("hashtags", [])))
    if full.get("source_hashtags"):
        w |= words(" ".join(full["source_hashtags"]))
    return w - STOP


_SNAP = {}


def load_snaps(uid, snap_dir):
    snaps = []
    for fp in glob.glob(f"{ROOT}/results/{snap_dir}/{uid}/memory_states/*.json"):
        snaps.append(json.load(open(fp)))
    snaps.sort(key=lambda d: d["t_test"])
    return snaps


def snap_for(uid, T, snap_dir):
    snaps = _SNAP.setdefault((uid, snap_dir), load_snaps(uid, snap_dir))
    best = None
    for d in snaps:
        if d["t_test"] <= T and (best is None or d["t_test"] > best["t_test"]):
            best = d
    return best


def locate(gw, snap):
    """-> (present, position_fraction, freq) for the best-matching memory line."""
    lines = [l for l in snap["memory"].split("\n") if l.strip().startswith("-")]
    if not lines or not gw:
        return False, None, None
    best_ov, best_i = 0, None
    for i, l in enumerate(lines):
        ov = len(gw & words(l))
        if ov > best_ov:
            best_ov, best_i = ov, i
    if best_ov < 2:                     # require 2-word overlap to call it present
        return False, None, None
    freq = None
    m = re.search(r"\[×(\d+)\]", lines[best_i])
    if m:
        freq = int(m.group(1))
    return True, best_i / len(lines), freq


def insts(uid):
    return {it["query_id"]: it for it in json.load(open(f"{ROOT}/backend/{uid}/test.json"))}


def main():
    inst_cache = {}
    for label, mdir, snap_dir in MEM_MODES:
        rows = []
        for fp in sorted(glob.glob(f"{ROOT}/results/{mdir}/*/results.csv")):
            uid = Path(fp).parent.name
            if uid not in MATCHED:
                continue
            ic = inst_cache.setdefault(uid, insts(uid))
            for r in csv.DictReader(open(fp)):
                tt = normalize_task_type(r.get("task_type", ""))
                if tt not in TASKS:
                    continue
                try:
                    m = json.loads(r.get("metrics_json") or "{}")
                except Exception:
                    m = {}
                acc = _accuracy_value(r.get("task_type", ""), m, r.get("status", ""))
                if acc is None:
                    continue
                it = ic.get(r.get("query_id"))
                if not it:
                    continue
                full = it.get("instance_full") or {}
                T = it.get("ts") or full.get("t_test")
                snap = snap_for(uid, T, snap_dir)
                if not snap:
                    continue
                present, frac, freq = locate(gt_words(full), snap)
                rows.append({"task": tt, "acc": acc, "present": present, "frac": frac, "freq": freq})

        print("=" * 78)
        print(f"{label}   (n={len(rows)})   [keyword match — presence is a LOWER bound]")
        print("=" * 78)
        # presence rate per task
        bytask = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            bytask[r["task"]][0] += 1
            bytask[r["task"]][1] += r["present"]
        print("  presence in consolidated memory, per task:")
        for t, (n, k) in sorted(bytask.items()):
            print(f"    {t:34} {k:>3}/{n:<3}  {100*k/n if n else 0:>3.0f}% present")
        # accuracy present vs absent
        pres = [r["acc"] for r in rows if r["present"]]
        absn = [r["acc"] for r in rows if not r["present"]]
        print(f"\n  accuracy | present   = {statistics.mean(pres):.0f}  (n={len(pres)})" if pres else "")
        print(f"  accuracy | ABSENT    = {statistics.mean(absn):.0f}  (n={len(absn)})" if absn else "")
        # accuracy by position third
        thirds = collections.defaultdict(list)
        for r in rows:
            if r["frac"] is None:
                continue
            t = "top" if r["frac"] < 1/3 else ("middle" if r["frac"] < 2/3 else "bottom")
            thirds[t].append(r["acc"])
        print("  accuracy by position in the memory doc:")
        for t in ("top", "middle", "bottom"):
            v = thirds.get(t, [])
            if v:
                print(f"    {t:8} {statistics.mean(v):.0f}  (n={len(v)})")
        print()


if __name__ == "__main__":
    main()
