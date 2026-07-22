#!/usr/bin/env python3
"""Figure-1-style forgetting curve, but CONCEPT-level (rewording-robust).

Same axes as the verbatim curve — % of memory content first added on day d that
is still present at saturation — but a line "survives" if a line in the saturated
memory is SEMANTICALLY the same (cosine >= T on text-embedding-3-large), not byte
identical. Consolidation rewording therefore no longer counts as forgetting.

Concepts are formed by greedy clustering of all bullet lines (across daily
snapshots, in day order): a line joins an existing concept if cosine>=T to its
representative, else starts a new one. A concept's first_day = the day its first
member appeared; it "survives" iff a member appears in the saturated snapshot.
"""
import json, glob, re, statistics, collections, pickle, hashlib, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MATCHED = {"1", "2", "3", "5", "6", "8", "9", "10", "13", "14"}
MEM_DIR = "llm_memory_gpt5.5"
THRESHOLD = 0.45
EMB_CACHE = ROOT / "results/_scripts/_cache_mem_line_emb.pkl"


def _lines(m):
    return [re.sub(r"\[×\d+\]", "", l).lstrip("- ").strip().lower()
            for l in m.split("\n") if l.strip().startswith("-")]


def _snaps(uid, mem_dir=MEM_DIR):
    # All daily snapshots. The memory FREEZES once events end (~day 9), so the
    # day 10–34 snapshots are identical copies; survival is measured to the LAST.
    return sorted((json.load(open(fp)) for fp in
                   glob.glob(f"{ROOT}/results/{mem_dir}/{uid}/memory_states/*.json")),
                  key=lambda d: d["t_test"])


_EMB = pickle.load(open(EMB_CACHE, "rb")) if EMB_CACHE.exists() else {}


def _embed(texts):
    """Cached Azure embeddings -> {text: unit-vec}. One batched call for misses."""
    miss = sorted({t for t in texts if t not in _EMB})
    if miss:
        from dotenv import load_dotenv; load_dotenv()
        from openai import AzureOpenAI
        from evaluation.mem0_backend import _azure_env
        e = _azure_env()
        cli = AzureOpenAI(api_key=e["key"], azure_endpoint=e["endpoint"], api_version=e["api_version"])
        for i in range(0, len(miss), 512):
            chunk = miss[i:i + 512]
            r = cli.embeddings.create(model=e["embed_deployment"], input=chunk)
            for t, d in zip(chunk, r.data):
                v = np.array(d.embedding, dtype=np.float32)
                _EMB[t] = v / (np.linalg.norm(v) + 1e-9)
        pickle.dump(_EMB, open(EMB_CACHE, "wb"))
    return {t: _EMB[t] for t in texts}


def curve(threshold=THRESHOLD, mem_dir=MEM_DIR, min_per_day=5):
    by_day = collections.defaultdict(list)   # first_day -> [survived?]
    for uid in MATCHED:
        snaps = _snaps(uid, mem_dir)
        if len(snaps) < 3:
            continue
        day_lines = [_lines(s["memory"]) for s in snaps]
        allv = _embed(sorted({l for ls in day_lines for l in ls}))
        sat = set(day_lines[-1])          # final (frozen) memory state
        # greedy clusters in day order
        reps = []          # (vec, first_day, survived)
        for d, ls in enumerate(day_lines):
            for l in ls:
                v = allv[l]
                hit = None
                for k, (rv, _fd, _s) in enumerate(reps):
                    if float(v @ rv) >= threshold:
                        hit = k; break
                if hit is None:
                    reps.append([v, d, l in sat])
                else:
                    if l in sat:
                        reps[hit][2] = True   # a member survives to saturation
        for _v, fd, surv in reps:
            by_day[fd].append(surv)
    # min 5 concepts/day to plot (late days are thin: few genuinely-new concepts
    # appear once the consolidator is mostly rewording existing facts).
    days = sorted(d for d in by_day if len(by_day[d]) >= min_per_day)
    return [(d, 100 * sum(by_day[d]) / len(by_day[d]), len(by_day[d])) for d in days]


if __name__ == "__main__":
    for T in (0.40, 0.45, 0.55):
        c = curve(T)
        overall = statistics.mean([s for _, s, _ in c])
        print(f"\nthreshold={T}  (concept survival % by first-added day; mean={overall:.0f}%)")
        for d, s, n in c:
            print(f"  day {d:>2}: {s:>3.0f}%  (n={n})")
