#!/usr/bin/env python3
"""Option A — semantic retention of a preference in the iteratively-built memory.

Uses GPT-5.5 as a SEMANTIC locator (fixes the rewording confound of exact-line
matching): for each Personalization/Recommendation recall query, does the GT
preference survive into the consolidated GPT-5.5 memory snapshot (<= t_test), and
at what line? Then relate retention to (a) evidence age and (b) the memory
reader's accuracy. Caches every locator verdict to JSONL so reruns are free.

Usage:
  python results/_scripts/memory_retention_llm.py --limit 20   # smoke
  python results/_scripts/memory_retention_llm.py              # full
  python results/_scripts/memory_retention_llm.py --analyze    # no calls, report cache
"""
import csv, json, glob, re, sys, argparse, statistics, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10**9)
from scripts.aggregate_eval import _accuracy_value          # noqa: E402
from evaluation.task_registry import normalize_task_type    # noqa: E402

spec_dir = ROOT / "results/_scripts"
import importlib.util
_nd = importlib.util.spec_from_file_location("nd", spec_dir / "needle_depth_vs_accuracy.py")
nd = importlib.util.module_from_spec(_nd); _nd.loader.exec_module(nd)
_mp = importlib.util.spec_from_file_location("mp", spec_dir / "memory_position.py")
mp = importlib.util.module_from_spec(_mp); _mp.loader.exec_module(mp)

MATCHED = nd.MATCHED
MEM_DIR = "llm_memory_gpt5.5"            # GPT-5.5-built memory + its reader accuracy
TASKS = {"personalized_recommendation", "chatbot_personalized_response"}
CACHE = spec_dir / "_cache_mem_retention_gpt.jsonl"


def pref_desc(full, task):
    if task == "chatbot_personalized_response":
        tgt = (full.get("gt_slice", {}) or {}).get("target", [])
        if tgt:
            p = tgt[0]
            return f"{p.get('persona_item','')} (category: {p.get('category','')})"
        hp = full.get("held_out_preference", {})
        return f"{hp.get('persona_item','')} (category: {hp.get('category','')})"
    # recsys: the held-out item's topic = its hashtags
    ho = full.get("held_out_idx"); cands = full.get("candidates") or []
    if ho is not None and 0 <= ho < len(cands):
        c = cands[ho]
        tags = ", ".join("#" + h.lstrip("#") for h in c.get("hashtags", []))
        return f"An interest in this kind of content: {tags}"
    return ""


def numbered_mem(snap):
    lines = [l.strip() for l in snap["memory"].split("\n") if l.strip().startswith("-")]
    return lines, "\n".join(f"{i+1}: {l.lstrip('- ').strip()}" for i, l in enumerate(lines))


PROMPT = """You check whether a user preference is recorded in a consolidated USER MEMORY.

PREFERENCE:
{pref}

USER MEMORY (numbered lines):
{mem}

Does the memory record this preference or interest? A paraphrase counts as present;
require a genuine topical match, not a vague tangential one. Reply with ONLY JSON:
{{"present": true or false, "line": <the single best line number, or null>}}"""


def build_units():
    units, inst_cache = [], {}
    for fp in sorted(glob.glob(f"{ROOT}/results/{MEM_DIR}/*/results.csv")):
        uid = Path(fp).parent.name
        if uid not in MATCHED:
            continue
        ic = inst_cache.setdefault(uid, mp.insts(uid))
        ev = nd.load_events(uid)
        for r in csv.DictReader(open(fp)):
            tt = normalize_task_type(r.get("task_type", ""))
            if tt not in TASKS:
                continue
            m = json.loads(r.get("metrics_json") or "{}")
            acc = _accuracy_value(r.get("task_type", ""), m, r.get("status", ""))
            if acc is None:
                continue
            it = ic.get(r.get("query_id"))
            if not it:
                continue
            full = it.get("instance_full") or {}
            T = it.get("ts") or full.get("t_test")
            depth, _, _ = nd.needle_depth(full, T, ev)
            if depth is None:
                continue
            snap = mp.snap_for(uid, T, MEM_DIR)
            if not snap:
                continue
            desc = pref_desc(full, tt)
            if not desc.strip():
                continue
            units.append({"qid": r["query_id"], "uid": uid, "task": tt, "acc": acc,
                          "depth": depth, "snap": snap, "desc": desc})
    return units


def load_cache():
    if not CACHE.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in CACHE.open() if l.strip()}


def parse_json(txt):
    txt = re.sub(r"^```(json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def run(limit=None, workers=8):
    units = build_units()
    cache = load_cache()
    todo = [u for u in units if u["qid"] not in cache]
    if limit:
        todo = todo[:limit]
    print(f"units={len(units)}  cached={len(cache)}  to call={len(todo)}")
    if not todo:
        return
    from query_llm import QueryLLM
    client = QueryLLM({"models": {"llm_model": "gpt-5.5"}}, rate_limit_per_min=50)

    def one(u):
        lines, numbered = numbered_mem(u["snap"])
        try:
            out = client.query_llm(PROMPT.format(pref=u["desc"], mem=numbered))
            j = parse_json(out)
            present = bool(j.get("present"))
            line = j.get("line") if isinstance(j.get("line"), int) else None
            frac = (line - 1) / len(lines) if (present and line and 1 <= line <= len(lines)) else None
            return {"qid": u["qid"], "task": u["task"], "depth": u["depth"], "acc": u["acc"],
                    "present": present, "line": line, "n_lines": len(lines), "frac": frac}
        except Exception as e:
            return {"qid": u["qid"], "error": str(e)[:120]}

    done = 0
    with CACHE.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, todo):
            f.write(json.dumps(res) + "\n"); f.flush()
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(todo)}")
    print(f"done {done}")


def analyze():
    cache = load_cache()
    rows = [v for v in cache.values() if "present" in v]
    print(f"\nOption A — semantic retention (GPT-5.5 locator), n={len(rows)}")
    # retention by evidence age
    B = [(0, 2), (2, 4), (4, 7), (7, 99)]; L = ["0–2d", "2–4d", "4–7d(mid)", "7d+"]
    print(f"  {'age':>12}{'n':>5}{'retained%':>11}{'acc':>6}{'acc|kept':>10}{'acc|lost':>10}")
    for (lo, hi), lab in zip(B, L):
        sub = [r for r in rows if lo <= r["depth"] < hi]
        if not sub:
            continue
        kept = [r["acc"] for r in sub if r["present"]]
        lost = [r["acc"] for r in sub if not r["present"]]
        ret = 100 * len(kept) / len(sub)
        print(f"  {lab:>12}{len(sub):>5}{ret:>10.0f}%{statistics.mean(r['acc'] for r in sub):>6.0f}"
              f"{(statistics.mean(kept) if kept else float('nan')):>10.0f}{(statistics.mean(lost) if lost else float('nan')):>10.0f}")
    kept = [r["acc"] for r in rows if r["present"]]; lost = [r["acc"] for r in rows if not r["present"]]
    print(f"\n  overall retained: {100*len(kept)/len(rows):.0f}%   acc|retained={statistics.mean(kept):.0f}  acc|dropped={statistics.mean(lost):.0f}")
    # position of retained pref by age
    print("\n  position (fraction down the memory doc) of the retained pref, by age:")
    for (lo, hi), lab in zip(B, L):
        fr = [r["frac"] for r in rows if lo <= r["depth"] < hi and r.get("frac") is not None]
        if fr:
            print(f"    {lab:>12}  median {statistics.median(fr):.2f}  (n={len(fr)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        run(limit=a.limit, workers=a.workers)
    analyze()
