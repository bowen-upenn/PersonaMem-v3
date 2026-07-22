#!/usr/bin/env python3
"""First-principles validation of distance-in-context -> performance.

Controlled needle-in-context: take a persona's real chronological history (the
same _compact_event lines the long-context eval uses), insert ONE unique,
verifiable fact at a controlled position, append a direct question, and check the
answer. Only the needle's POSITION varies — content, query, history, copy-count
(=1) are all held fixed — so any accuracy change is caused by distance alone.

position f: 0.0 = top of context (far from query) ... 1.0 = just above the query.
Reports accuracy + token-offset (tokens between needle and query) per position.

Usage: python results/_scripts/niah_position_validation.py --personas 1 --positions 0,0.5,1.0   # smoke
       python results/_scripts/niah_position_validation.py                                       # full
"""
import json, glob, re, sys, argparse, statistics, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.inference_utils import _compact_event, count_tokens   # noqa: E402
from evaluation.backend_query import APPS as BQ_APPS                  # noqa: E402

MATCHED = ["1", "2", "3", "5", "6", "8", "9", "10", "13", "14"]
CACHE = ROOT / "results/_scripts/_cache_niah.jsonl"
POSITIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# A few unique, unguessable needles (verbatim-checkable). One persona-plausible
# self-note each; the model must reproduce the CODE.
NEEDLES = [
    ("reservation confirmation code", "ZQ7X4K",
     "Note to self: my reservation confirmation code for the trip is ZQ7X4K.",
     "What is my reservation confirmation code? Reply with just the code."),
    ("storage locker number", "B-2947",
     "Reminder: I moved my stuff into storage locker number B-2947 today.",
     "What is my storage locker number? Reply with just the number."),
    ("bike lock combination", "61-08-33",
     "Setting my new bike lock combination to 61-08-33 so I do not forget.",
     "What is my bike lock combination? Reply with just the combination."),
]


def lines_for(uid):
    """Chronological _compact_event JSON lines (ts,app order), same as long-ctx."""
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
            tl.append((ts, app, line))
    tl.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in tl]


PREAMBLE = "# Cross-app engagement timeline (oldest first; one event per line)\n"
QTMPL = "\n\nYou are the user's assistant. Using the timeline above, answer.\nQuestion: {q}\nAnswer:"


def load_cache():
    if not CACHE.exists():
        return {}
    return {json.loads(l)["key"]: json.loads(l) for l in CACHE.open() if l.strip()}


def run(personas, positions, workers=6):
    cache = load_cache()
    todo = []
    base = {}
    for uid in personas:
        base[uid] = lines_for(uid)
    for uid in personas:
        lines = base[uid]
        for (topic, code, needle_line, q) in NEEDLES:
            nl = json.dumps({"app": "chatbot", "type": "self_note", "user_message": needle_line},
                            ensure_ascii=False)
            for f in positions:
                key = f"{uid}|{code}|{f}"
                if key in cache:
                    continue
                todo.append((key, uid, lines, nl, f, code, q))
    print(f"to call: {len(todo)} (cached {len(cache)})")
    if not todo:
        return
    from query_llm import QueryLLM
    from concurrent.futures import ThreadPoolExecutor
    client = QueryLLM({"models": {"llm_model": "gpt-5.5"}}, rate_limit_per_min=40)

    def one(job):
        key, uid, lines, nl, f, code, q = job
        idx = int(round(f * len(lines)))
        merged = lines[:idx] + [nl] + lines[idx:]
        # token offset from needle to the query (tokens of lines AFTER the needle)
        tok_after = count_tokens("\n".join(merged[idx + 1:]), "gpt-5.5")
        prompt = PREAMBLE + "\n".join(merged) + QTMPL.format(q=q)
        try:
            resp = client.query_llm(prompt) or ""
        except Exception as e:
            return {"key": key, "error": str(e)[:120]}
        ok = code.lower().replace(" ", "") in resp.lower().replace(" ", "")
        return {"key": key, "uid": uid, "code": code, "f": f, "tok_after": tok_after,
                "n_lines": len(lines), "correct": int(ok), "resp": resp[:60]}

    done = 0
    with CACHE.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, todo):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(todo)}")
    print("done", done)


def analyze():
    cache = load_cache()
    rows = [v for v in cache.values() if "correct" in v]
    print(f"\nNIAH position validation — n={len(rows)} (controlled: 1 copy, fixed content, position varies)")
    byf = collections.defaultdict(list)
    for r in rows:
        byf[r["f"]].append(r)
    print(f"  {'pos f':>6}{'n':>5}{'acc%':>7}{'med tok-from-query':>20}")
    for f in sorted(byf):
        s = byf[f]
        acc = 100 * sum(x["correct"] for x in s) / len(s)
        tok = statistics.median([x["tok_after"] for x in s])
        print(f"  {f:>6}{len(s):>5}{acc:>7.0f}{tok/1000:>17.0f}k")
    # by absolute token-offset bins
    print("  by token distance from query:")
    TB = [(0, 30_000), (30_000, 80_000), (80_000, 150_000), (150_000, 300_000), (300_000, 10**12)]
    TL = ["<30k", "30–80k", "80–150k", "150–300k", "300k+"]
    for (lo, hi), lab in zip(TB, TL):
        s = [r for r in rows if lo <= r["tok_after"] < hi]
        if s:
            print(f"    {lab:>9}: acc={100*sum(x['correct'] for x in s)/len(s):.0f}  (n={len(s)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default=",".join(MATCHED))
    ap.add_argument("--positions", default=",".join(map(str, POSITIONS)))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        run(a.personas.split(","), [float(x) for x in a.positions.split(",")], a.workers)
    analyze()
