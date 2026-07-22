#!/usr/bin/env python3
"""Score the FIXED persona-1 judge-agreement prompts with an extra judge
(gpt-5.4-mini by default) and merge into all_scores.jsonl. Idempotent: any prior
rows for this judge are dropped before the fresh ones are written. 50-way parallel."""
import json, sys, time, threading, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/_scripts"))
from judge_agreement import _score, RUBRIC_DIMS  # reuse exact rubric parsing
from query_llm import QueryLLM

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="gpt-5.4-mini")
ap.add_argument("--out_dir", default="results/audit/judge_agreement_p1")
ap.add_argument("--workers", type=int, default=50)
args = ap.parse_args()

OUT = ROOT / args.out_dir
JUDGE = args.judge
work = [json.loads(l) for l in open(OUT / "prompts.jsonl")]
print(f"[add_judge] {JUDGE}: scoring {len(work)} prompts, {args.workers}-way parallel")

client = QueryLLM({"models": {"llm_model": JUDGE}}, rate_limit_per_min=600)
rows, lock, done, t0 = [], threading.Lock(), [0], time.time()


def do(w):
    sc = _score(client, w["prompt"])
    r = {"item_key": w["item_key"], "query_id": w["query_id"], "task_type": w["task_type"],
         "population": w["population"], "polarity": w["polarity"], "judge": JUDGE, **sc}
    with lock:
        rows.append(r)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(work):
            print(f"[add_judge]   {done[0]}/{len(work)} ({time.time()-t0:.0f}s)", flush=True)
    return r


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for f in as_completed([ex.submit(do, w) for w in work]):
        f.result()

# idempotent merge
asf = OUT / "all_scores.jsonl"
keep = [json.loads(l) for l in open(asf)] if asf.exists() else []
keep = [r for r in keep if r.get("judge") != JUDGE]
with open(asf, "w") as f:
    for r in keep + rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

import numpy as np
allv = [r[d] for r in rows for d in RUBRIC_DIMS if r.get(d) is not None]
miss = [r for r in rows if any(r.get(d) is None for d in RUBRIC_DIMS)]
print(f"[add_judge] DONE: {len(rows)} rows, {len(allv)} dim-scores, "
      f"mean={np.mean(allv):.2f} std={np.std(allv):.2f}; rows with a missing dim: {len(miss)}")
