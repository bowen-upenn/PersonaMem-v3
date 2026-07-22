#!/usr/bin/env python3
"""Score the fixed single-persona judge-agreement prompts with a Claude judge
(sonnet/opus via `claude -p`) and merge into all_scores.jsonl. Idempotent."""
import argparse, json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "results/_scripts"))
from judge_agreement import _score, RUBRIC_DIMS
from run_qa_audit_opus import ClaudeLLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="sonnet")                  # claude -p alias
ap.add_argument("--judge_name", default="claude-sonnet-4.6")  # label in all_scores
ap.add_argument("--out_dir", default="results/audit/judge_agreement_p1")
ap.add_argument("--workers", type=int, default=50)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

OUT = ROOT / args.out_dir
work = [json.loads(l) for l in open(OUT / "prompts.jsonl")]
if args.limit:
    work = work[: args.limit]
print(f"[add_claude] {args.judge_name} (claude {args.model}): {len(work)} prompts, {args.workers}-way")
client = ClaudeLLM(args.model)
rows, lock, done, t0 = [], threading.Lock(), [0], time.time()


def do(w):
    sc = _score(client, w["prompt"])
    r = {"item_key": w["item_key"], "query_id": w["query_id"], "task_type": w["task_type"],
         "population": w["population"], "polarity": w["polarity"], "judge": args.judge_name, **sc}
    with lock:
        rows.append(r); done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(work):
            print(f"[add_claude]   {done[0]}/{len(work)} ({time.time()-t0:.0f}s)", flush=True)
    return r


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for f in as_completed([ex.submit(do, w) for w in work]):
        f.result()

if not args.limit:
    asf = OUT / "all_scores.jsonl"
    keep = [json.loads(l) for l in open(asf)]
    keep = [r for r in keep if r.get("judge") != args.judge_name]
    with open(asf, "w") as f:
        for r in keep + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

import numpy as np
allv = [r[d] for r in rows for d in RUBRIC_DIMS if r.get(d) is not None]
miss = sum(1 for r in rows if any(r.get(d) is None for d in RUBRIC_DIMS))
print(f"[add_claude] DONE {len(rows)} rows, {len(allv)} dim-scores, "
      f"mean={np.mean(allv):.2f}; rows missing a dim: {miss}")
