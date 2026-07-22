#!/usr/bin/env python3
"""Build gemini-3.5-flash's OWN textual memory (not reusing gpt5.5 prebuilt),
for the evaluated personas. Boundaries come from the existing gpt5.5
memory_states so the build is apples-to-apples. Captures real build tokens +
gemini cost. Writes nothing outside results/_gemini_own_memory/.

Per-user builds are independent → parallelizable across users (each user's
own walk stays sequential because each step depends on the prior memory).
"""
import os, sys, glob, json, argparse, time
os.environ["EVAL_GEMINI_BATCH"] = "0"   # force synchronous, standard pricing
from concurrent.futures import ThreadPoolExecutor, as_completed
REPO = "/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"
sys.path.insert(0, REPO)
from evaluation.backend_query import BackendQuery
from evaluation.memory_builder import build_checkpoints, default_memory_config
from evaluation.cost_model import gemini_cost
from query_llm import QueryLLM

ALL = sorted({int(u) for u in os.environ.get("PERSONAS", "").split()} or
             {int(u) for u in os.listdir("results/agent_tools_opus4.8") if u.isdigit()})
MODEL = "gemini-3.5-flash"
OUTBASE = f"{REPO}/results/_gemini_own_memory"

def boundaries_for(u):
    fs = glob.glob(f"{REPO}/results/llm_memory_gpt5.5/{u}/memory_states/{u}_llm_memory_T*.json")
    return sorted(int(f.split("_T")[-1].split(".")[0]) for f in fs)

def disk_stats(u, root):
    """Read cumulative build_stats from a user's max-T checkpoint."""
    fs = glob.glob(f"{root}/{u}/memory_states/{u}_llm_memory_T*.json")
    if not fs: return None
    last = max(fs, key=lambda f: int(f.split("_T")[-1].split(".")[0]))
    d = json.load(open(last))
    return {"in": d.get("build_input_tokens", 0), "out": d.get("build_output_tokens", 0),
            "calls": d.get("build_calls", 0)}

def build_one(u):
    uid = str(u); bnd = boundaries_for(u)
    bq = BackendQuery(f"{REPO}/backend")                 # own bq per thread (cache safety)
    client = QueryLLM({"models": {"llm_model": MODEL}}, rate_limit_per_min=50)
    cfg = default_memory_config(); cfg["builder_model"] = MODEL
    rd = f"{OUTBASE}/{uid}"; os.makedirs(rd, exist_ok=True)
    t0 = time.time()
    led = build_checkpoints(bq, uid, bnd, client, cfg, algo="llm_memory", run_dir=rd)
    bs = led.build_stats
    stats = {"in": bs.get("input_tokens", 0), "out": bs.get("output_tokens", 0), "calls": bs.get("calls", 0)}
    stats["cost"] = gemini_cost(MODEL, stats["in"], stats["out"]); stats["sec"] = time.time() - t0
    g = disk_stats(u, f"{REPO}/results/llm_memory_gpt5.5")
    print(f"[build] u{uid}: gem in={stats['in']:,} out={stats['out']:,} calls={stats['calls']} "
          f"${stats['cost']:.3f} | gpt out was {g['out']:,} ({stats['sec']:.0f}s)", flush=True)
    return uid, stats

def aggregate():
    grand = {"in": 0, "out": 0, "calls": 0}; per = {}
    for u in ALL:
        s = disk_stats(u, OUTBASE)
        if not s: print(f"[agg] WARNING user {u} has no gemini build on disk"); continue
        per[str(u)] = s; grand["in"] += s["in"]; grand["out"] += s["out"]; grand["calls"] += s["calls"]
    std = gemini_cost(MODEL, grand["in"], grand["out"]); bat = gemini_cost(MODEL, grand["in"], grand["out"], batch=True)
    out = {"model": MODEL, "n_users": len(per), "per_user": per, "totals": grand,
           "cost_std": round(std, 3), "cost_batch": round(bat, 3)}
    json.dump(out, open(f"{OUTBASE}/build_cost.json", "w"), indent=2)
    print(f"\n=== GEMINI OWN MEMORY BUILD — {len(per)} users (from disk) ===")
    print(f"input={grand['in']:,}  output={grand['out']:,}  calls={grand['calls']}")
    print(f"cost: standard=${std:.3f}   batch=${bat:.3f}")
    print(f"[wrote] {OUTBASE}/build_cost.json")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default=",".join(map(str, ALL)))
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--max_workers", type=int, default=9)
    ap.add_argument("--aggregate_only", action="store_true")
    a = ap.parse_args()
    if a.aggregate_only:
        aggregate(); return
    users = [int(x) for x in a.users.split(",") if x.strip()]
    if a.parallel:
        with ThreadPoolExecutor(max_workers=a.max_workers) as ex:
            futs = {ex.submit(build_one, u): u for u in users}
            for f in as_completed(futs):
                try: f.result()
                except Exception as e: print(f"[build] u{futs[f]} FAILED: {e!r}", flush=True)
    else:
        for u in users: build_one(u)
    aggregate()   # always aggregate ALL 10 from disk (includes earlier user-1 build)

if __name__ == "__main__":
    main()
