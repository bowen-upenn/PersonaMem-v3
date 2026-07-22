import os, sys, glob, json, time, argparse
os.environ["EVAL_GEMINI_BATCH"]="0"   # sequential build → sync (batch would stall dependent steps)
from concurrent.futures import ThreadPoolExecutor, as_completed
REPO="/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"; sys.path.insert(0,REPO)
from evaluation.backend_query import BackendQuery
from evaluation.memory_builder import build_checkpoints, default_memory_config
from evaluation.cost_model import gemini_cost
from query_llm import QueryLLM
PERS=[1,2,3,5,6,8,9,10,13,14]
def boundaries(u):
    fs=glob.glob(f"{REPO}/results/llm_memory_gpt5.5/{u}/memory_states/{u}_llm_memory_T*.json")
    return sorted(int(f.split('_T')[-1].split('.')[0]) for f in fs)
def build_one(model,u):
    bq=BackendQuery(f"{REPO}/backend")
    client=QueryLLM({"models":{"llm_model":model}}, rate_limit_per_min=50)
    cfg=default_memory_config(); cfg["builder_model"]=model
    rd=f"{REPO}/results/_reeval_newmem/{model.replace('.','_').replace('-','_')}/{u}"; os.makedirs(rd,exist_ok=True)
    t0=time.time()
    led=build_checkpoints(bq,str(u),boundaries(u),client,cfg,algo="llm_memory",run_dir=rd)
    bs=led.build_stats
    print(f"[rebuild] {model} u{u}: in={bs.get('input_tokens',0):,} out={bs.get('output_tokens',0):,} calls={bs.get('calls')} ({time.time()-t0:.0f}s)",flush=True)
    return model,bs
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); a=ap.parse_args()
tot={"in":0,"out":0}
with ThreadPoolExecutor(max_workers=5) as ex:
    futs={ex.submit(build_one,a.model,u):u for u in PERS}
    for f in as_completed(futs):
        try:_,bs=f.result(); tot["in"]+=bs.get("input_tokens",0); tot["out"]+=bs.get("output_tokens",0)
        except Exception as e: print(f"[rebuild] FAIL {a.model}/{futs[f]}: {e!r}",flush=True)
print(f"\n[rebuild] {a.model} TOTAL: in={tot['in']:,} out={tot['out']:,}",flush=True)
if "gemini" in a.model: print(f"  gemini build cost ~${gemini_cost(a.model,tot['in'],tot['out']):.2f}",flush=True)
print(f"[rebuild] {a.model} DONE",flush=True)
