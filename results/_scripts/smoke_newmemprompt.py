import os, sys, glob, time
os.environ["EVAL_GEMINI_BATCH"]="0"
REPO="/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"; sys.path.insert(0,REPO)
from evaluation.backend_query import BackendQuery
from evaluation.memory_builder import build_checkpoints, default_memory_config
from query_llm import QueryLLM
u="1"
bq=BackendQuery(f"{REPO}/backend")
bnd=sorted(int(f.split('_T')[-1].split('.')[0]) for f in glob.glob(f"{REPO}/results/llm_memory_gpt5.5/{u}/memory_states/{u}_llm_memory_T*.json"))
print(f"[smoke] persona {u}: {len(bnd)} day-boundaries, NEW minimal prompt, both models", flush=True)
for model in ["gemini-3.5-flash","gpt-5.5"]:
    t0=time.time()
    client=QueryLLM({"models":{"llm_model":model}}, rate_limit_per_min=50)
    cfg=default_memory_config(); cfg["builder_model"]=model
    rd=f"{REPO}/results/_smoke_newmemprompt/{model.replace('.','_').replace('-','_')}/{u}"; os.makedirs(rd,exist_ok=True)
    led=build_checkpoints(bq,u,bnd,client,cfg,algo="llm_memory",run_dir=rd)
    mem=led.checkpoints[max(led.checkpoints)]; bs=led.build_stats
    print(f"\n{'#'*92}\n# {model}: {len(mem)} chars, {mem.count(chr(10)+'-')} '-'lines, in={bs.get('input_tokens'):,} out={bs.get('output_tokens'):,} calls={bs.get('calls')}, {time.time()-t0:.0f}s\n{'#'*92}", flush=True)
    print(mem, flush=True)
print("\n[smoke] DONE", flush=True)
