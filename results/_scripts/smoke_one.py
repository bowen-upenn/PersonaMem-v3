import os, sys, time
os.environ["EVAL_GEMINI_BATCH"]="0"
REPO="/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"; sys.path.insert(0,REPO)
from evaluation.backend_query import BackendQuery
from evaluation.memory_builder import update_step, build_global_stream, EMPTY_MEMORY, DEFAULT_MEMORY_TOKEN_CAP
from query_llm import QueryLLM
u="1"
bq=BackendQuery(f"{REPO}/backend")
rows=build_global_stream(bq,u,10**12)
chunk=rows[:60]   # ONE example chunk
print(f"[smoke] persona {u}: one update_step on {len(chunk)} events (of {len(rows)}), NEW prompt\n", flush=True)
print("=== rendered prompt (first 900 chars) ===")
from evaluation.prompts import llm_memory_update_prompt
from evaluation.memory_builder import _render_chunk
print(llm_memory_update_prompt(EMPTY_MEMORY,"",_render_chunk(chunk),token_cap=DEFAULT_MEMORY_TOKEN_CAP)[:900], flush=True)
for model in ["gemini-3.5-flash","gpt-5.5"]:
    t0=time.time()
    client=QueryLLM({"models":{"llm_model":model}}, rate_limit_per_min=50)
    nm,ns,it,ot=update_step(EMPTY_MEMORY,"",chunk,client,model=model,token_cap=DEFAULT_MEMORY_TOKEN_CAP)
    print(f"\n{'#'*90}\n# {model}: {len(nm)} chars, in={it} out={ot}, {time.time()-t0:.0f}s\n{'#'*90}", flush=True)
    print(nm, flush=True)
print("\n[smoke] DONE", flush=True)
