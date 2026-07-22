#!/bin/bash
# Re-eval hidden_persona_recommendation after the harder-slate regen, for the 7
# configs runnable here (codex EXCLUDED — its CLI isn't on PATH in this env).
# Delete stale hp rows (backup), then run_eval --task ... --resume on new slates.
# Parallel by endpoint: gpt ‖ gemini ‖ mem0(MEM0_DIR-isolated) ‖ opus ‖ sonnet.
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
JUDGE=gpt-5.5
MATCHED="1 2 3 5 6 8 9 10 13 14"
LOG=/tmp/hp_hardreeval; mkdir -p "$LOG"
ts(){ date +%H:%M:%S; }

del_rows(){
  python3 - "$1" <<'PY'
import csv,sys,os
csv.field_size_limit(2**31-1)
p=sys.argv[1]
if not os.path.exists(p): raise SystemExit
rows=list(csv.DictReader(open(p)))
if not rows: raise SystemExit
cols=list(rows[0].keys())
keep=[r for r in rows if r.get("task_type")!="hidden_persona_recommendation"]
if len(keep)==len(rows): raise SystemExit
b=p.replace("results/","results/_hp_hardreeval_backup/",1)
os.makedirs(os.path.dirname(b),exist_ok=True)
if not os.path.exists(b): open(b,"w").write(open(p).read())
import csv as c
w=c.DictWriter(open(p,"w",newline=""),fieldnames=cols); w.writeheader(); w.writerows(keep)
PY
}
run_llm(){ # cfg mode model extra conc
  local cfg=$1 mode=$2 model=$3 extra=$4 conc=$5
  echo "[$(ts)] START $cfg"
  for u in $MATCHED; do
    rd="results/$cfg/$u"; [ -d "$rd" ] || continue
    del_rows "$rd/results.csv"
    python -u -m evaluation.run_eval --user_id "$u" --backend_dir backend --run_dir "$rd" \
      --mode "$mode" --model "$model" --judge_model "$JUDGE" $extra \
      --task hidden_persona_recommendation --resume --workers 8 > "$LOG/${cfg}.$u.log" 2>&1 &
    while [ "$(jobs -r|wc -l)" -ge "$conc" ]; do sleep 3; done
  done; wait; echo "[$(ts)] DONE $cfg"
}
run_mem0(){ # MEM0_DIR-isolated, parallel users
  local cfg=mem0_gpt5.5
  echo "[$(ts)] START $cfg (MEM0_DIR-isolated)"
  for u in $MATCHED; do
    rd="results/$cfg/$u"; [ -d "$rd" ] || continue
    del_rows "$rd/results.csv"; D=/tmp/mem0dir_re_$u; mkdir -p "$D"
    MEM0_DIR="$D" python -u -m evaluation.run_eval --user_id "$u" --backend_dir backend --run_dir "$rd" \
      --mode mem0 --model gpt-5.5 --judge_model "$JUDGE" \
      --task hidden_persona_recommendation --resume --workers 8 > "$LOG/${cfg}.$u.log" 2>&1 &
    while [ "$(jobs -r|wc -l)" -ge 8 ]; do sleep 3; done
  done; wait; echo "[$(ts)] DONE $cfg"
}
run_agent(){ # cfg claude_model
  local cfg=$1 cm=$2
  echo "[$(ts)] START $cfg (agentic)"
  for u in $MATCHED; do
    rd="results/$cfg/$u"; [ -d "$rd" ] || continue
    del_rows "$rd/results.csv"
    python -u -m evaluation.run_eval --user_id "$u" --backend_dir backend --run_dir "$rd" \
      --mode agent_tools --claude_model "$cm" --judge_model "$JUDGE" --prune_invalid \
      --task hidden_persona_recommendation --resume > "$LOG/${cfg}.$u.log" 2>&1 || true
  done; echo "[$(ts)] DONE $cfg"
}
gpt(){    run_llm llm_longctx_gpt5.5_judged llm_longctx gpt-5.5 "" 3
          run_llm llm_memory_gpt5.5 llm_memory gpt-5.5 "--memory_builder_model gpt-5.5" 3; }
gemini(){ run_llm llm_longctx_gemini3.5flash_judged llm_longctx gemini-3.5-flash "" 4
          run_llm llm_memory_gemini3.5flash_judged llm_memory gemini-3.5-flash "--memory_builder_model gemini-3.5-flash" 4; }

echo "[$(ts)] === hp re-eval (7 configs) START ==="
gpt        > "$LOG/stream_gpt.log" 2>&1 &
gemini     > "$LOG/stream_gemini.log" 2>&1 &
run_mem0   > "$LOG/stream_mem0.log" 2>&1 &
run_agent agent_tools_opus4.8 opus     > "$LOG/stream_opus.log" 2>&1 &
run_agent agent_tools_sonnet4.6 sonnet > "$LOG/stream_sonnet.log" 2>&1 &
wait
echo "[$(ts)] === hp re-eval COMPLETE ==="
