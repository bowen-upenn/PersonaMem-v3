#!/bin/bash
# Re-eval ONLY hidden_persona_recommendation after the surface-decoy slate regen.
# Per config: delete stale hp rows (backup under results/_hp_reeval_rowbackup/),
# then `run_eval --task hidden_persona_recommendation --resume` on the NEW slates.
# PARALLEL by endpoint group (no judge for this task, so only answer endpoints
# constrain us): Azure-gpt ‖ Google-gemini ‖ codex ‖ opus ‖ sonnet.
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
JUDGE=gpt-5.5
MATCHED="1 2 3 5 6 8 9 10 13 14"
CODEX_PERSONAS="1 2 3 5 6"
LOG=/tmp/hp_reeval; mkdir -p "$LOG"
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
b=p.replace("results/","results/_hp_reeval_rowbackup/",1)
os.makedirs(os.path.dirname(b),exist_ok=True)
if not os.path.exists(b): open(b,"w").write(open(p).read())
import csv as c
w=c.DictWriter(open(p,"w",newline=""),fieldnames=cols); w.writeheader(); w.writerows(keep)
PY
}

# LLM config: $1 cfg $2 mode $3 model $4 extra $5 personas $6 concurrency
run_llm(){
  local cfg=$1 mode=$2 model=$3 extra=$4 personas=$5 conc=$6
  echo "[$(ts)] START $cfg"
  for uid in $personas; do
    rd="results/$cfg/$uid"; [ -d "$rd" ] || continue
    del_rows "$rd/results.csv"
    python -u -m evaluation.run_eval --user_id "$uid" --backend_dir backend --run_dir "$rd" \
      --mode "$mode" --model "$model" --judge_model "$JUDGE" $extra \
      --task hidden_persona_recommendation --resume --workers 8 \
      > "$LOG/${cfg}.${uid}.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge "$conc" ]; do sleep 3; done
  done; wait
  echo "[$(ts)] DONE $cfg"
}

# Agentic config: $1 cfg $2 mode $3 modelarg $4 personas $5 concurrency
run_agent(){
  local cfg=$1 mode=$2 modelarg=$3 personas=$4 conc=$5
  echo "[$(ts)] START $cfg (agentic)"
  for uid in $personas; do
    rd="results/$cfg/$uid"; [ -d "$rd" ] || continue
    del_rows "$rd/results.csv"
    python -u -m evaluation.run_eval --user_id "$uid" --backend_dir backend --run_dir "$rd" \
      --mode "$mode" $modelarg --judge_model "$JUDGE" --prune_invalid \
      --task hidden_persona_recommendation --resume \
      > "$LOG/${cfg}.${uid}.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge "$conc" ]; do sleep 5; done
  done; wait
  echo "[$(ts)] DONE $cfg"
}

stream_gpt(){     # Azure gpt-5.5 — 3 gpt configs sequential, 3 personas each
  run_llm llm_longctx_gpt5.5_judged llm_longctx gpt-5.5 "" "$MATCHED" 3
  run_llm llm_memory_gpt5.5         llm_memory  gpt-5.5 "--memory_builder_model gpt-5.5" "$MATCHED" 3
  run_llm mem0_gpt5.5               mem0        gpt-5.5 "" "$MATCHED" 3
}
stream_gemini(){  # Google — 2 gemini configs sequential, 4 personas each
  run_llm llm_longctx_gemini3.5flash_judged llm_longctx gemini-3.5-flash "" "$MATCHED" 4
  run_llm llm_memory_gemini3.5flash_judged  llm_memory  gemini-3.5-flash "--memory_builder_model gemini-3.5-flash" "$MATCHED" 4
}
stream_codex(){   run_agent codex_agent_gpt5.5    codex_agent "--model gpt-5.5"      "$CODEX_PERSONAS" 3; }
stream_opus(){    run_agent agent_tools_opus4.8   agent_tools "--claude_model opus"  "$MATCHED" 1; }
stream_sonnet(){  run_agent agent_tools_sonnet4.6 agent_tools "--claude_model sonnet" "$MATCHED" 1; }

echo "[$(ts)] === hidden_persona re-eval START (parallel by endpoint) ==="
stream_gpt    > "$LOG/stream_gpt.log"    2>&1 &
stream_gemini > "$LOG/stream_gemini.log" 2>&1 &
stream_codex  > "$LOG/stream_codex.log"  2>&1 &
stream_opus   > "$LOG/stream_opus.log"   2>&1 &
stream_sonnet > "$LOG/stream_sonnet.log" 2>&1 &
wait
echo "[$(ts)] === hidden_persona re-eval COMPLETE ==="
