#!/usr/bin/env bash
# Re-run over_personalization_repetition_chatbot + personal_qa_hallucination
# for agent_tools OPUS ONLY (first-10) WITHOUT the binding budget/turn caps.
# (Sonnet skipped per user: its rows for these two tasks were 1/20 and 2/68
# capped — already effectively uncapped.)
# Caps are lifted via env (read at claude_subagent import) — no code edits, so
# no collision with concurrent sessions. Backstops stay as runaway protection:
#   budget $5.00 (was 0.30/0.60 sonnet-baseline; opus ×5/3 -> $8.33)
#   turns 60/60 (was 15/30)
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
export EVAL_AGENT_MAX_BUDGET_USD=5.00
export EVAL_AGENT_HEAVY_BUDGET_USD=5.00
export EVAL_AGENT_MAX_TURNS=60
export EVAL_AGENT_HEAVY_TURNS=60
ts(){ date +%H:%M:%S; }

echo "[uncap $(ts)] stripping target-task rows from both models"
python - <<'PY'
import csv, os
csv.field_size_limit(10_000_000)
TARGET={"over_personalization_repetition_chatbot","personal_qa_hallucination"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status","duration_ms","error","agent_response"]
tot=0
for suf in ("opus4.8",):
    for u in (1,2,3,5,6,8,9,10,13,14):
        f=f"results/agent_tools_{suf}/{u}/results.csv"
        if not os.path.exists(f): continue
        rows=list(csv.DictReader(open(f)))
        keep=[r for r in rows if r["task_type"] not in TARGET]
        n=len(rows)-len(keep)
        if n:
            tmp=f+".tmp"
            with open(tmp,"w",newline="",encoding="utf-8") as fh:
                w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
                for r in keep: w.writerow(r)
            os.replace(tmp,f); tot+=n
print(f"stripped {tot} rows (repetition_chatbot + personal_qa_hallucination)")
PY

run_one(){  # $1=suffix $2=model $3=uid
  local suf="$1" model="$2" uid="$3"
  local d="results/agent_tools_${suf}/$uid"
  local log="results/_logs/uncap.$suf.$uid.stdout"
  : > "$log"; : > "results/_logs/uncap.$suf.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode agent_tools --claude_model "$model" --judge_model gpt-5.5 \
    --workers 4 --resume > "$log" 2> "results/_logs/uncap.$suf.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    grep -q "wrote .*summary.json" "$log" 2>/dev/null && { sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break; }
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[uncap $(ts)] done ${suf}/$uid"
}

CONC=3
running=0
for uid in 1 2 3 5 6 8 9 10 13 14; do
  run_one opus4.8 opus "$uid" &
  running=$((running+1))
  if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[uncap $(ts)] aggregating..."
python scripts/aggregate_eval.py --results_root results \
  --modes llm_longctx_gpt5.5_judged,llm_memory_gpt5.5,mem0_gpt5.5,llm_longctx_gemini3.5flash_judged,llm_memory_gemini3.5flash_judged,agent_tools_sonnet4.6,agent_tools_opus4.8 \
  > results/_logs/aggregate.uncap.log 2>&1 || true
echo "[uncap $(ts)] UNCAPPED RERUN DONE"
