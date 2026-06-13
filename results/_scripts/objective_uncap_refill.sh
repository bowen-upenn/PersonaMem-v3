#!/usr/bin/env bash
# Refill turn-cap-truncated objective rows (2026-06-12 audit): agent burned its
# turn budget on search/read and never emitted the final ranking → empty result
# scored a hard 0. 30 opus + 2 sonnet objective rows. Strip ONLY those empty
# rows (backup first), then --resume with caps raised so high they don't bind
# (the 600s subprocess timeout stays as the real safety bound). The permanent
# opus 1.5x turn factor (claude_subagent.py) is also now in effect.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
BK=results/_objcap_backup_20260612
ts(){ date +%H:%M:%S; }

# Effectively-uncapped for this refill: ~180 opus turns (120 base ×1.5), ample $.
export EVAL_AGENT_MAX_TURNS=120
export EVAL_AGENT_HEAVY_TURNS=120
export EVAL_AGENT_MAX_BUDGET_USD=2.50
export EVAL_AGENT_HEAVY_BUDGET_USD=2.50

strip_empty_objective(){  # $1 = results.csv -> prints n stripped
  python - "$1" <<'PY'
import csv, os, sys
csv.field_size_limit(10**9)
OBJ={"personalized_recommendation","hidden_persona_recommendation",
     "at_ai_directive_followup","active_mistake_prevention",
     "local_recommendation_geo_shift"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status",
      "duration_ms","error","agent_response"]
f=sys.argv[1]; rows=list(csv.DictReader(open(f)))
keep=[r for r in rows if not (r["task_type"] in OBJ and not (r.get("agent_response") or "").strip())]
if len(keep)<len(rows):
    tmp=f+".tmp"
    with open(tmp,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
        for r in keep: w.writerow(r)
    os.replace(tmp,f)
print(len(rows)-len(keep),end="")
PY
}

run_one(){  # $1=suffix $2=model $3=uid
  local suf="$1" model="$2" uid="$3" d log
  d="results/agent_tools_${suf}/$uid"
  log="results/_logs/objcap.${suf}.$uid.stdout"
  : > "$log"; : > "results/_logs/objcap.${suf}.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode agent_tools --claude_model "$model" --judge_model gpt-5.5 \
    --workers 4 --resume > "$log" 2> "results/_logs/objcap.${suf}.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    grep -q "wrote .*summary.json" "$log" 2>/dev/null && { sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break; }
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[objcap $(ts)] done ${suf}/$uid"
}

# suffix:model:personas-with-empty-objective-rows
OPUS_P="2 3 5 6 8 9 10 13 14"
SONNET_P="1 5"

echo "[objcap $(ts)] phase 0: backup + strip"
total=0
for spec in "opus4.8:$OPUS_P" "sonnet4.6:$SONNET_P"; do
  suf="${spec%%:*}"; ps="${spec#*:}"
  for uid in $ps; do
    d="results/agent_tools_${suf}/$uid"
    [ -f "$d/results.csv" ] || { echo "[objcap $(ts)] MISSING $d — skip"; continue; }
    mkdir -p "$BK/agent_tools_${suf}/$uid"
    [ -f "$BK/agent_tools_${suf}/$uid/results.csv" ] || cp "$d/results.csv" "$BK/agent_tools_${suf}/$uid/results.csv"
    n=$(strip_empty_objective "$d/results.csv")
    total=$((total+n)); echo "[objcap $(ts)] stripped ${suf}/$uid: $n"
  done
done
echo "[objcap $(ts)] phase 0 done: $total stripped (expect 32)"

echo "[objcap $(ts)] phase 1: refill opus then sonnet (sequential, uncapped)"
for uid in $OPUS_P;   do run_one opus4.8  opus   "$uid"; done
for uid in $SONNET_P; do run_one sonnet4.6 sonnet "$uid"; done
echo "[objcap $(ts)] OBJECTIVE UNCAP REFILL DONE"
