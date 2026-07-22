#!/usr/bin/env bash
# Proactive judge-replay under the NEW rubric (fa361bd) for all non-agent_tools
# configs. NO generation (--replay_from disables the model); judge ON gpt-5.5
# for ALL configs including gemini (user directive). NO Claude Code / no
# subscription. Idempotent: personas whose proactive rows already carry the
# new-rubric key are skipped, so re-running after a crash is cheap.
# Lives under results/_scripts (NOT /tmp — tmp cleaner killed a prior run by
# deleting the script under the running bash).
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
SNAP=results/_replay_src
CONC=45
ts(){ date +%H:%M:%S; }

strip_proactive(){  # $1 = live results.csv
  python - "$1" <<'PY'
import csv, os, sys
csv.field_size_limit(10_000_000)
PRO={"proactive_close_friend_update","proactive_friend_feed_react",
     "proactive_trending_feed_react","proactive_overactive_check",
     "restraint_sensitive_event_silence"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status",
      "duration_ms","error","agent_response"]
f=sys.argv[1]
rows=list(csv.DictReader(open(f)))
keep=[r for r in rows if r["task_type"] not in PRO]
if len(keep)<len(rows):
    tmp=f+".tmp"
    with open(tmp,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
        for r in keep: w.writerow(r)
    os.replace(tmp,f)
print(f"stripped {len(rows)-len(keep)}", end="")
PY
}

run_one(){  # $1=run_mode_dir $2=mode $3=model $4=replay_root $5=uid
  local mdir="$1" mode="$2" model="$3" rroot="$4" uid="$5"
  local d="results/$mdir/$uid"
  [ -f "$d/results.csv" ] || { echo "[replay $(ts)] skip $mdir/$uid (no results)"; return; }
  mkdir -p "$SNAP/$mdir/$uid"
  [ -f "$SNAP/$mdir/$uid/results.csv" ] || cp "$d/results.csv" "$SNAP/$mdir/$uid/results.csv"
  # Idempotence: skip when every snapshot proactive row already exists live
  # WITH the new-rubric key (decision_score).
  local needs
  needs=$(python - "$d/results.csv" "$rroot/$uid/results.csv" <<'PY'
import csv, sys
csv.field_size_limit(10_000_000)
PRO={"proactive_close_friend_update","proactive_friend_feed_react",
     "proactive_trending_feed_react","proactive_overactive_check",
     "restraint_sensitive_event_silence"}
live, src = sys.argv[1], sys.argv[2]
try: want=sum(1 for r in csv.DictReader(open(src)) if r["task_type"] in PRO)
except FileNotFoundError: want=-1
have=0
for r in csv.DictReader(open(live)):
    if r["task_type"] in PRO and '"decision_score"' in (r.get("metrics_json") or ""):
        have+=1
print(0 if (want>=0 and have>=want) else 1, end="")
PY
)
  if [ "$needs" = "0" ]; then echo "[replay $(ts)] skip $mdir/$uid (already new rubric)"; return; fi
  local n; n=$(strip_proactive "$d/results.csv")
  local log="results/_logs/replay.$mdir.$uid.stdout"
  : > "$log"; : > "results/_logs/replay.$mdir.$uid.stderr"
  MEM0_DIR="$d/.mem0dir" python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode "$mode" --model "$model" --judge_model gpt-5.5 \
    --resume --replay_from "$rroot" \
    > "$log" 2> "results/_logs/replay.$mdir.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "wrote .*summary.json" "$log" 2>/dev/null; then
      sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[replay $(ts)] done $mdir/$uid ($n)"
}

P20="1 2 3 5 6 8 9 10 13 14 26 105 115 209 229 282 461 655 760 835"
P10="1 2 3 5 6 8 9 10 13 14"

# mdir|mode|model|replay_root|personas — _judged configs replay from the raw
# SNAPSHOTS (immutable), never the live raw dirs.
CFGS=(
  "llm_longctx_gpt5.5|llm_longctx|gpt-5.5|$SNAP/llm_longctx_gpt5.5|$P20"
  "llm_memory_gpt5.5|llm_memory|gpt-5.5|$SNAP/llm_memory_gpt5.5|$P20"
  "mem0_gpt5.5|mem0|gpt-5.5|$SNAP/mem0_gpt5.5|$P20"
  "llm_longctx_gemini3.5flash|llm_longctx|gemini-3.5-flash|$SNAP/llm_longctx_gemini3.5flash|$P10"
  "llm_memory_gemini3.5flash|llm_memory|gemini-3.5-flash|$SNAP/llm_memory_gemini3.5flash|$P10"
  "llm_longctx_gpt5.5_judged|llm_longctx|gpt-5.5|$SNAP/llm_longctx_gpt5.5|$P10"
  "llm_longctx_gemini3.5flash_judged|llm_longctx|gemini-3.5-flash|$SNAP/llm_longctx_gemini3.5flash|$P10"
  "llm_memory_gemini3.5flash_judged|llm_memory|gemini-3.5-flash|$SNAP/llm_memory_gemini3.5flash|$P10"
)

# Phase 0: all raw snapshots upfront (only if absent — never overwrite).
for cfg in "${CFGS[@]}"; do
  IFS='|' read -r mdir mode model rroot personas <<< "$cfg"
  case "$mdir" in *_judged) continue;; esac
  for uid in $personas; do
    [ -f "results/$mdir/$uid/results.csv" ] || continue
    mkdir -p "$SNAP/$mdir/$uid"
    [ -f "$SNAP/$mdir/$uid/results.csv" ] || cp "results/$mdir/$uid/results.csv" "$SNAP/$mdir/$uid/results.csv"
  done
done
echo "[replay $(ts)] snapshots ready: $(find $SNAP -name results.csv | wc -l) files"

echo "[replay $(ts)] proactive judge-replay: 8 configs, CONC=$CONC, no generation"
running=0
for cfg in "${CFGS[@]}"; do
  IFS='|' read -r mdir mode model rroot personas <<< "$cfg"
  for uid in $personas; do
    run_one "$mdir" "$mode" "$model" "$rroot" "$uid" &
    running=$((running+1))
    if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
  done
done
wait
echo "[replay $(ts)] PROACTIVE REPLAY DONE"
