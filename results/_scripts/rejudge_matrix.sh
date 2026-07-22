#!/usr/bin/env bash
# Re-score the existing matrix under the calibrated judge anchors (9f0ad5a):
# scripts/rejudge_existing.py --write_back per config, Azure gpt-5.5 judge,
# saved responses only (no model-under-test calls). Sequential configs so the
# single QueryLLM 50/min limiter is the only Azure consumer.
#
# Scope = pr.score (unified rubric) tasks only — the anchor change touched
# _JUDGE_PREFACE + helpfulness/preference_alignment/voice_match dim defs;
# tasks with their own judges (proactive_*, implicit_qa, pqa, sycophancy,
# preference_shift, repetition) are untouched and stay comparable.
#
# agent_tools_opus4.8 + codex_agent_gpt5.5 are DEFERRED (user directive):
# both are being actively written by another session; re-judge them with this
# same script (just edit CFGS) once their generation passes finish.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }

CFGS=(
  agent_tools_sonnet4.6
  llm_longctx_gpt5.5_judged
  llm_memory_gpt5.5
  mem0_gpt5.5
  llm_longctx_gemini3.5flash_judged
  llm_memory_gemini3.5flash_judged
)

# All configs in PARALLEL (user directive) — each process carries its own
# 50/min QueryLLM limiter (~300/min aggregate; the Jun-12 proactive replay ran
# 45 such processes against the same deployment without throttling). Each
# config touches only its own results dir, so there is no write contention.
run_cfg(){
  local cfg="$1"
  local users
  users=$(ls "results/$cfg" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | paste -sd,)
  [ -n "$users" ] || { echo "[rejudge $(ts)] $cfg: no persona dirs — skip"; return; }
  local log="results/_logs/rejudge.$cfg.log"
  : > "$log"
  echo "[rejudge $(ts)] start $cfg users=$users"
  python -u scripts/rejudge_existing.py \
    --results_dir "results/$cfg" --users "$users" \
    --judge_model gpt-5.5 --workers 8 --write_back \
    --out "/tmp/eval_regen/rejudge_summary.$cfg.json" \
    >> "$log" 2>&1
  echo "[rejudge $(ts)] done $cfg (summary: /tmp/eval_regen/rejudge_summary.$cfg.json)"
}

for cfg in "${CFGS[@]}"; do run_cfg "$cfg" & done
wait
echo "[rejudge $(ts)] REJUDGE MATRIX DONE"
