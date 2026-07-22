#!/bin/bash
# ============================================================================
# Batch persona generation — run the full pipeline for any set of users.
# ============================================================================
#
# Generates backend/<uid>/ for every requested user in the input CSV. Generate
# as many personas as you want, as long as the source data has the users
# (see scripts/download_gistbench.py to build the input CSV from
# https://huggingface.co/datasets/facebook/gistbench).
#
# Backend: Azure OpenAI
#   flagship  = gpt-5.5        (deployment AZURE_OPENAI_DEPLOYMENT_NAME)
#   mini-tier = gpt-5.4-mini   (deployment literally named "gpt-5.4-mini")
# Both read from .env (AZURE_OPENAI_ENDPOINT / _KEY / _API_VERSION).
#
# Properties:
#   * RESUMABLE  — skips any user whose backend/<uid>/profile.json already exists.
#   * PER-USER LOGS — /tmp/persona_regen/<uid>.{stdout,stderr} (CLAUDE.md convention).
#   * BOUNDED CONCURRENCY — CONCURRENCY users at a time (API quota friendly).
#   * No subset CSV is built (avoids csv round-trip footguns); each invocation
#     re-parses the input CSV and filters with --user_id. Parse overhead
#     (~seconds/user) is negligible next to LLM wall-time.
#
# !!! DO NOT run without an explicit go-ahead — LLM calls are expensive. !!!
#
# USAGE:
#   # calibration smoke: one user, watch cost + quality first
#   USERS="17" CONCURRENCY=1 bash scripts/run_persona_batch.sh
#
#   # all users in the input CSV that don't have a persona yet
#   bash scripts/run_persona_batch.sh
#
#   # knobs
#   INPUT_CSV=data/gistbench_sample_10users.csv bash scripts/run_persona_batch.sh
#   NUM_USERS=25 bash scripts/run_persona_batch.sh            # first N users only
#   USERS="17 18 19" bash scripts/run_persona_batch.sh        # explicit id list
#   CONCURRENCY=2 PARALLEL=30 bash scripts/run_persona_batch.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

INPUT_CSV="${INPUT_CSV:-data/gistbench_input.csv}"
MODEL="${MODEL:-gpt-5.5}"
MINI_MODEL="${MINI_MODEL:-gpt-5.4-mini}"
PARALLEL="${PARALLEL:-30}"        # internal concurrent LLM calls per user
CONCURRENCY="${CONCURRENCY:-3}"   # number of users processed simultaneously
RATE_LIMIT="${RATE_LIMIT:-50}"    # per-min budget passed to each QueryLLM client
LOGDIR="${LOGDIR:-/tmp/persona_regen}"
MASTER_LOG="$LOGDIR/_batch.master.log"

[ -f "$INPUT_CSV" ] || { echo "input CSV not found: $INPUT_CSV" >&2; exit 1; }

# Default user set: every distinct user_id in the input CSV (numeric sort).
# Schema: interaction_type,user_id,object_id,interaction_time,object_text
if [ -z "${USERS:-}" ]; then
  USERS="$(tail -n +2 "$INPUT_CSV" | cut -d, -f2 | sort -nu | tr '\n' ' ')"
  if [ -n "${NUM_USERS:-}" ]; then
    USERS="$(echo $USERS | tr ' ' '\n' | head -n "$NUM_USERS" | tr '\n' ' ')"
  fi
fi

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run_one() {
  local uid="$1"
  if [ -s "backend/$uid/profile.json" ]; then
    log "[skip] user $uid — backend/$uid/profile.json already exists"
    return 0
  fi
  log "[start] user $uid -> $LOGDIR/$uid.{stdout,stderr}"
  /usr/bin/time -v python scripts/run_persona_pipeline.py \
      --input_csv "$INPUT_CSV" --user_id "$uid" \
      --model "$MODEL" --mini_model "$MINI_MODEL" \
      --parallel "$PARALLEL" --rate_limit "$RATE_LIMIT" --verbose \
      > "$LOGDIR/$uid.stdout" 2> "$LOGDIR/$uid.stderr"
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -s "backend/$uid/profile.json" ]; then
    log "[done]  user $uid"
  else
    log "[FAIL]  user $uid (rc=$rc) — tail $LOGDIR/$uid.stderr"
  fi
}

log "=== persona batch: ${CONCURRENCY} concurrent, parallel=${PARALLEL}, model=${MODEL}/${MINI_MODEL}, input=${INPUT_CSV} ==="
n_total=$(echo $USERS | wc -w)
log "Requested $n_total users; resumable (existing profiles are skipped)."

# Bounded-concurrency dispatch using bash job control (wait -n).
for uid in $USERS; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
  run_one "$uid" &
done
wait

# Summary.
done_n=0; fail_n=0
for uid in $USERS; do
  if [ -s "backend/$uid/profile.json" ]; then done_n=$((done_n+1)); else fail_n=$((fail_n+1)); fi
done
log "=== persona batch complete: $done_n with profile.json, $fail_n missing (rerun to resume) ==="

# ----------------------------------------------------------------------------
# Recommended launch (background + live tmux dashboard tailing the master log):
#
#   nohup bash scripts/run_persona_batch.sh > /tmp/persona_regen/_batch.nohup 2>&1 &
#   tmux new-session -d -s personabatch -x 220 -y 50 \
#       "tail -F /tmp/persona_regen/_batch.master.log"
#   tmux attach -t personabatch   # read-only: add -r ; detach: Ctrl-b d
#
# Per-user detail lives in /tmp/persona_regen/<uid>.{stdout,stderr}.
# ----------------------------------------------------------------------------
