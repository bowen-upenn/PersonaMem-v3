#!/usr/bin/env bash
# 5-config evaluation matrix over the PersonaMem-v3 personas.
#
#   modes  : llm_longctx  llm_memory  mem0        (Azure gpt-5.5 baselines)
#            agent_tools   mcp_agent              (Claude Code, opus 4.8)
#            codex_agent                          (Codex CLI, GPT-5.5)
#   judge  : gpt-5.5 (Azure) for all
#   layout : results/{mode}/{uid}/results.csv   (+ per-(mode,uid) logs under results/_logs/)
#
# Usage:
#   scripts/run_eval_matrix.sh [--personas "1 2 3"] [--modes "llm_longctx mem0"] \
#       [--limit N] [--gpt-model gpt-5.5] [--claude-model opus] \
#       [--gpt-workers 8] [--agent-workers 1] [--jobs 1] [--no-resume]
#
# --limit N         cap rows per (mode,persona) — use for the smoke test.
# --jobs N          run up to N gpt-mode (mode,persona) jobs concurrently.
#                   Agent modes (agent_tools/mcp_agent) ALWAYS run one at a time
#                   to protect the Claude subscription + avoid overlay races.
# --no-resume       ignore prior partial results.csv (default: --resume).
#
# NOTE: this launches real LLM/eval runs (evaluation/run_eval.py). Per project
# rules it must only be run with explicit user approval.
set -uo pipefail
cd "$(dirname "$0")/.."

# Persona cohort is defined in a local, untracked file (see .gitignore);
# override per-run with --personas "...".
[ -f scripts/personas.local.sh ] && . scripts/personas.local.sh
PERSONAS="${PERSONAS:-${PERSONAS_EXTENDED:-}}"
MODES="llm_longctx llm_memory mem0 agent_tools mcp_agent"
GPT_MODEL="gpt-5.5"          # QueryLLM maps this to AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.5
CLAUDE_MODEL="opus"             # alias -> latest opus (opus-4.8)
JUDGE_MODEL="gpt-5.5"
GPT_WORKERS=8
AGENT_WORKERS=1
JOBS=1            # cross-persona concurrency for llm_memory (workers=8 intra-persona)
MEM0_JOBS=20     # cross-persona concurrency for mem0 — DEFAULT: the whole cohort at once.
                 # mem0 is workers=1 (serial per persona) AND rate-light (~1 in-flight
                 # call/persona, ~36% of the Azure cap at 6-way), so run them all in
                 # parallel; backoff absorbs any overflow. REQUIRES per-persona
                 # MEM0_DIR isolation (run_one) — else personas collide on the global
                 # ~/.mem0/migrations_qdrant lock (portalocker.AlreadyLocked).
AGENT_JOBS=1     # cross-persona concurrency for the opus agent modes
RATE_LIMIT=""    # per-client concurrency semaphore (run_eval --rate_limit); blank=default 50
LIMIT=""
RESUME="--resume"

while [ $# -gt 0 ]; do
  case "$1" in
    --personas)      PERSONAS="$2"; shift 2;;
    --modes)         MODES="$2"; shift 2;;
    --limit)         LIMIT="--limit $2"; shift 2;;
    --gpt-model)     GPT_MODEL="$2"; shift 2;;
    --claude-model)  CLAUDE_MODEL="$2"; shift 2;;
    --gpt-workers)   GPT_WORKERS="$2"; shift 2;;
    --agent-workers) AGENT_WORKERS="$2"; shift 2;;
    --jobs)          JOBS="$2"; shift 2;;
    --mem0-jobs)     MEM0_JOBS="$2"; shift 2;;
    --agent-jobs)    AGENT_JOBS="$2"; shift 2;;
    --rate-limit)    RATE_LIMIT="$2"; shift 2;;
    --resume)        RESUME="--resume"; shift 1;;
    --no-resume)     RESUME=""; shift 1;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$PERSONAS" ] || { echo "ERROR: set --personas \"...\", PERSONAS=..., or create scripts/personas.local.sh" >&2; exit 2; }

LOGDIR="results/_logs"
mkdir -p "$LOGDIR"
echo "[matrix] personas=[$PERSONAS]"
echo "[matrix] modes=[$MODES]  gpt_model=$GPT_MODEL  claude_model=$CLAUDE_MODEL  judge=$JUDGE_MODEL"
echo "[matrix] gpt_workers=$GPT_WORKERS agent_workers=$AGENT_WORKERS jobs=$JOBS limit='${LIMIT:-none}' resume='${RESUME:-off}'"

is_agent_mode() { case "$1" in agent_tools|mcp_agent|codex_agent) return 0;; *) return 1;; esac; }

run_one() {
  local mode="$1" uid="$2"
  local rundir="results/$mode/$uid"
  local out="$LOGDIR/$mode.$uid.stdout" err="$LOGDIR/$mode.$uid.stderr"
  mkdir -p "$rundir"
  # NOTE: the LLM judge is ON by default (--enable_llm_judge, BooleanOptionalAction);
  # pass --no-enable_llm_judge to disable. Do NOT pass a hyphenated variant.
  local args=(--user_id "$uid" --backend_dir backend --run_dir "$rundir"
              --mode "$mode" --judge_model "$JUDGE_MODEL"
              --memory_token_cap 4096)
  [ -n "$RESUME" ] && args+=($RESUME)
  [ -n "$LIMIT" ]  && args+=($LIMIT)
  [ -n "$RATE_LIMIT" ] && args+=(--rate_limit "$RATE_LIMIT")
  if is_agent_mode "$mode"; then
    if [ "$mode" = "codex_agent" ]; then
      args+=(--model "$GPT_MODEL" --workers "$AGENT_WORKERS")
    else
      args+=(--claude_model "$CLAUDE_MODEL" --workers "$AGENT_WORKERS")
    fi
  else
    args+=(--model "$GPT_MODEL" --workers "$GPT_WORKERS")
  fi
  echo "[matrix] START $mode/$uid -> $rundir  (log: $out)"
  # mem0 keeps a GLOBAL store at $HOME/.mem0 (migrations_qdrant) that portalocker-locks;
  # concurrent personas collide (AlreadyLocked). Isolate MEM0_DIR per persona so
  # MEM0_JOBS>1 is safe. (run_one runs in a backgrounded subshell, so this export is
  # per-persona.) Does NOT touch HOME (that breaks the Python env).
  if [ "$mode" = "mem0" ]; then export MEM0_DIR="$rundir/.mem0dir"; mkdir -p "$MEM0_DIR"; else unset MEM0_DIR; fi
  if python -m evaluation.run_eval "${args[@]}" >"$out" 2>"$err"; then
    echo "[matrix] DONE  $mode/$uid"
  else
    echo "[matrix] FAIL  $mode/$uid (exit $?) — see $err" >&2
  fi
}

for mode in $MODES; do
  if is_agent_mode "$mode"; then
    # Agent modes: $AGENT_JOBS personas concurrent. Each persona has its OWN
    # time-masked snapshot + write-overlay + run_dir + claude (opus) process, so
    # cross-persona parallelism is race-free. agent_workers stays intra-persona
    # (mcp_agent writes accumulate per persona). Concurrency is bounded by what
    # the Claude subscription tolerates.
    echo "[matrix] $mode: $AGENT_JOBS persona(s) concurrent x $AGENT_WORKERS worker(s)"
    running=0
    for uid in $PERSONAS; do
      run_one "$mode" "$uid" &
      running=$((running+1))
      if [ "$running" -ge "$AGENT_JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
    done
    wait
  else
    # GPT modes. mem0 is intra-persona SERIAL (workers=1, unpicklable qdrant
    # store), so parallelize it across personas with $JOBS (each persona has its
    # own store dir — no conflict). The long-context / text-memory modes already
    # parallelize intra-persona via $GPT_WORKERS, so run their personas one at a
    # time — otherwise $JOBS×$GPT_WORKERS would blow past the Azure rate limit.
    # mem0 AND llm_memory both have a SERIAL per-persona memory BUILD phase (the
    # bottleneck), so parallelize them across personas with $JOBS (each persona =
    # own process + own independent builder client). Keep $GPT_WORKERS low for
    # these (jobs x workers = concurrency) so the answer phase stays within the
    # rate limit. llm_longctx (no build) keeps jobs=1 x high workers.
    case "$mode" in
      mem0)       mode_jobs="$MEM0_JOBS";;   # workers=1 + rate-light → parallelize hard
      llm_memory) mode_jobs="$JOBS";;        # workers=8 intra-persona → keep jobs low
      *)          mode_jobs=1;;
    esac
    echo "[matrix] $mode: $mode_jobs persona(s) concurrent x $GPT_WORKERS worker(s)"
    running=0
    for uid in $PERSONAS; do
      run_one "$mode" "$uid" &
      running=$((running+1))
      if [ "$running" -ge "$mode_jobs" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
    done
    wait
  fi
done

echo "[matrix] all jobs finished — aggregating"
python scripts/aggregate_eval.py --results_root results --modes "$(echo "$MODES" | tr ' ' ',')"
echo "[matrix] done. cross-mode comparison: results/aggregate/comparison.csv"
