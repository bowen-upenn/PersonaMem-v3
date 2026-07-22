#!/usr/bin/env bash
# Recovery for mem0: run the 10 personas SERIALLY so the per-user qdrant local
# stores never lock-contend (the parallel orchestrator run hit
# portalocker.AlreadyLocked). Each persona rebuilds its store cleanly, then
# evals only the new chatbot_personalized_response rows. --retry_failed clears
# any partial/errored CPR rows left by the failed parallel attempt.
set -u
cd "$(dirname "$0")/.."
mkdir -p /tmp/cpr_eval
for uid in 1 2 3 5 6 8 9 10 13 14; do
  echo "=== mem0 persona $uid $(date +%H:%M:%S) ==="
  python -m evaluation.run_eval --user_id "$uid" --backend_dir backend \
    --run_dir "results/mem0_gpt5.5/$uid" --mode mem0 --model gpt-5.5 \
    --judge_model gpt-5.5 --workers 4 --memory_token_cap 4096 \
    --task chatbot_personalized_response --resume --retry_failed \
    > "/tmp/cpr_eval/mem0_serial.${uid}.log" 2>&1
  echo "   exit=$? CPR_rows=$(python3 -c "import csv;csv.field_size_limit(10**9);print(sum(1 for r in csv.DictReader(open('results/mem0_gpt5.5/$uid/results.csv')) if r['task_type']=='chatbot_personalized_response'))" 2>/dev/null)"
done
echo "MEM0 SERIAL DONE"
