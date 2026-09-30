#!/usr/bin/env bash
# 250-word float32 GRPO pilot. Reuses scripts/20_grpo-diagnostics/run_grpo_baseline.py.
# Does not edit the published trainer.
set -euo pipefail
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=results/grpo-250w-fp32
LOG="$RUN/run.log"
UV=/home/dsi/eli-bogdanov/.local/bin/uv

if [[ -e "$RUN/metrics.csv" || -e "$RUN/rollouts.jsonl" ]]; then
  echo "refuse: metrics.csv or rollouts.jsonl already exist in $RUN" >&2
  exit 1
fi

mkdir -p "$RUN"
{
  echo "=== grpo-250w-fp32 $(date -Is) ==="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "python: $($UV run python -c 'import sys; print(sys.executable); print(sys.version)')"
} > "$LOG"

# Held-out judge only. Not loaded by the trainer and not part of the reward.
if [[ ! -d models/distilbert ]]; then
  (
    env -u CUDA_VISIBLE_DEVICES "$UV" run python -u \
      scripts/15_classifier-api/download-models.py --fetch --model distilbert
  ) > "$RUN/distilbert-download.log" 2>&1 &
  echo $! > "$RUN/distilbert-download.pid"
  echo "distilbert download pid $(cat "$RUN/distilbert-download.pid")" >> "$LOG"
else
  echo "models/distilbert already present" >> "$LOG"
fi

prune_extra_checkpoints() {
  [[ -f "$LOG" ]] || return 0
  grep -F "Saved checkpoint: " "$LOG" | while IFS= read -r line; do
    path="${line#Saved checkpoint: }"
    base=$(basename "$path")
    case "$base" in
      step-00020|step-00050|step-00100) continue ;;
    esac
    case "$path" in
      */results/grpo-250w-fp32/checkpoints/step-*)
        if [[ -d "$path" ]]; then
          rm -rf -- "$path"
          printf '%s pruned %s\n' "$(date -Is)" "$base" >> "$RUN/checkpoint-prune.log"
        fi
        ;;
    esac
  done || true
}

"$UV" run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-fp32 \
  --run-name grpo-250w-fp32 \
  --policy-dtype float32 \
  --target-words 250 \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 100 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 1616 \
  --learning-rate 1e-5 \
  --checkpoint-every 10 \
  --log-samples-every 1 \
  --seed 42 \
  --skip-zero-advantage-updates \
  --allow-existing \
  >> "$LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$RUN/train.pid"
echo "train pid $TRAIN_PID" >> "$LOG"

(
  set +e
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    prune_extra_checkpoints
    sleep 5
  done
  prune_extra_checkpoints
) &
PRUNE_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
echo "train exit $TRAIN_STATUS" >> "$LOG"
kill "$PRUNE_PID" 2>/dev/null || true
wait "$PRUNE_PID" 2>/dev/null || true
prune_extra_checkpoints
exit "$TRAIN_STATUS"
