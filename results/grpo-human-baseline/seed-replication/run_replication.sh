#!/usr/bin/env bash
# Seed replication: seeds 43,44,45 x {bfloat16, float32} = 6 runs.
# Wave 1: four runs, one per GPU. Wave 2: the remaining two. Each run is
# identical to run A / run B except for --seed and --policy-dtype.
# See PREREGISTRATION.md for the decision rule fixed before launch.
set -uo pipefail
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
OUT=results/grpo-human-baseline/seed-replication

launch() {  # gpu seed dtype
  local gpu=$1 seed=$2 dtype=$3 tag
  tag="seed${seed}-${dtype}"
  CUDA_VISIBLE_DEVICES=$gpu ~/.local/bin/uv run python -u \
    scripts/20_grpo-diagnostics/run_grpo_baseline.py \
    --run-dir "$OUT/$tag" --run-name "$tag" \
    --policy-dtype "$dtype" --seed "$seed" \
    --target-words 50 100 --steps 500 --num-rollouts 4 --rollout-batch-size 4 \
    --max-new-tokens 256 --checkpoint-every 100 --log-samples-every 1 \
    --skip-zero-advantage-updates \
    > "$OUT/$tag.log" 2>&1
  echo "EXIT=$?" >> "$OUT/$tag.log"
}

echo "wave 1 start $(date)"
launch 0 43 bfloat16 & launch 1 43 float32 & launch 2 44 bfloat16 & launch 3 44 float32 &
wait
echo "wave 1 done $(date)"
launch 0 45 bfloat16 & launch 1 45 float32 &
wait
echo "wave 2 done $(date)"
