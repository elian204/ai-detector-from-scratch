#!/usr/bin/env bash
# Run A -- policy dtype bfloat16.
# Identical in every other respect: same seed, prompts, reward, loss, and
# hyperparameters as the published stage-18 trainer, which is not edited.
set -euo pipefail
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=0
exec ~/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-human-baseline/run-A-bf16 \
  --run-name "run-A-bf16" \
  --policy-dtype bfloat16 \
  --target-words 50 100 \
  --steps 500 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 256 \
  --checkpoint-every 50 \
  --log-samples-every 1 \
  --skip-zero-advantage-updates
