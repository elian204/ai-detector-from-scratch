#!/usr/bin/env bash
# Raschka's documented 1-step smoke test, run VERBATIM (README 2.3), no edits.
# Purpose: confirm the pipeline runs end-to-end on this box and capture real
# verifier P_AI values for the bf16 reward-resolution diagnostic.
set -euo pipefail
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=0
exec ~/.local/bin/uv run python -u \
  scripts/18_reinforcement-learning/05_train_grpo_human_writing.py \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 1 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 256
