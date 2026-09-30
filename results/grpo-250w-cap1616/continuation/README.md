# grpo-250w-cap1616 continuation

Steps 51–100 of the 1616-token run. Weights are loaded from `results/grpo-250w-cap1616/checkpoints/step-00050`. Logs from steps 1–60 stay in the parent directory and are not overwritten.

## Conclusion

Finished, exit 0. Step 100 is about 69 words, length score 0.276, but every rollout is cut at 1616 tokens because the tail has no spaces. Training P(human) is 0.997, and DistilBERT scores those four answers about 0.999 human. This is not the noun-phrase loop from the no-penalty step 60. Checkpoints and model weights are not in this commit.

AdamW is created fresh inside `train()` after the weights load, so the optimizer is reset at step 51. Checkpoint files do not store Adam state.

`train()` reads `start_index` (default 0) and loops `range(start_index, total_steps)`. `--start-index 50` with `--steps 100` runs steps 51 through 100. The prompt is `train_data[step_index % len]`. Reward, loss, and `--uncap-response-tokens` are unchanged.

`--checkpoint-every 10` still saves 60, 70, 80, 90, and 100. `run.sh` deletes every saved checkpoint except `step-00100`. The trainer also writes `step-00100-final` at the end; that file is kept. The original `step-00050` is outside this directory and is not deleted.

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-cap1616/continuation \
  --run-name grpo-250w-cap1616-cont \
  --policy-model results/grpo-250w-cap1616/checkpoints/step-00050 \
  --policy-dtype float32 \
  --target-words 250 \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 100 \
  --start-index 50 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 1616 \
  --uncap-response-tokens \
  --learning-rate 1e-5 \
  --checkpoint-every 10 \
  --log-samples-every 1 \
  --seed 42 \
  --skip-zero-advantage-updates \
  --allow-existing
```

Log: `results/grpo-250w-cap1616/continuation/run.log`

```bash
tmux attach -t grpo-250w-cap1616-cont
```
