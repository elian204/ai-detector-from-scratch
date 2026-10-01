&nbsp;
# grpo-250w-kl-beta0.5

Same setup as `results/grpo-250w-kl`, for 60 steps, starting from `Qwen/Qwen3-0.6B-Base`. `--trigram-repetition` is off. The only change is β = 0.5 instead of 0.05. This directory does not overwrite `results/grpo-250w-kl/`.

&nbsp;
## Conclusion

Finished, exit 0. β = 0.5 and the trigram factor is off. At step 60 all four answers are spaced loops of a return instruction, cut at 1616 tokens, with kl 0.06–0.14. The winner is "Return again" 190 times (385 words). The β = 0.05 winner was `<content>`. Both the training detector and DistilBERT still score these loops human. Checkpoints and model weights are not in this commit.

&nbsp;
## Reward

`kl` is the mean, over completion tokens, of `log π_current - log π_base`.

`reward = P(human) * length_score - 0.5 * kl`

`π_base` is a second load of `Qwen/Qwen3-0.6B-Base` in float32, eval mode, `requires_grad` false, and not in the AdamW optimizer.

`metrics.csv` adds a `kl` column, the mean of the four rollouts. Each rollout in `rollouts.jsonl` also has its own `kl`.

&nbsp;
## Memory

The β = 0.05 run peaked at 17.50 GB allocated on one 24 GB card, with policy, frozen base, and verifier together. This run uses that same placement on GPU 0.

&nbsp;
## Command

GPU: `CUDA_VISIBLE_DEVICES=0`.

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-kl-beta0.5 \
  --run-name grpo-250w-kl-beta0.5 \
  --policy-model Qwen/Qwen3-0.6B-Base \
  --policy-dtype float32 \
  --target-words 250 \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 60 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 1616 \
  --uncap-response-tokens \
  --kl-beta 0.5 \
  --learning-rate 1e-5 \
  --checkpoint-every 10 \
  --log-samples-every 1 \
  --seed 42 \
  --skip-zero-advantage-updates \
  --allow-existing
```

Log: `results/grpo-250w-kl-beta0.5/run.log`

```bash
tmux attach -t grpo-250w-kl-beta0.5
```

`--checkpoint-every 10` saves steps 10, 20, 30, 40, 50, and 60. `run.sh` deletes every saved checkpoint except 20, 50, and 60. The trainer also writes `step-00060-final` at the end; that file is kept.
