&nbsp;
# grpo-250w-kl

Same setup as `results/grpo-250w-cap1616`, for 60 steps, starting from `Qwen/Qwen3-0.6B-Base`. `--trigram-repetition` is off. The extra term is a KL penalty toward a frozen copy of that base model, with β = 0.05.

&nbsp;
## Conclusion

Finished, exit 0. β = 0.05 and the trigram factor is off. The step 60 winner is 496 copies of `<content>`, cut at 1616 tokens, with kl 0.03, so the penalty barely applies. The other three answers are short HTML stubs with kl about 1–2. Both the training detector and DistilBERT still score them human. Checkpoints and model weights are not in this commit.

&nbsp;
## Reward

`kl` is the mean, over completion tokens, of `log π_current - log π_base`.

`reward = P(human) * length_score - kl_beta * kl`

`kl_beta` is 0.05 for this run. `--kl-beta 0` leaves the reward exactly `P(human) * length_score`.

`π_base` is a second load of `Qwen/Qwen3-0.6B-Base` in the policy dtype (float32 here), eval mode, `requires_grad` false, and not in the AdamW optimizer.

`metrics.csv` adds a `kl` column, the mean of the four rollouts. Each rollout in `rollouts.jsonl` also has its own `kl`.

&nbsp;
## Memory

The 1616-token float32 run peaked at 15.50 GB allocated. The frozen base is about 2.4 GB of weights and is not optimized, so policy, base, and verifier share GPU 0. A second card was free and was not used.

&nbsp;
## Command

GPU: `CUDA_VISIBLE_DEVICES=0`.

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-kl \
  --run-name grpo-250w-kl \
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
  --kl-beta 0.05 \
  --learning-rate 1e-5 \
  --checkpoint-every 10 \
  --log-samples-every 1 \
  --seed 42 \
  --skip-zero-advantage-updates \
  --allow-existing
```

Log: `results/grpo-250w-kl/run.log`

```bash
tmux attach -t grpo-250w-kl
```

`--checkpoint-every 10` saves steps 10, 20, 30, 40, 50, and 60. `run.sh` deletes every saved checkpoint except 20, 50, and 60. The trainer also writes `step-00060-final` at the end; that file is kept.
