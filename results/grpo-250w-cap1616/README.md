# grpo-250w-cap1616

Same run as `results/grpo-250w-fp32`, except a 250-word answer may use the full 1616 new tokens. Generation still stops on EOS. The 416 cut (`ceil(250 * 1.6) + 16`) is not applied.

## Conclusion

Through step 60, training P(human) is 0.999 (0.999047 at step 60). The answers are loops that stop on EOS at different lengths: step 60 mean length is 299.5 words, generated tokens run 236 / 330.5 / 408, and 0 of 4 hit 1616. Across steps 1–60, 22 of 240 rollouts hit the cap. DistilBERT, held out, also scores the step-60 loops human (mean P(human) 0.997). The identical 250-word "of the U.S." ending of the 416-token run does not appear. The run died while saving the step-60 checkpoint because the disk was full. There is no step 100. Checkpoints and model weights are not in this commit.

Reward is the published trainer reward, unchanged: `r = (1 - P_AI) * length_score` from frozen `qwen3-variable`. No KL, no clip, no second term.

`scripts/18_reinforcement-learning/05_train_grpo_human_writing.py` is not edited. Reward, advantages, and the REINFORCE loss are unchanged. With `--uncap-response-tokens` off, `response_token_limit` is still `min(maximum, max(64, ceil(target_words * 1.6) + 16))`.

## One-line code difference

In `scripts/20_grpo-diagnostics/run_grpo_baseline.py`, only when the flag is set:

```python
trainer.response_token_limit = lambda target_words, maximum: maximum
```

## Command

GPU: `CUDA_VISIBLE_DEVICES=2` (shared 4× RTX 6000; GPU 2 had the most free memory, about 19 GB). `--policy-device cuda` and `--verifier-device cuda`.

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-cap1616 \
  --run-name grpo-250w-cap1616 \
  --policy-dtype float32 \
  --target-words 250 \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 100 \
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

Log: `results/grpo-250w-cap1616/run.log`

```bash
tmux attach -t grpo-250w-cap1616
```

`--checkpoint-every 10` is the stride that hits 20, 50, and 100. `run.sh` deletes any other `Saved checkpoint` directory after the save line is logged, so steps 10, 30, 40, 60, 70, 80, and 90 are not kept. The trainer also writes `step-00100-final` at the end; that file is kept.

`rollouts.jsonl` has every rollout: text, word count, P_AI, length score, reward, advantage, gen tokens, hit_token_cap. `hit_token_cap` is true when gen tokens reach 1616.

`--allow-existing` is set because this README and `run.sh` are already in the run directory. The launcher refuses to start if `metrics.csv` or `rollouts.jsonl` already exist.

DistilBERT is a held-out judge. It is not loaded by the trainer and is not in the reward.
