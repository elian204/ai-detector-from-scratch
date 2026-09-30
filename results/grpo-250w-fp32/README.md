&nbsp;
# grpo-250w-fp32

Float32 GRPO pilot matching `results/grpo-human-baseline/run-B-fp32`, except target length 250 and 100 steps.

&nbsp;
## Conclusion

Finished, exit 0. Training P(human) is 0.9998 by step 20. By step 100 the four answers are one "of the U.S." loop, the length score is 1, and the update is skipped. DistilBERT, held out, scores that loop about 0.996 human. 381 of 400 answers hit the 416-token cap. KL was not started. Checkpoints and model weights are not in this commit.

Reward is the published trainer reward, unchanged: `r = (1 - P_AI) * length_score` from frozen `qwen3-variable`. No KL, no clip, no second term.

`max_new_tokens` is the trainer default 1616. The trainer then caps a 250-word answer at `min(1616, ceil(250 * 1.6) + 16) = 416` tokens. The short pilot's 256 is not used.

The trainer only accepts a checkpoint stride. `--checkpoint-every 10` is the stride that hits 20, 50, and 100. `run.sh` deletes any other `Saved checkpoint` directory after the save line is logged, so steps 10, 30, 40, 60, 70, 80, and 90 are not kept. The trainer also writes `step-00100-final` at the end; that file is kept. There is no step-0 save in the wrapper. The base policy is `Qwen/Qwen3-0.6B-Base`.

`samples.txt` is still the trainer's first three rollouts. Every rollout (all 4) is in `rollouts.jsonl`: text, word count, P_AI, length score, reward, advantage, gen tokens, hit_token_cap.

`--allow-existing` is set because this README and `run.sh` are already in the run directory. The launcher refuses to start if `metrics.csv` or `rollouts.jsonl` already exist.

DistilBERT is a held-out judge. It is not loaded by the trainer and is not in the reward. If `models/distilbert` is missing, `run.sh` downloads it in parallel.

GPU: `CUDA_VISIBLE_DEVICES=1` (this host had no idle GPU; GPU 1 had the same free memory as the others and 99% util from another user's job). `--policy-device cuda` and `--verifier-device cuda` (the flags do not accept `cuda:N`).

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
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
  --allow-existing
```

Log: `results/grpo-250w-fp32/run.log`

```bash
tmux attach -t grpo-250w-fp32
```
