# grpo-250w-trigram

Same setup as `results/grpo-250w-cap1616`, for 60 steps, starting from `Qwen/Qwen3-0.6B-Base`. The extra factor is on only because this command passes `--trigram-repetition`.

## Conclusion

Finished, exit 0. Step 60 is a template essay of about 233 words, repetition_score 0.763, length score 0.919. Training P(human) is 0.998, and DistilBERT also scores it human. The no-penalty step 60, same prompt, was the noun-phrase loop ("player career progression and team performance analysis" repeated). Checkpoints and model weights are not in this commit.

## Reward

With the flag off, `reward = P(human) * length_score`.

With `--trigram-repetition`:

`repetition_score = 1` when the answer has fewer than 3 whitespace words, otherwise unique word-trigrams / all word-trigrams.

`reward = P(human) * length_score * repetition_score`

`metrics.csv` adds a `repetition_score` column, the mean of the four rollouts. Each rollout in `rollouts.jsonl` also has its own `repetition_score`.

## Command

GPU: `CUDA_VISIBLE_DEVICES=1`. The continuation `grpo-250w-cap1616-cont` stays on GPU 0.

```bash
cd /home/dsi/eli-bogdanov/ai-detector-from-scratch
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dsi/eli-bogdanov/.local/bin/uv run python -u scripts/20_grpo-diagnostics/run_grpo_baseline.py \
  --run-dir results/grpo-250w-trigram \
  --run-name grpo-250w-trigram \
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
  --trigram-repetition \
  --learning-rate 1e-5 \
  --checkpoint-every 10 \
  --log-samples-every 1 \
  --seed 42 \
  --skip-zero-advantage-updates \
  --allow-existing
```

Log: `results/grpo-250w-trigram/run.log`

```bash
tmux attach -t grpo-250w-trigram
```

`--checkpoint-every 10` saves steps 10, 20, 30, 40, 50, and 60. `run.sh` deletes every saved checkpoint except 20, 50, and 60. The trainer also writes `step-00060-final` at the end; that file is kept.

Checkpoints for `grpo-250w-cap1616`, its continuation, and `grpo-250w-fp32` are not deleted.
