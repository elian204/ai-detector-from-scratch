&nbsp;
# Slope run, seed 2

This replicates the seed-42 slope run of `results/grpo-slope-reward.md` with seed 2. The reward, the settings, the checks, and the stop rules are the same. The run stopped at step 41 on the median-length rule. It did not reach step 80, so there is no step-80 check and no `gpt-5.5` check.
&nbsp;
## Setup

Seed 2 sets the torch seed for rollout sampling and the policy update. The training prompts are the same 80 prompts as in the seed-42 run, in the same order. The prompts come from the dataset shuffled with seed 42, and the length-matched base pools were built for those prompts, so the data order was not changed. `Qwen/Qwen3-0.6B-Base`, group size 4, `max_new_tokens` 700, temperature 0.8, `top_p` 0.9, AdamW at 1e-5, gradient clip 1.0, no KL, T = 1. reward_i = hard_i * length_score_i * trigram_ratio_i ** 2 * (q_i + 0.3 * sigmoid(z_i / 1)). Zero-advantage steps skip backward and `optimizer.step()`. The script is `/tmp/grpo_len200_slope_train.py` with `RUN_SEED=2`, the tmux session was `grpo-slope-seed2`, and the log is `results/grpo-length-200-slope-seed2-t1/run.log`. The checks are those of the seed-42 run: Opus at steps 20, 40, and 60 on 50 validation prompts, step 80 on 100 prompts with Opus and `gpt-5.5`, both orders, agreed pairs. Weights are in `results/grpo-length-200-slope-seed2-t1-weights` on the lab machine and are not committed.

The stop rules are an Opus win rate under 40%, under 45% twice in a row, the gate-fire count rising for 20 steps, more than half of the last 10 steps with no gradient, and a median length over 350 words.
&nbsp;
## Result

The run stopped at step 41. The recorded reason is a median rollout length of 362.5 words over the last 10 steps, which is over 350. The median over the last 10 steps was 280 at steps 37 to 39, 330.5 at step 40, and 362.5 at step 41. No other rule came close. The Opus win rate stayed at 60% or more, there were no steps without a gradient, the gate-fire rise streak peaked at 1, and the reward judge had 0 parse failures over 316 calls. Over the 41 steps the 164 rollouts were judged against their matched base as rollout 90, base 44, split 21, tie 3, and 6 were gated out. The gate fires were 700-token runaways, 1 at step 9, 2 at step 12, 1 at step 28, and 2 at step 39.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 28/45 | 19/31 | 0.077524 | 0.102338 | 0.083371 | 0.133028 |
| 40 | Opus | 27/45 | 22/39 | 0.061849 | 0.147017 | 0.120923 | 0.216925 |

Win rates on all agreed pairs are 62.2% at step 20 and 60.0% at step 40. Within 20% length they are 61.3% and 56.4%. Opus ties were 4 and 4, and each check had 1 unparsed pair. The 95% Wilson intervals are 0.48 to 0.75 and 0.45 to 0.73. The seed-42 run had 54.3% and 56.4% on all pairs at the same steps, and 58.5% and 54.8% within 20% length. The two seeds agree on quality at steps 20 and 40, within the noise of 45 pairs.

The detector scores did not rise. At step 40 the trained sigmoid is 0.062 against 0.147 for the matched base, and the trained DistilBERT P(human) is 0.121 against 0.217. At step 20 the trained values are also below the base, 0.078 against 0.102 and 0.083 against 0.133. That matches the seed-42 run, where the detector means were also at or below the base.
&nbsp;
## Length is where the seeds differ

The seed-2 policy drifted to longer answers early. The mean training rollout length over the first 41 steps is 290 words, against 191 for the seed-42 run over the same steps. At step 20 the trained evaluation answers average 301 words with a median of 311.5, against 273 and 275.5 for the matched base answers, and 27 of 50 are over 300 words. The base answers at that step are long as well, because the matched base follows the rollout length. By step 40 the evaluation answers are back near the target, 215 words against 207, but the training rollouts then grew again, with a mean of 344 to 438 words in 5 of the last 8 steps. At no point in the seed-42 run did the 10-step median pass 205 words.

The reward does little to hold the length down. The length score is a ratio that falls slowly, 0.68 at 300 words and 0.51 at 400, and a longer answer can win its judged comparison against the length-matched base. A 300-word answer with q = 1 earns more than a 200-word answer with q = 0.5. The seed-42 run did not drift, and the seed-2 run did, so the reward does not pin the length by itself and the seed decides. The median-length stop rule did what it was written for.
&nbsp;
## Training trajectory

| steps | mean q of judged rollouts | mean words | mean sigmoid | mean DistilBERT | mean trigram ratio | mean reward | gate fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 to 10 | 0.525 | 272 | 0.023 | 0.050 | 0.962 | 0.331 | 1 |
| 11 to 20 | 0.713 | 314 | 0.002 | 0.026 | 0.981 | 0.382 | 2 |
| 21 to 30 | 0.617 | 242 | 0.073 | 0.130 | 0.974 | 0.381 | 1 |
| 31 to 40 | 0.738 | 330 | 0.011 | 0.086 | 0.958 | 0.423 | 2 |
| 41 | 0.750 | 315 | 0.002 | 0.008 | 0.972 | 0.401 | 0 |
&nbsp;
## Cost

| Part | Calls | USD |
| --- | ---: | ---: |
| Reward judge, `claude-sonnet-5-5` | 316 | 0.948 |
| Opus checks, `claude-opus-5-5` | 202 | 2.337 |
| `gpt-5.5` | 0 | 0.000 |
| Total | 518 | 3.285 |

The run took about 2 hours of wall time. The CPU, three cores, was shared with the rank run, and the detector scoring runs on CPU.
