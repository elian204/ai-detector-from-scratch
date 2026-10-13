&nbsp;
# Slope reward

The quality-gate reward, `results/grpo-quality-gate-reward.md`, passed its dry run and then failed in training. The reward was 0 for most rollouts, so 28 of 40 steps had no judged rollout. The count of steps with all advantages zero is 33, as the confound section below shows, and the optimizer still stepped on them. The policy drifted to answers over 300 words. This note replaces it with a reward that has a slope everywhere. It records the optimizer check, the reward, and a dry run on two sets of 20 groups. The run that followed is at the end of the note.
&nbsp;
## Optimizer check

The question was what a zero-advantage step does to the optimizer. The repo trainer, `scripts/18_reinforcement-learning/05_train_grpo_human_writing.py`, has a flag `--skip-zero-advantage-updates`, and the flag is off by default. With it on, `compute_grpo_loss` computes the advantages and tests them with `torch.allclose(advantages, 0, atol=1e-8, rtol=0)`. If they are all zero it returns `loss_tensor = None` before any log-probability forward. The train loop runs `zero_grad`, `backward`, `clip_grad_norm_` and `optimizer.step()` only inside `if stats["loss_tensor"] is not None`. So with the flag on, a zero-advantage step skips the optimizer entirely. AdamW momentum, the second moment, the step count, and weight decay are all untouched.

The earlier sweep, quality-gate, and Sonnet JSON 40-step scripts did not do this. They have their own training loop, they did not pass the flag, and they called `optimizer.step()` on every step. On a zero-advantage step the loss was exactly zero and the gradient was zero, but AdamW still ran. It applied the stored momentum, which moves the weights, it applied weight decay, and it incremented the step count.

A short check shows the difference. `/tmp/grpo_zero_adv_check.py` imports the repo functions, builds a small linear model with AdamW at learning rate 1e-5, and takes 3 normal steps. Then it takes one zero-advantage step both ways.

- Optimizer called with a zero gradient: the weights changed, with a maximum change of 3.1e-6. The first moment changed, and the Adam step count went from 3 to 4. Of that change, 3e-8 is weight decay and the rest is momentum.
- The repo skip path: the weights, the first moment, the second moment, and the step count were all identical before and after.
- A reward spread of 1e-6 gives advantages of 0.0075 and -0.0025, which is not zero under the repo criterion, so the step is not skipped. The skip applies only when the rewards are equal to within about 1e-8 after normalization.

The slope run uses the skip. Its script, `/tmp/grpo_len200_slope_train.py`, does the same test. On a zero-advantage step it sets the loss to 0.0 and does not compute log-probabilities, call `backward`, clip, or call `optimizer.step()`. The step record has a `no_gradient` field. The earlier sweep and quality-gate numbers stand as they were reported, but any zero-advantage step in them also moved the weights through momentum, so those runs were not strictly "no update" on those steps.
&nbsp;
## Confound in the earlier runs

The T sweep, `results/grpo-length-200-t-sweep.md`, the quality-gate run, `results/grpo-quality-gate-reward.md`, and the Sonnet JSON 40-step run, `results/grpo-sonnet-json.md`, did not skip zero-advantage steps. Their training loops called `optimizer.step()` on every step. On a step whose advantages were all zero the loss and the gradient were exactly zero, but AdamW still applied its stored momentum and its weight decay, and its step count advanced. So the weights moved on those steps. The results of those runs are confounded on this point. They should be read as runs with an extra update on every zero-advantage step, not as runs that made no update there.

Counting the steps by the repo criterion, advantages all within 1e-8 of zero, in the saved step logs:

| Run | Steps | Zero-advantage steps |
| --- | ---: | ---: |
| T = 4 | 40 | 2 |
| T = 2 | 80 | 4 |
| T = 1 | 40 | 4 |
| Quality gate, T = 1 | 40 | 33 |
| Sonnet JSON 40-step run | 40 | 24 |

For the sweep the count is small, so the effect on those results is probably small, though it cannot be measured from the logs. For the quality-gate run it is large. 33 of 40 steps moved the weights through momentum alone. The earlier note counted 28 steps with no judged rollout. The other 5 had a judged rollout, but every reward in the group was equal. The drift to answers over 300 words in that run cannot be separated from this. The Sonnet JSON run had 24 of 40 steps with all four rewards equal to 0, found in its saved step log and checked against its script, which has a single unconditional `optimizer.step()`. Its zero-advantage steps moved the weights through momentum as well, so its result is confounded in the same way. None of these runs were repeated with the skip. The slope runs, and every run after them, skip these steps, and zero steps had no gradient in the seed-42 slope run.
&nbsp;
## Reward

q_i = 1 if `claude-sonnet-5-5` picks the rollout over the matched base in both orders. q_i = 0 if it picks the base in both orders. q_i = 0.5 in every other case: a tie, a split between the orders, or a call that does not parse. Parse failures are counted and reported.

s_i = sigmoid(z_i / T), T = 1, where z_i is the raw human-class logit of `qwen3-variable` before the saved temperature.

reward_i = hard_i * length_score_i * trigram_ratio_i ** 2 * (q_i + 0.3 * s_i)

hard_i is 0 only for a stub under 30 words or a completion that reached 700 tokens. There is no 100 to 300 word window. length_score_i = min(w, 203) / max(w, 203), where w is the whitespace word count. The prompt asks for approximately 200 words. The trigram ratio is distinct word trigrams over all word trigrams, lowercased.

The judge is the JSON-schema verdict as before: `claude-sonnet-5-5`, no tools, no temperature, both orders, one pair per call. A rollout is judged only when hard_i is 1. Rollout against the base is judged with the rollout in slot A and again in slot B. The matched base is the gated-in answer among the eight frozen base answers whose word count is closest to the rollout, and only the stub and runaway rules gate the pool. A tie takes the lowest pool index. If a rollout is word-for-word the matched base, q_i is 0.5 and no call is made. If a gated-in rollout has no gated-in matched base, q_i is 0.5.

Wins against the other rollouts are not used. N is not used.

The reward is positive for every rollout that is not a stub or a runaway, because s_i is above 0 and q_i is at least 0. That is the change from the quality-gate reward. A rollout that loses to its base in both orders still earns 0.3 * s_i * length * trigram squared, so the group keeps a ranking, though a low one. q_i dominates, since it moves the reward by up to 1 and the detector term by at most 0.3. Inside one q the ranking follows s_i, the length score, and the trigram ratio. A group where every rollout has the same q is ranked by those three alone.

The advantage is the repo one: (reward - mean) / (std with unbiased False + 1e-4). A group with no real spread gets advantages near zero. Spread is real when the range of the four rewards is above 1e-4 and the maximum is above 0.
&nbsp;
## Dry run

The 20 groups are `validation-00001` through `validation-00020`, four rollouts each. Temperature 0.8, `top_p` 0.9, 700 tokens, and the 200-word prompt, for both sets. Set one is the 80 saved base-model rollouts of `results/grpo-length-matched-dry-run.md`. Set two is 80 new rollouts from the T = 1 sweep step-40 checkpoint, `results/grpo-length-200-t1-weights`, generated with seed 42. The matched bases come from the same 160-answer pool. The z values are from `qwen3-variable` on CPU. The pairs of set one were already judged in both orders, so those verdicts came from the call cache and cost no new request. The 160 calls of set two were new.

The dry run passes. Both sets have a real spread in 20 of 20 groups. The bar was 15.
&nbsp;
### Summary

| Item | Set one, base | Set two, step 40 |
| --- | --- | --- |
| Groups with a real spread, of 20 | 20 | 20 |
| Groups with spread above 0.01, of 20 | 18 | 19 |
| Groups with a nonzero advantage, so a gradient, of 20 | 20 (1.00) | 20 (1.00) |
| Parse failures, of 80 | 0 | 0 |
| Gate fires, of 80 | 0 | 0 |
| Gated-in rollouts with no matched base | 0 | 0 |
| Verdicts: rollout, base, split, tie | 42, 26, 9, 3 | 25, 40, 12, 3 |
| q = 1, 0.5, 0 | 42, 12, 26 | 25, 15, 40 |
| Groups where q differs inside the group, of 20 | 16 | 16 |
| Mean q | 0.600 | 0.406 |
| Mean sigmoid(z) | 0.107 | 0.476 |
| Mean reward | 0.400 | 0.350 |
| Median spread per group | 0.633 | 0.617 |
| Words, median and mean | 188 and 214.8 | 210.5 and 208.7 |
| Rollouts over 300 words, of 80 | 19 | 12 |
| Sonnet calls per step | 8 | 8 |
| USD per step, mean (min to max) | 0.0194 (0.0125 to 0.0301) | 0.0185 (0.0111 to 0.0242) |
| New API requests | 0 | 160 |

Cost is computed from the usage the API returned, at $2 per million input tokens and $10 per million output tokens. For 80 steps this is about $1.5 for the reward judge, before the evaluation checks.

On set one, the rollouts beat their matched base in 42 of 80 and lose in 26. On set two they win in 25 and lose in 40. That agrees with the sweep, where the step-40 policy lost to the base on quality at T = 1. The reward still has spread there, because q differs inside 16 of the 20 groups and sigmoid(z) is higher.

Two groups in set one, `validation-00010` and `validation-00017`, have q = 0 for all four rollouts. Their spreads are 0.0002 and 0.0008, which clear the 1e-4 bar, and they rank only by the small detector term. They give a weak gradient. In `validation-00010` the largest absolute advantage is 0.52, because the 1e-4 in the denominator is comparable to the standard deviation. Counting only groups with a spread above 0.01 gives 18 and 19, so the result passes either way.
&nbsp;
### Length among rollouts with the same q

For every pair of gated-in rollouts in a group that have the same q and different word counts, the longer one's reward is compared with the shorter one's.

| Pairs | Set one | Set two |
| --- | --- | --- |
| Same q, different length | 52 | 64 |
| Longer ranks below shorter | 23 (44%) | 32 (50%) |
| Longer ranks above shorter | 29 | 32 |
| Shorter is over 203 words, so both over target | 14 | 21 |
| In those, longer ranks below shorter | 12 (86%) | 13 (62%) |

So longer answers do not rank below shorter ones in general. Below the target the longer answer is closer to 203 and often ranks above, and s_i and the trigram ratio change the order as well. Above 203 words the longer answer ranks below the shorter one in 12 of 14 and 13 of 21 pairs, which is the pull toward the target the length score gives. The length score is a ratio, so past 203 it falls slowly: 300 words gives 0.68. Because q moves the reward by up to 1, a 300-word answer with q = 1 still out-earns a 200-word answer with q = 0.5. The run has a stop rule for the median length, 350 words.
&nbsp;
### Rewards per group

In each row the entries are the four rollouts in order. `v1` is `validation-00001`.

### Set one, base rollouts

| Group | Rewards | q | Words | Spread | Real |
| --- | --- | --- | --- | --- | --- |
| v1 | 0.7587 0.1330 0.6159 0.9080 | 1 0 0.5 1 | 252 95 168 184 | 0.7749 | yes |
| v2 | 0.6432 0.7115 0.9622 0.2516 | 1 1 1 0 | 134 139 209 180 | 0.7106 | yes |
| v3 | 0.2291 0.0007 0.0002 0.9705 | 0.5 0 0 1 | 342 91 301 201 | 0.9704 | yes |
| v4 | 0.0031 0.5892 0.6762 0.0158 | 0 1 1 0 | 201 121 123 174 | 0.6731 | yes |
| v5 | 0.4717 0.0003 0.6407 0.0004 | 0.5 0 1 0 | 207 213 138 182 | 0.6404 | yes |
| v6 | 0.0283 0.9641 0.7497 0.1469 | 0 1 1 0 | 159 153 251 103 | 0.9358 | yes |
| v7 | 0.5422 0.2663 0.6493 0.5688 | 1 0.5 1 1 | 116 108 280 300 | 0.3830 | yes |
| v8 | 0.5985 0.2380 0.6055 0.0003 | 1 0.5 1 0 | 284 397 293 109 | 0.6052 | yes |
| v9 | 0.0001 0.1977 0.9515 0.0001 | 0 0.5 1 0 | 403 430 197 464 | 0.9514 | yes |
| v10 | 0.0003 0.0005 0.0003 0.0004 | 0 0 0 0 | 220 172 137 153 | 0.0002 | yes |
| v11 | 0.6217 0.8746 0.7284 0.6760 | 1 1 1 1 | 130 147 260 137 | 0.2529 | yes |
| v12 | 0.6478 0.6709 0.4344 0.8825 | 1 1 1 1 | 301 138 92 226 | 0.4481 | yes |
| v13 | 0.0004 0.2086 0.6258 0.4594 | 0 0.5 1 1 | 97 90 131 93 | 0.6254 | yes |
| v14 | 0.6079 0.4019 0.2244 0.9388 | 1 0.5 0.5 1 | 299 238 91 192 | 0.7144 | yes |
| v15 | 0.4931 0.4288 0.3652 0.0002 | 1 0.5 1 0 | 100 178 74 301 | 0.4929 | yes |
| v16 | 0.2934 0.2475 0.3134 0.8643 | 1 1 0.5 1 | 484 379 241 233 | 0.6168 | yes |
| v17 | 0.0009 0.0001 0.0001 0.0005 | 0 0 0 0 | 110 83 62 110 | 0.0008 | yes |
| v18 | 0.5114 0.0055 0.4582 0.6179 | 1 0 1 1 | 358 138 416 311 | 0.6123 | yes |
| v19 | 0.3865 0.0003 0.5246 0.6478 | 1 0 1 1 | 496 351 383 301 | 0.6476 | yes |
| v20 | 0.0002 0.5929 0.3061 0.8391 | 0 1 0.5 1 | 362 313 127 225 | 0.8390 | yes |

### Set two, T = 1 step-40 checkpoint

| Group | Rewards | q | Words | Spread | Real |
| --- | --- | --- | --- | --- | --- |
| v1 | 0.8451 0.6891 1.1466 0.2490 | 1 1 1 0 | 231 276 199 187 | 0.8976 | yes |
| v2 | 0.6935 0.0626 0.0021 0.1794 | 1 0 0 0 | 267 430 305 133 | 0.6914 | yes |
| v3 | 0.8139 0.0223 0.0562 0.0250 | 1 0 0 0 | 160 168 244 204 | 0.7915 | yes |
| v4 | 0.1572 0.2469 0.5250 0.1862 | 0 0 1 0 | 241 226 103 216 | 0.3677 | yes |
| v5 | 0.0007 0.0010 0.0002 0.0473 | 0 0 0 0 | 258 128 349 141 | 0.0471 | yes |
| v6 | 0.7541 0.3706 1.0513 0.1904 | 1 1 1 0 | 126 397 188 152 | 0.8609 | yes |
| v7 | 0.0020 0.8592 0.5412 0.7767 | 0 1 1 1 | 144 210 327 247 | 0.8572 | yes |
| v8 | 0.1572 0.0925 0.4162 0.2041 | 0 0 0.5 0 | 157 362 211 164 | 0.3237 | yes |
| v9 | 0.0030 0.0019 0.0073 0.0026 | 0 0 0 0 | 315 268 252 108 | 0.0054 | yes |
| v10 | 0.0012 0.0020 0.0016 0.1900 | 0 0 0 0 | 242 215 125 277 | 0.1887 | yes |
| v11 | 0.3063 0.4791 0.5160 0.3676 | 0.5 0.5 0.5 0.5 | 87 134 164 128 | 0.2096 | yes |
| v12 | 0.5686 0.6771 0.4047 0.6497 | 1 1 0.5 1 | 112 141 205 123 | 0.2724 | yes |
| v13 | 0.1787 0.1269 0.2260 0.1643 | 0.5 0 0.5 0 | 72 102 60 120 | 0.0991 | yes |
| v14 | 1.0286 0.5512 0.1499 0.3216 | 1 0.5 0 0.5 | 217 246 239 284 | 0.8788 | yes |
| v15 | 0.1246 0.1442 0.0042 0.8773 | 0 0 0 1 | 108 228 149 205 | 0.8731 | yes |
| v16 | 0.5469 0.4909 0.4678 0.0048 | 1 0.5 0.5 0 | 92 152 287 330 | 0.5421 | yes |
| v17 | 0.1780 0.0026 0.7533 0.0873 | 0 0 1 0 | 162 109 244 273 | 0.7508 | yes |
| v18 | 0.3674 0.5263 0.7679 0.5669 | 0.5 1 1 1 | 226 313 253 333 | 0.4005 | yes |
| v19 | 0.0831 0.3234 0.5164 0.8682 | 0 0.5 0.5 1 | 158 139 217 228 | 0.7851 | yes |
| v20 | 1.0937 0.1035 0.0052 0.7727 | 1 0 0 1 | 193 334 328 249 | 1.0885 | yes |

&nbsp;
## Run

The dry run passed on both sets, so the run started. Seed 42, T = 1, `Qwen/Qwen3-0.6B-Base`, group size 4, 80 steps from scratch, `max_new_tokens` 700, temperature 0.8, `top_p` 0.9, AdamW at 1e-5, gradient clip 1.0, no KL, bfloat16, gradient checkpointing, GPU 3. The prompt asks for approximately 200 words. Zero-advantage steps skip the optimizer as in the optimizer check. The script is `/tmp/grpo_len200_slope_train.py`, the tmux session was `grpo-slope-t1`, and the log is `results/grpo-length-200-slope-t1/run.log`. Weights are in `results/grpo-length-200-slope-t1-weights` on the lab machine and are not committed. One seed only.

The run finished all 80 steps. No stop rule fired. Zero steps had no gradient, so the skip never triggered. The gate fired once, at step 2, on a 700-token runaway. The gate-fire rise streak peaked at 1. The Opus win rate never fell under 40%. The median rollout length over any 10 steps peaked at 205 words, far under 350. The reward judge had 0 parse failures over 638 calls. Over the 80 steps the 320 rollouts were judged against their matched base as rollout 164, base 104, split 45, tie 6, and 1 was gated out.
&nbsp;
### Checks

Opus is `claude-opus-5-5`. At steps 20, 40, and 60 the check is 50 validation prompts with Opus only. At step 80 it is 100 prompts with Opus and `gpt-5.5` (returned id `gpt-5.5-2026-04-23`). Both orders, agreed pairs only. Ties and unparsed pairs stay out of the denominator. The band is trained words divided by base words between 0.8 and 1.2. The base answer is the length-matched base answer from the eval pool. Detector means are over all prompts for the same texts.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 25/46 | 24/41 | 0.078708 | 0.194924 | 0.092788 | 0.191228 |
| 40 | Opus | 22/39 | 17/31 | 0.120580 | 0.111463 | 0.139844 | 0.144639 |
| 60 | Opus | 28/45 | 24/39 | 0.113588 | 0.172391 | 0.132594 | 0.186357 |
| 80 | Opus | 49/83 | 36/63 | 0.112886 | 0.131564 | 0.150557 | 0.235134 |
| 80 | gpt-5.5 | 51/80 | 43/65 | 0.112886 | 0.131564 | 0.150557 | 0.235134 |

Win rates on all agreed pairs are 54.3%, 56.4%, 62.2%, and 59.0% for Opus at steps 20, 40, 60, and 80, and 63.8% for `gpt-5.5` at step 80. Within 20% length they are 58.5%, 54.8%, 61.5%, and 57.1% for Opus, and 66.2% for `gpt-5.5`. Opus ties were 4, 10, 5, and 15, and `gpt-5.5` ties were 20. At step 80 the 95% Wilson interval for the Opus rate is 0.48 to 0.69 and for the `gpt-5.5` rate 0.53 to 0.73. Opus had 5 unparsed pairs over the run, 1 at step 40 and 4 at step 80. `gpt-5.5` had 0.

At step 80 the trained answers have a mean of 231 words and a median of 206, against 215 and 197 for the base answers. 17 of 100 trained answers and 12 of 100 base answers are over 300 words.
&nbsp;
### Detector scores did not rise

The policy won on quality and did not become more human to either detector. At step 80 the trained sigmoid(z) is 0.113 against 0.132 for the base, and the trained DistilBERT P(human) is 0.151 against 0.235. At steps 20, 60, and 80 both detector means are below the base. At step 40 the sigmoid is level and DistilBERT is slightly below. The mean raw logit z at step 80 is -4.47 for trained and -4.37 for base.

That is the opposite corner from the T = 1 sweep run, which raised both detectors about three times and lost on quality. In this reward q_i carries up to 1 and the detector term at most 0.3, and the sigmoid is small for almost every rollout. Over the run the mean sigmoid of the training rollouts was 0.005 to 0.24 per block of 10 steps, mostly under 0.1. With such small values the detector term hardly separates rollouts, so the ranking inside a group is set by q, length, and the trigram ratio. The policy learned to beat its base, and the detector term did not push it toward human text.
&nbsp;
### Training trajectory

| steps | mean q of judged rollouts | mean words | mean sigmoid | mean DistilBERT | mean trigram ratio | mean reward | gate fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 to 10 | 0.438 | 181 | 0.076 | 0.101 | 0.969 | 0.266 | 1 |
| 11 to 20 | 0.537 | 166 | 0.007 | 0.087 | 0.971 | 0.384 | 0 |
| 21 to 30 | 0.613 | 187 | 0.017 | 0.100 | 0.983 | 0.453 | 0 |
| 31 to 40 | 0.625 | 227 | 0.005 | 0.054 | 0.985 | 0.475 | 0 |
| 41 to 50 | 0.600 | 195 | 0.242 | 0.190 | 0.987 | 0.521 | 0 |
| 51 to 60 | 0.700 | 149 | 0.097 | 0.107 | 0.987 | 0.484 | 0 |
| 61 to 70 | 0.637 | 214 | 0.044 | 0.016 | 0.984 | 0.472 | 0 |
| 71 to 80 | 0.588 | 227 | 0.007 | 0.074 | 0.980 | 0.426 | 0 |

The mean q rose from 0.44 to about 0.6 to 0.7 and stayed there. The trigram ratio rose from 0.969 to 0.98 or more. The length stayed near the target and did not drift up, which is where the quality-gate run failed. The block of steps 41 to 50 has a high mean sigmoid because of a few rollouts with a high z, not a trend.
&nbsp;
### Cost

| Part | Calls | USD |
| --- | ---: | ---: |
| Reward judge, `claude-sonnet-5-5` | 638 | 1.416 |
| Opus checks, `claude-opus-5-5` | 505 | 5.367 |
| `gpt-5.5` at step 80 | 201 | 1.168 |
| Total | 1,344 | 7.952 |

The reward judge cost about $0.018 per step, as the dry run predicted. The run took about 3 hours 15 minutes of wall time, most of it the step-80 check, because GPU 3 was shared with other jobs.
&nbsp;
## Seed 2

The run was repeated with seed 2 in `results/grpo-slope-seed2.md`. It stopped at step 41 on the median-length rule. The Opus win rate was 62% at step 20 and 60% at step 40, in line with seed 42, and the detector means stayed below the base. The length drifted up in that run, to a 10-step median of 362.5 words, and did not in the seed-42 run. So the quality result held across two seeds, the length result did not, and the seed-42 run reaching step 80 is not guaranteed by the reward.
