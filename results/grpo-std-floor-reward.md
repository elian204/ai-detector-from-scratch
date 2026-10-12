&nbsp;
# Std-floor reward

The rank run, `results/grpo-rank-reward.md`, raised both detector scores to about 4 times the base and lost on quality, with an Opus win rate of 13 of 48 at step 40. GRPO divides the advantages of a group by the group's standard deviation. In a group where q is the same for every rollout, the only reward differences are the rank term, 0.3 * r, times the length and trigram factors. The division scales those small differences back up to advantages of order 1, so the detector drove the update there. Lowering the rank weight does not fix this, because the scaling cancels it in such groups. This note changes the advantage so that small reward differences stay small. It records a dry run on two sets of 20 groups. The runs that followed are at the end of the note.
&nbsp;
## Changes

Advantage with a std floor. A_i = (reward_i - mean) / max(std, 0.25), where std is the population standard deviation of the group's four rewards, as the repo uses. When std is under 0.25 the divisor is 0.25, so a reward difference of d gives an advantage of d / 0.25 and no longer d / std. A group with a small spread now gets small advantages. When std is 0.25 or more the advantage is the same as before, apart from the old 1e-4 term.

Stronger length control. The length score is squared. reward_i = hard_i * length_score_i ** 2 * trigram_ratio_i ** 2 * (q_i + w * r_i). A 362-word answer scores 0.31 on length squared instead of 0.56, and a 300-word answer 0.46 instead of 0.68. r_i is the within-group rank of z_i among the four rollouts scaled to [0, 1], with ties sharing the average rank. q_i, hard_i, the matched base, the judge, the 203-word target, and the gate are unchanged. The two weights tested are w = 0.1 and w = 0.3.

```python
def slope_reward(hard, q, tri, rank, length, weight):
    if hard == 0:
        return 0.0
    return hard * (length ** 2) * (tri ** 2) * (q + weight * rank)

def floor_advantages(rewards, floor=0.25):
    t = torch.tensor(rewards, dtype=torch.float32)
    return (t - t.mean()) / max(float(t.std(unbiased=False)), floor)

def no_gradient_step(rewards):
    return (max(rewards) - min(rewards)) <= 1e-4
```

Zero-advantage skip. A step is skipped, with no backward, no clip, and no `optimizer.step()`, when the reward range is at most 1e-4, which is the same bar the earlier dry runs used for a real spread. This is a rule on the rewards and not the repo test on the advantages. The repo test, `allclose(advantages, 0, atol=1e-8)`, was written for the std + 1e-4 division, where a reward difference of 1e-6 gives advantages of 0.0075 and counts as nonzero. Under the floor a difference of d gives an advantage of d / 0.25, so a difference of 1e-4 gives 4e-4. Differences that small no longer count as nonzero at full size, and a repo test with atol 1e-8 would send nearly all of them through the optimizer as a tiny step with AdamW momentum still applied. The reward-range rule skips them. A group with a larger range keeps its small advantages, and the step is not skipped. Skipping still means no optimizer call at all, as in the slope run. The zero-advantage skip itself is as before.
&nbsp;
## Pass criteria

These were the criteria for the dry run. The numeric reading of the first one was written after the dry-run numbers were in view, so it is not a blind test.

1. Groups where q ties should have clearly smaller advantages than groups where q differs. The reading used is a median abs(A) in tied groups under 0.75 times the median in differing groups, on both sets and at both weights, with the old normalization shown on the same groups for comparison.
2. A real spread, a reward range above 1e-4 with a maximum above 0, in at least 15 of 20 groups on both sets.

A tied group is one where every gated-in rollout has the same q. A gated-out rollout has reward 0 and is not counted for q. The other groups are the q-differs groups. The abs(A) statistics pool the four rollouts of each group in the class. The old normalization is (reward - mean) / (std + 1e-4), applied to the same new rewards.
&nbsp;
## Dry run

The 20 groups are `validation-00001` through `validation-00020`, four rollouts each. Set one is the 80 saved base-model rollouts. Set two is the 80 rollouts of the seed-42 slope run step-80 checkpoint that the rank dry run generated. The matched bases, the judge verdicts, and the z values are the ones already in the call cache and the saved files, so the dry run made no new API request and cost $0. It ran on CPU. Both sets have 0 parse failures and 0 gate fires. The verdicts are rollout, base, split, tie 42, 26, 9, 3 for set one and 44, 23, 11, 2 for set two.

The dry run passes both criteria. The spread is real in 20 of 20 groups on both sets at both weights. The median abs(A) in tied groups is 0.16 to 0.60 times the median in differing groups, so the first criterion is met, and the old normalization gives a ratio of 1.0 on the same groups.

Only 4 of 20 groups in set one and 8 of 20 in set two are tied by this definition, so the tied-group statistics rest on 16 and 32 advantages. In 17 of 20 groups on both sets several rollouts share the top q, but in most of those groups some other rollout has a different q, which makes them differing groups here. The tied-group effect is the same in direction at the two weights and on both sets, and it is weakest on set two at w = 0.3, where the median ratio is 0.60 and the mean ratio 0.77. A tied group there still has a median abs(A) of 0.47, against 0.78 in differing groups.
&nbsp;
### Advantage sizes and spread

Rank weight 0.1.

| Item | Set one | Set two |
| --- | --- | --- |
| Groups where q ties, groups where q differs | 4, 16 | 8, 12 |
| Floor abs(A), q ties: mean, median, max | 0.250, 0.112, 1.318 | 0.488, 0.297, 1.673 |
| Old abs(A), q ties: mean, median, max | 0.898, 0.961, 1.527 | 0.900, 0.913, 1.673 |
| Floor abs(A), q differs: mean, median, max | 0.720, 0.721, 1.722 | 0.794, 0.837, 1.516 |
| Old abs(A), q differs: mean, median, max | 0.875, 0.934, 1.722 | 0.903, 0.907, 1.714 |
| Tied over differs, floor median ratio | 0.16 | 0.35 |
| Tied over differs, floor mean ratio | 0.35 | 0.61 |
| Tied over differs, old median ratio | 1.03 | 1.01 |
| Groups with std under the 0.25 floor | 12 | 12 |
| Groups with a real spread, of 20 | 20 | 20 |
| Groups with spread above 0.01, of 20 | 20 | 20 |
| Smallest and median spread | 0.029 and 0.463 | 0.069 and 0.574 |
| Groups with a gradient (reward range above 1e-4), of 20 | 20 | 20 |
| Groups the old repo zero test would skip | 0 | 0 |
| Top differs from top without the detector term | 2 of 20 | 0 of 20 |
| Groups with q = 0 for all four, rank picks the top | 2 | 3 |
| Any pair changes order | 8 of 20 | 7 of 20 |
| Full-reward top is the highest-z rollout | 5 of 20 | 10 of 20 |

Rank weight 0.3.

| Item | Set one | Set two |
| --- | --- | --- |
| Groups where q ties, groups where q differs | 4, 16 | 8, 12 |
| Floor abs(A), q ties: mean, median, max | 0.337, 0.199, 1.377 | 0.598, 0.470, 1.629 |
| Old abs(A), q ties: mean, median, max | 0.905, 0.933, 1.534 | 0.892, 0.896, 1.674 |
| Floor abs(A), q differs: mean, median, max | 0.719, 0.720, 1.719 | 0.777, 0.779, 1.565 |
| Old abs(A), q differs: mean, median, max | 0.868, 0.851, 1.718 | 0.906, 0.947, 1.573 |
| Tied over differs, floor median ratio | 0.28 | 0.60 |
| Tied over differs, floor mean ratio | 0.47 | 0.77 |
| Tied over differs, old median ratio | 1.10 | 0.95 |
| Groups with std under the 0.25 floor | 12 | 10 |
| Groups with a real spread, of 20 | 20 | 20 |
| Groups with spread above 0.01, of 20 | 20 | 20 |
| Smallest and median spread | 0.088 and 0.512 | 0.208 and 0.607 |
| Groups with a gradient (reward range above 1e-4), of 20 | 20 | 20 |
| Groups the old repo zero test would skip | 0 | 0 |
| Top differs from top without the detector term | 3 of 20 | 3 of 20 |
| Groups with q = 0 for all four, rank picks the top | 2 | 3 |
| Any pair changes order | 9 of 20 | 10 of 20 |
| Full-reward top is the highest-z rollout | 6 of 20 | 11 of 20 |


In every cell the old normalization gives a median abs(A) of 0.85 to 0.96 for tied and differing groups alike. That is the effect described at the top of this note: the old division made a tied group as loud as any other. The floor lowers the tied-group median to 0.11 to 0.47 and leaves the differing groups at 0.72 to 0.84. The floor was active in 12 of 20 groups for both sets at w = 0.1, and in 12 and 10 at w = 0.3. In the other groups std is above 0.25 and the floor changes nothing.

The detector term changes the top-ranked rollout rarely. The comparison is against the reward without the detector term, hard * length squared * trigram squared * q. In a group where q is 0 for all four rollouts that product is 0 for all of them, so the rank alone picks the top, and those groups are counted separately. At w = 0.1 the top changes in 2 of 20 groups on set one and 0 of 20 on set two, and at w = 0.3 in 3 and 3. The highest-z rollout is the full-reward top in 5 and 10 of 20 groups at w = 0.1 and 6 and 11 at w = 0.3, against 5 expected by chance among four.
&nbsp;
### Rewards and advantages per group

In each row the entries are the four rollouts in order. `v1` is `validation-00001`. "(ties)" marks a group where every gated-in rollout has the same q. Floor A and Old A are at w = 0.3.

### Set one, base rollouts

| Group | q | Rewards, w = 0.1 | Rewards, w = 0.3 | Spread, 0.1 and 0.3 | Floor A, w = 0.3 | Old A, w = 0.3 |
| --- | --- | --- | --- | --- | --- | --- |
| v1  | 1 0 0.5 1 | 0.518 0.014 0.357 0.822 | 0.612 0.043 0.401 0.822 | 0.807 and 0.779 | 0.50 -1.48 -0.24 1.22 | 0.50 -1.48 -0.24 1.22 |
| v2  | 1 1 1 0 | 0.437 0.500 0.934 0.075 | 0.465 0.563 0.934 0.225 | 0.859 and 0.709 | -0.32 0.06 1.52 -1.26 | -0.32 0.06 1.52 -1.26 |
| v3  | 0.5 0 0 1 | 0.162 0.013 0.015 0.961 | 0.217 0.040 0.044 0.961 | 0.947 and 0.921 | -0.26 -0.73 -0.71 1.70 | -0.26 -0.73 -0.71 1.70 |
| v4  | 0 1 1 0 | 0.000 0.361 0.397 0.035 | 0.000 0.384 0.469 0.104 | 0.397 and 0.469 | -0.96 0.58 0.92 -0.54 | -1.24 0.75 1.19 -0.70 |
| v5  | 0.5 0 1 0 | 0.493 0.000 0.464 0.078 | 0.555 0.000 0.522 0.233 | 0.493 and 0.555 | 0.91 -1.31 0.78 -0.38 | 1.00 -1.44 0.86 -0.42 |
| v6  | 0 1 1 0 | 0.000 0.617 0.592 0.017 | 0.000 0.729 0.630 0.050 | 0.617 and 0.729 | -1.07 1.14 0.84 -0.92 | -1.07 1.14 0.84 -0.92 |
| v7  | 1 0.5 1 1 | 0.330 0.142 0.486 0.423 | 0.371 0.142 0.517 0.500 | 0.345 and 0.376 | -0.04 -0.96 0.54 0.47 | -0.07 -1.60 0.90 0.78 |
| v8  | 1 0.5 1 0 | 0.428 0.134 0.440 0.029 | 0.428 0.158 0.482 0.086 | 0.412 and 0.396 | 0.56 -0.52 0.77 -0.81 | 0.82 -0.77 1.14 -1.19 |
| v9  | 0 0.5 1 0 | 0.008 0.112 0.984 0.000 | 0.023 0.149 1.107 0.000 | 0.984 and 1.107 | -0.65 -0.37 1.72 -0.70 | -0.65 -0.37 1.72 -0.70 |
| v10 (ties) | 0 0 0 0 | 0.000 0.047 0.014 0.056 | 0.000 0.140 0.043 0.168 | 0.056 and 0.168 | -0.35 0.21 -0.18 0.32 | -1.27 0.76 -0.65 1.17 |
| v11 (ties) | 1 1 1 1 | 0.424 0.538 0.568 0.471 | 0.477 0.635 0.568 0.501 | 0.144 and 0.159 | -0.27 0.36 0.09 -0.18 | -1.11 1.46 0.36 -0.72 |
| v12 (ties) | 1 1 1 1 | 0.437 0.486 0.216 0.819 | 0.437 0.546 0.255 0.872 | 0.603 and 0.616 | -0.36 0.08 -1.09 1.38 | -0.40 0.08 -1.21 1.53 |
| v13  | 0 0.5 1 1 | 0.008 0.110 0.404 0.224 | 0.023 0.147 0.404 0.252 | 0.396 and 0.381 | -0.73 -0.24 0.79 0.18 | -1.31 -0.43 1.41 0.33 |
| v14  | 1 0.5 0.5 1 | 0.413 0.365 0.114 0.963 | 0.413 0.411 0.141 1.139 | 0.850 and 0.998 | -0.31 -0.31 -1.04 1.65 | -0.30 -0.31 -1.04 1.65 |
| v15  | 1 0.5 1 0 | 0.259 0.388 0.146 0.007 | 0.291 0.413 0.173 0.021 | 0.381 and 0.392 | 0.27 0.76 -0.21 -0.81 | 0.46 1.30 -0.36 -1.40 |
| v16  | 1 1 0.5 1 | 0.123 0.129 0.299 0.778 | 0.123 0.152 0.369 0.828 | 0.655 and 0.705 | -0.87 -0.76 0.00 1.63 | -0.87 -0.76 0.00 1.63 |
| v17 (ties) | 0 0 0 0 | 0.029 0.006 0.000 0.020 | 0.088 0.017 0.000 0.059 | 0.029 and 0.088 | 0.19 -0.10 -0.16 0.07 | 1.36 -0.70 -1.18 0.51 |
| v18  | 1 0 1 1 | 0.290 0.046 0.231 0.428 | 0.290 0.139 0.246 0.482 | 0.382 and 0.343 | 0.00 -0.60 -0.17 0.77 | 0.01 -1.21 -0.35 1.55 |
| v19  | 1 0 1 1 | 0.163 0.032 0.278 0.466 | 0.174 0.097 0.278 0.524 | 0.434 and 0.427 | -0.38 -0.69 0.04 1.02 | -0.58 -1.06 0.06 1.59 |
| v20  | 0 1 0.5 1 | 0.000 0.409 0.227 0.782 | 0.000 0.460 0.303 0.832 | 0.782 and 0.832 | -1.33 0.20 -0.32 1.44 | -1.33 0.20 -0.32 1.44 |

### Set two, seed-42 slope run step-80 checkpoint

| Group | q | Rewards, w = 0.1 | Rewards, w = 0.3 | Spread, 0.1 and 0.3 | Floor A, w = 0.3 | Old A, w = 0.3 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (ties) | 1 1 1 1 | 0.883 0.612 0.522 0.855 | 0.883 0.723 0.588 0.910 | 0.361 and 0.322 | 0.43 -0.21 -0.75 0.54 | 0.82 -0.40 -1.45 1.03 |
| v2  | 0 0.5 0 0 | 0.030 0.318 0.000 0.096 | 0.090 0.392 0.000 0.288 | 0.318 and 0.392 | -0.41 0.80 -0.77 0.38 | -0.66 1.28 -1.24 0.61 |
| v3 (ties) | 0 0 0 0 | 0.000 0.022 0.083 0.014 | 0.000 0.067 0.249 0.043 | 0.083 and 0.249 | -0.36 -0.09 0.64 -0.19 | -0.94 -0.24 1.67 -0.49 |
| v4 (ties) | 1 1 1 1 | 0.743 0.646 1.016 0.877 | 0.743 0.727 1.201 0.933 | 0.371 and 0.475 | -0.63 -0.70 1.20 0.13 | -0.83 -0.91 1.57 0.17 |
| v5  | 1 0 1 0 | 0.621 0.062 0.938 0.092 | 0.621 0.187 0.998 0.277 | 0.875 and 0.811 | 0.31 -1.04 1.49 -0.76 | 0.31 -1.04 1.49 -0.76 |
| v6  | 1 0.5 0.5 1 | 0.908 0.440 0.565 0.975 | 0.908 0.543 0.753 1.038 | 0.535 and 0.495 | 0.39 -1.07 -0.23 0.91 | 0.53 -1.45 -0.31 1.23 |
| v7 (ties) | 1 1 1 1 | 0.802 0.503 0.403 0.943 | 0.902 0.503 0.429 1.114 | 0.540 and 0.685 | 0.58 -0.83 -1.09 1.34 | 0.58 -0.83 -1.09 1.34 |
| v8 (ties) | 1 1 1 1 | 1.013 0.647 0.184 0.706 | 1.079 0.765 0.184 0.794 | 0.829 and 0.894 | 1.15 0.18 -1.60 0.27 | 1.15 0.18 -1.60 0.27 |
| v9  | 0.5 1 0.5 0 | 0.243 0.653 0.477 0.000 | 0.273 0.772 0.589 0.000 | 0.653 and 0.772 | -0.46 1.23 0.61 -1.38 | -0.46 1.23 0.61 -1.38 |
| v10 (ties) | 0 0 0 0 | 0.063 0.069 0.011 0.000 | 0.188 0.208 0.032 0.000 | 0.069 and 0.208 | 0.33 0.40 -0.30 -0.43 | 0.88 1.09 -0.81 -1.16 |
| v11  | 0.5 1 1 1 | 0.352 0.618 0.804 0.702 | 0.435 0.658 0.804 0.829 | 0.452 and 0.395 | -0.99 -0.09 0.49 0.59 | -1.57 -0.15 0.78 0.94 |
| v12 (ties) | 0 0 0 0 | 0.000 0.063 0.086 0.023 | 0.000 0.190 0.257 0.070 | 0.086 and 0.257 | -0.52 0.24 0.51 -0.24 | -1.29 0.61 1.27 -0.59 |
| v13  | 1 0 0 1 | 0.821 0.035 0.083 0.944 | 0.821 0.106 0.248 1.005 | 0.908 and 0.899 | 0.73 -1.16 -0.79 1.22 | 0.73 -1.16 -0.79 1.22 |
| v14  | 1 1 1 0.5 | 0.984 0.606 0.650 0.135 | 1.163 0.681 0.692 0.135 | 0.849 and 1.028 | 1.36 0.04 0.07 -1.46 | 1.36 0.04 0.07 -1.46 |
| v15  | 1 1 0 1 | 0.250 0.654 0.046 0.625 | 0.282 0.654 0.138 0.666 | 0.608 and 0.528 | -0.61 0.88 -1.19 0.92 | -0.66 0.95 -1.29 1.00 |
| v16  | 1 1 0.5 0.5 | 0.451 0.721 0.493 0.462 | 0.480 0.721 0.657 0.571 | 0.270 and 0.241 | -0.51 0.46 0.20 -0.15 | -1.40 1.25 0.55 -0.40 |
| v17  | 0.5 1 0 0.5 | 0.467 0.698 0.000 0.171 | 0.623 0.743 0.000 0.211 | 0.698 and 0.743 | 0.76 1.16 -1.31 -0.61 | 0.76 1.16 -1.31 -0.61 |
| v18  | 0.5 1 1 1 | 0.293 0.933 0.484 0.698 | 0.362 1.103 0.515 0.698 | 0.640 and 0.740 | -1.11 1.56 -0.56 0.10 | -1.11 1.56 -0.56 0.10 |
| v19 (ties) | 1 1 1 1 | 0.847 0.151 0.666 0.702 | 1.001 0.151 0.709 0.790 | 0.697 and 0.851 | 1.08 -1.63 0.15 0.40 | 1.08 -1.63 0.15 0.40 |
| v20  | 1 1 0 0.5 | 0.617 0.562 0.005 0.094 | 0.729 0.633 0.016 0.094 | 0.612 and 0.713 | 1.14 0.84 -1.11 -0.87 | 1.14 0.84 -1.11 -0.87 |

&nbsp;
## Runs

The dry run passed, so two runs started. Both used seed 42, `Qwen/Qwen3-0.6B-Base`, group size 4, `max_new_tokens` 700, temperature 0.8, `top_p` 0.9, AdamW at 1e-5, gradient clip 1.0, no KL, bfloat16, gradient checkpointing, the 200-word prompt, the std-floor advantage, length score squared, and the reward-range skip. The run with w = 0.1 is `results/grpo-length-200-floor-w01` and the run with w = 0.3 is `results/grpo-length-200-floor-w03`. The script is `/tmp/grpo_len200_floor_train.py` with `RANK_W` set. Weights are in `results/grpo-length-200-floor-w01-weights` and `results/grpo-length-200-floor-w03-weights` on the lab machine and are not committed. The checks and stop rules are those of the slope run: Opus at steps 20, 40, and 60 on 50 validation prompts, step 80 on 100 prompts with Opus and `gpt-5.5`, both orders, agreed pairs. The stop rules are an Opus win rate under 40%, under 45% twice in a row, the gate-fire count rising for 20 steps, more than half of the last 10 steps with no gradient, and a median length over 350 words.

The two runs shared GPU 0. GPUs 1 and 2 had about 5 GB free, which is not enough, and GPU 3 was left alone for the breakfast job. Each run used 6 to 7 GB, so both fit beside the other user's 6 GB process on GPU 0. The second run had its memory capped at a fraction of 0.35. The tmux sessions were `grpo-floor-w01` and `grpo-floor-w03`.

Neither run reached step 80. Both hit a stop rule. The w = 0.3 run stopped at step 20 on the Opus rule, and the w = 0.1 run stopped at step 37 on the median-length rule. No run had a step with no gradient, and the reward judge had 0 parse failures in both.
&nbsp;
### w = 0.3

The run stopped at step 20. The recorded reason is an Opus win rate under 40%. That rate is 13 of 48 agreed pairs, which is 0.271. Within 20% length it is 5 of 29, or 0.172. The gate-fire rise streak peaked at 1, with 3 fires in total, and the median length over 10 steps peaked at 174.5 words. Reward-judge parse failures are 0 over 154 calls and Opus parse failures are 0. Over the 20 steps the 80 rollouts were judged as rollout 29, base 39, split 6, tie 3, and 3 were gated out.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 13/48 | 5/29 | 0.424078 | 0.137102 | 0.560591 | 0.147150 |

The detector scores are already high at step 20. The trained answers have a mean raw logit z of -0.86 against -4.39 for the matched base, and a sigmoid three times the base. This is the same outcome as the rank run, reached in 20 steps and not 40. The floor and the squared length did not stop the detector from driving the update at this weight.

| steps | mean q of judged rollouts | mean words | mean sigmoid | mean DistilBERT | mean trigram ratio | mean reward | gate fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 to 10 | 0.388 | 185 | 0.112 | 0.236 | 0.970 | 0.222 | 0 |
| 11 to 20 | 0.475 | 173 | 0.196 | 0.365 | 0.966 | 0.284 | 3 |

| Part | Calls | USD |
| --- | ---: | ---: |
| Reward judge, `claude-sonnet-5-5` | 154 | 0.348 |
| Opus check, `claude-opus-5-5` | 100 | 0.888 |
| `gpt-5.5` | 0 | 0.000 |
| Total | 254 | 1.236 |
&nbsp;
### w = 0.1

The run stopped at step 37. The recorded reason is a median rollout length of 368.5 words over the last 10 steps, which is over 350. The median over the last 10 steps was 174.5 at step 30 and 266.5 at step 35, 274 at step 36, and 368.5 at step 37. The step-20 check was 21 of 41 agreed pairs, which is 0.512, so no Opus rule fired. 9 gate fires came in 7 steps, 8, 9, 30, 32, 33, 35, and 36, and the rise streak peaked at 2. Reward-judge parse failures are 0 over 278 calls and Opus parse failures are 0. Over the 37 steps the 148 rollouts were judged as rollout 60, base 57, split 18, tie 4, and 9 were gated out. There was no step-40 check, so the table has one row.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 21/41 | 18/33 | 0.121667 | 0.188124 | 0.167181 | 0.183812 |

At step 20 the win rate is 51.2% on all pairs and 54.5% within 20% length, with 9 Opus ties. The 95% Wilson interval on all pairs is 0.36 to 0.66. The trained answers average 199 words against 182 for the base, and both detector scores are below the base, as in the slope runs. At this weight the detector term does not move the policy. The length is what failed, and the squared length score did not hold it. The slope run at seed 42 never passed a 10-step median of 205 words. This run reached 368.5 words by step 37 even with the squared score, as the slope run at seed 2 did.

| steps | mean q of judged rollouts | mean words | mean sigmoid | mean DistilBERT | mean trigram ratio | mean reward | gate fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 to 10 | 0.458 | 243 | 0.097 | 0.175 | 0.963 | 0.219 | 2 |
| 11 to 20 | 0.550 | 193 | 0.005 | 0.015 | 0.977 | 0.293 | 0 |
| 21 to 30 | 0.571 | 230 | 0.113 | 0.257 | 0.952 | 0.312 | 1 |
| 31 to 37 | 0.446 | 368 | 0.075 | 0.127 | 0.944 | 0.167 | 6 |

| Part | Calls | USD |
| --- | ---: | ---: |
| Reward judge, `claude-sonnet-5-5` | 278 | 0.698 |
| Opus check, `claude-opus-5-5` | 100 | 1.063 |
| `gpt-5.5` | 0 | 0.000 |
| Total | 378 | 1.761 |

The w = 0.1 run took about 1 hour 5 minutes of wall time and the w = 0.3 run about 41 minutes.
&nbsp;
## Advantage sizes during training

Each step is classed as tied when every gated-in rollout of its group has the same q, and differing otherwise. The statistics pool the four rollouts of each step in the class. The old columns are the std + 1e-4 normalization on the same logged rewards.

| Run | Class | Steps | Floor abs(A) mean, median | Old abs(A) mean, median | Mean of the largest abs(A) |
| --- | --- | ---: | --- | --- | ---: |
| w = 0.1 | q ties | 3 | 0.562, 0.398 | 0.838, 0.677 | 1.124 |
| w = 0.1 | q differs | 34 | 0.660, 0.512 | 0.871, 0.787 | 1.172 |
| w = 0.3 | q ties | 5 | 0.405, 0.231 | 0.883, 0.954 | 0.648 |
| w = 0.3 | q differs | 15 | 0.603, 0.542 | 0.873, 0.879 | 1.029 |

The tied-group advantages are smaller than the differing-group advantages under the floor, as in the dry run, and the old normalization gives about the same size in both classes. The difference is clear at w = 0.3, with a median of 0.23 against 0.54, and small at w = 0.1, with 0.40 against 0.51. The comparison is weak for a more basic reason: tied steps are rare. Only 3 of 37 steps at w = 0.1 and 5 of 20 at w = 0.3 had all gated-in rollouts at the same q, so the premise that q ties in most groups does not hold in these runs. The average abs(A) over all steps is 0.652 at w = 0.1 and 0.554 at w = 0.3. These are advantage magnitudes, not weight changes, because the clip and AdamW rescale the update.

The step-level correlation of the advantage with q and with the rank shows where the update direction comes from. Over the steps of each run, the mean Pearson correlation of the four advantages with the four q values is 0.80 at w = 0.1, 0.82 at w = 0.3, and 0.91 in the rank run. The mean correlation with the rank r is -0.05, 0.12, and 0.09. So the advantages follow q almost entirely at the step level, and the rank term shows up as a small positive correlation at w = 0.3 and none at w = 0.1. That is a description and not a causal test. It does not show that the rank term has no effect, because a small consistent push on the detector across 20 steps can still move the policy, and at w = 0.3 it did. It shows that the large advantages are not coming from tied groups in these logs.
&nbsp;
## What the runs show

The std floor and the squared length did not give a run that holds quality and raises the detector. At w = 0.3 the detector drove the policy and quality fell within 20 steps. At w = 0.1 quality held at step 20 and the detectors stayed at or below the base, and the length then drifted past the stop limit by step 37. Compared with the earlier runs at the first check they reached:

| Run | Check step | Opus agreed pairs | Trained sigmoid | Base sigmoid | Trained DistilBERT | Base DistilBERT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Slope, seed 42 | 20 | 25/46 | 0.079 | 0.195 | 0.093 | 0.191 |
| Rank, seed 42 | 20 | 24/45 | 0.140 | 0.145 | 0.150 | 0.129 |
| Std floor, w = 0.1 | 20 | 21/41 | 0.122 | 0.188 | 0.167 | 0.184 |
| Std floor, w = 0.3 | 20 | 13/48 | 0.424 | 0.137 | 0.561 | 0.147 |

The w = 0.3 run reached the detector gain and the quality loss that the rank run took 40 steps to reach. This suggests the floor did not weaken the detector's pull at w = 0.3, though one seed per weight cannot rule out a difference in speed from run noise. At w = 0.1 the detector term is too weak to separate rollouts, which is what the slope run's sigmoid term showed. The weights between 0.1 and 0.3 were not tried, and seed 2 was not run for either weight.
