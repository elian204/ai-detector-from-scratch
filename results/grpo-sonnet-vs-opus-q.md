&nbsp;
# Does the training judge see the quality loss

The explanation for the rank run and the w = 0.3 std-floor run is a weighted-sum trade-off. A policy gives up some average q when 0.3 times its rank gain is worth more than the q it loses, so w = 0.3 collapses quality and w = 0.1 moves nothing. This note checks one thing the adaptive-constraint idea depends on: whether the training judge, `claude-sonnet-5-5`, sees the quality loss that the evaluation judge, `claude-opus-5-5`, sees. It also checks how often length matching fails. It uses the saved step logs only. No training ran and no new judge call was made.
&nbsp;
## Method

q is the Sonnet verdict of a training rollout against its matched base, 1 if both orders pick the rollout, 0 if both pick the base, 0.5 otherwise. The mean training q of a block is the mean over the judged rollouts of its steps, about 20 per 5-step block. The Opus checks are on 50 or 100 validation prompts, both orders, agreed pairs, and the win rate is trained over trained plus base. The two judges see different texts. Sonnet judges the rollouts sampled during training on training prompts. Opus judges one sampled answer per validation prompt from the checkpoint at the check step. Both are matched to the base by word count. So the numbers can be compared in level and direction, and not pair by pair.

As a second view, "decided win" is rollouts that beat their base in both orders over rollouts that beat or lost to it, with ties and splits left out. This is the closest Sonnet number to the Opus win rate, which also drops ties.
&nbsp;
## Mean training q per 5-step block and the Opus checks

Each cell is the mean q of the block, with the Opus win rate at a check step in brackets, placed in the block that contains it. Blank cells are steps the run did not reach.

| Steps | Slope s42 | Slope s2 | Rank s42 (w 0.3) | Floor w 0.1 | Floor w 0.3 |
| --- | --- | --- | --- | --- | --- |
| 1 to 5 | 0.39 | 0.57 | 0.45 | 0.35 | 0.45 |
| 6 to 10 | 0.50 | 0.50 | 0.62 | 0.56 | 0.33 |
| 11 to 15 | 0.47 | 0.72 | 0.35 | 0.53 | 0.47 |
| 16 to 20 | 0.60 (Opus step 20: 25/46 = 54%) | 0.68 (Opus step 20: 28/45 = 62%) | 0.65 (Opus step 20: 24/45 = 53%) | 0.57 (Opus step 20: 21/41 = 51%) | 0.50 (Opus step 20: 13/48 = 27%) |
| 21 to 25 | 0.62 | 0.70 | 0.37 | 0.62 |  |
| 26 to 30 | 0.60 | 0.53 | 0.53 | 0.53 |  |
| 31 to 35 | 0.68 | 0.55 | 0.26 | 0.37 |  |
| 36 to 40 | 0.57 (Opus step 40: 22/39 = 56%) | 0.92 (Opus step 40: 27/45 = 60%) | 0.50 (Opus step 40: 13/48 = 27%) | 0.57 |  |
| 41 to 45 | 0.55 | 0.75 |  |  |  |
| 46 to 50 | 0.65 |  |  |  |  |
| 51 to 55 | 0.60 |  |  |  |  |
| 56 to 60 | 0.80 (Opus step 60: 28/45 = 62%) |  |  |  |  |
| 61 to 65 | 0.68 |  |  |  |  |
| 66 to 70 | 0.60 |  |  |  |  |
| 71 to 75 | 0.60 |  |  |  |  |
| 76 to 80 | 0.57 (Opus step 80: 49/83 = 59%) |  |  |  |  |

The Opus check at step 20 of the w = 0.3 run was 13 of 48. Sonnet's q for that run was 0.45 in steps 1 to 5, 0.33, 0.47, and 0.50 in steps 16 to 20. The run's Opus win rate fell to 27% and its Sonnet q did not move. For the 10 steps before the check, 16 rollouts beat their base and 17 lost, a decided win rate of 0.48, with a 95% Wilson interval of 0.33 to 0.65. A rate of 0.27 is outside that interval, though only just, because n is 33.
&nbsp;
## Sonnet against Opus at each check

Sonnet's decided win rate is over the 10 steps that end at the check step.

| Run | Check step | Opus agreed pairs | Opus win rate | Sonnet rollout vs base, last 10 steps | Sonnet decided win (95% interval) | Mean q |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Slope s42 | 20 | 25/46 | 0.54 | 19 vs 16 | 0.54 (0.38 to 0.70) | 0.54 |
| Slope s42 | 40 | 22/39 | 0.56 | 20 vs 10 | 0.67 (0.49 to 0.81) | 0.63 |
| Slope s42 | 60 | 28/45 | 0.62 | 27 vs 11 | 0.71 (0.55 to 0.83) | 0.70 |
| Slope s42 | 80 | 49/83 | 0.59 | 22 vs 15 | 0.59 (0.43 to 0.74) | 0.59 |
| Slope s2 | 20 | 28/45 | 0.62 | 24 vs 9 | 0.73 (0.56 to 0.85) | 0.70 |
| Slope s2 | 40 | 27/45 | 0.60 | 24 vs 7 | 0.77 (0.60 to 0.89) | 0.72 |
| Rank s42 | 20 | 24/45 | 0.53 | 18 vs 18 | 0.50 (0.34 to 0.66) | 0.50 |
| Rank s42 | 40 | 13/48 | 0.27 | 13 vs 22 | 0.37 (0.23 to 0.54) | 0.37 |
| Floor w 0.1 | 20 | 21/41 | 0.51 | 20 vs 16 | 0.56 (0.40 to 0.70) | 0.55 |
| Floor w 0.3 | 20 | 13/48 | 0.27 | 16 vs 17 | 0.48 (0.33 to 0.65) | 0.49 |

Across the 10 checks the Sonnet decided win rate and the Opus win rate have a Spearman correlation of 0.92 and a Pearson correlation of 0.82. The slope runs, which hold quality, are high on both, and the lowest Opus values, 0.27 twice, are the lowest or near-lowest Sonnet values. The agreement is partly a level effect between runs. Within a run the picture is mixed.

- Rank run, seed 42. Opus fell from 0.53 at step 20 to 0.27 at step 40. Sonnet fell with it. The decided win rate went from 0.50 to 0.37, and the mean q from 0.50 to 0.37. The block q in steps 31 to 35 is 0.26, with 5 rollouts beating the base and 14 losing. The Sonnet drop is smaller than the Opus drop, and the interval at step 40, 0.23 to 0.54, includes 0.5. The block values also swing, 0.65, 0.37, 0.53, 0.26, 0.50, so a single block is not a reliable reading.
- Floor w = 0.3. Opus put the run at 0.27 at step 20, and Sonnet put it at 0.48. The run's first Sonnet block, steps 1 to 5, is 0.45, before any real training, so the Sonnet q after 20 steps is at its starting level. Sonnet did not see this drop.
- Slope seed 2 at step 40. The Sonnet decided win rate is 0.77 and the block q over steps 36 to 40 is 0.92, with 15 rollouts winning and none losing. Opus says 0.60. In that run Sonnet reads more quality than Opus does, which fits a policy that wrote longer answers than their matched bases, as in the next section.

The answer to the question is mixed, and it depends on the run. In the rank run Sonnet's q fell as Opus fell, by a smaller amount and with wide intervals. In the w = 0.3 floor run, the run that collapsed fastest, Sonnet's q did not fall at all while Opus fell from about even to 27%. Sonnet did not clearly track the Opus drop, and it missed the larger of the two drops.
&nbsp;
## What a Lagrangian on q would have done on these logs

A rough check of the adaptive constraint on the logged q. The reward lam * q + r is q + r / lam up to scale, so lam = 3 is a rank weight of 0.33, close to the collapsing runs. The proposed update is lam = max(0.5, lam + 2 * (0.5 - qbar)), with qbar the mean q over the last 5 steps. Replaying the logged q values through that rule, open loop, so the policy cannot respond, gives these lam values.

| Run | Step 5 | Step 10 | Step 15 | Step 20 | Step 30 | Step 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Floor w 0.3 | 3.52 | 4.82 | 5.91 | 6.03 | | |
| Rank s42 | 3.81 | 2.81 | 3.51 | 2.66 | 3.21 | 6.05 |
| Floor w 0.1 | 3.47 | 4.17 | 3.85 | 3.45 | 1.35 | |
| Slope s42 | 3.47 | 4.13 | 4.68 | 3.68 | 1.78 | 0.50 |

In the w = 0.3 run, the replayed lam rises to about 6, which is a rank weight of 0.17. It rises because the training q starts below 0.5, 0.33 to 0.47 in the first 15 steps. It is not responding to a collapse in quality, since Sonnet's q did not drop. In the rank run lam stays between 2.7 and 3.8 through step 30, a rank weight of 0.26 to 0.38, and only reaches 6 at step 40, after the collapse is visible in Opus. This is not a simulation of the controller, because the policy would change its behavior under it. It only shows that a controller driven by Sonnet's q would have seen no clear fall in the one run that collapsed within 20 steps.
&nbsp;
## Length matching fails

The matched base is the gated-in pool sample closest in word count. When a rollout is longer than all 8 pool answers there is no base of that length. The match is then the longest gated-in pool answer, which is shorter than the rollout, so the judge compares a longer answer with a shorter one. The tables count rollouts whose word count is above the maximum of all 8 pool answers for the prompt. They give the share of all rollouts in the block, the number that were judged against a matched base, and the mean ratio of the rollout's words to the matched base's words.

Slope seed 2, 41 steps.

| Steps | Rollouts longer than all 8 | Share | Judged | Mean words over matched words |
| --- | ---: | ---: | ---: | ---: |
| 1 to 5 | 4 of 20 | 0.20 | 4 | 1.48 |
| 6 to 10 | 7 of 20 | 0.35 | 7 | 1.18 |
| 11 to 15 | 2 of 20 | 0.10 | 2 | 1.13 |
| 16 to 20 | 8 of 20 | 0.40 | 8 | 1.20 |
| 21 to 25 | 4 of 20 | 0.20 | 4 | 1.41 |
| 26 to 30 | 11 of 20 | 0.55 | 10 | 1.17 |
| 31 to 35 | 5 of 20 | 0.25 | 5 | 1.20 |
| 36 to 40 | 10 of 20 | 0.50 | 8 | 1.48 |
| 41 | 0 of 4 | 0.00 | 0 | |
| All | 51 of 164 | 0.31 | 48 | median 1.19 |

Floor w = 0.1, 37 steps.

| Steps | Rollouts longer than all 8 | Share | Judged | Mean words over matched words |
| --- | ---: | ---: | ---: | ---: |
| 1 to 5 | 2 of 20 | 0.10 | 2 | 1.05 |
| 6 to 10 | 3 of 20 | 0.15 | 1 | 1.00 |
| 11 to 15 | 1 of 20 | 0.05 | 1 | 1.12 |
| 16 to 20 | 0 of 20 | 0.00 | 0 | |
| 21 to 25 | 5 of 20 | 0.25 | 5 | 1.77 |
| 26 to 30 | 8 of 20 | 0.40 | 7 | 1.39 |
| 31 to 35 | 11 of 20 | 0.55 | 6 | 1.19 |
| 36 to 37 | 2 of 8 | 0.25 | 1 | 1.25 |
| All | 32 of 148 | 0.22 | 23 | median 1.18 |

The slope seed-42 run, which did not drift, is the reference. Its share is 0.11 over 80 steps, 34 of 320, and 0.15 or less in every block until steps 71 to 80, where it is 0.30. Some of the unjudged rollouts are runaways, which are gated out and get no match.

The matched base is a shorter answer by about 20% at the median, and the ratio is 1.4 to 1.8 in some blocks. The effect on q differs by run. In slope seed 2 the rollouts longer than all 8 had a mean q of 0.75 over 48 judged, against 0.60 for the other 110, so the judge favored the longer answer when no length-matched base existed. In the w = 0.1 run the same rollouts had a mean q of 0.39 over 23, against 0.53 for the others, so the longer answer did not win there. In the rank run the figures are 0.32 over 14 against 0.48, and in the slope seed-42 run 0.67 over 33 against 0.59. The matching failure is therefore one route by which seed 2 can earn q from length, but the w = 0.1 run shows it is not a consistent gain. The failure grows with the drift in length, both before the stop rule fired.
&nbsp;
## Decision

The rule for the adaptive constraint was to run it if Sonnet's q fell clearly in a way that tracks the Opus drop, and to stop and report if not. The evidence is mixed and the clearest case goes against it. In the w = 0.3 floor run Opus fell to 27% by step 20 and Sonnet's q stayed at 0.45 to 0.50. In the rank run Sonnet fell with Opus, from 0.50 to 0.37, though by less and with an interval that includes 0.5. Across the 10 checks the two judges correlate at 0.92, driven largely by the difference between runs that hold quality and runs that do not. So Sonnet reads the overall level of quality, and it missed the fastest collapse. Under the rule, no constraint run was started. The constraint would be driven by a q that did not show the one collapse that happened in 20 steps.

The adaptive constraint is not run, and no training started. The reward, `reward_i = len_factor_i * trigram_ratio_i ** 2 * (lam * q_i + r_i)` with the exponential penalty above 300 words, is unchanged and unimplemented.
