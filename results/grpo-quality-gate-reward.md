&nbsp;
# Quality-gate reward

The length-matched T sweep, `results/grpo-length-200-t-sweep.md`, showed a trade-off. T = 1 raised the detector scores and lost on quality. T = 2 and T = 4 kept quality and did not become more human. This note writes a reward that makes quality a condition and records a dry run of it at T = 1. No training happened for the dry run.
&nbsp;
## Reward

The comparison is each gated-in rollout against its matched base answer. The matched base is the gated-in sample among the eight frozen base answers whose word count is closest to the rollout. A tie on the absolute difference takes the lowest pool index. Wins against the other rollouts are dropped.

q_i = 1 if `claude-sonnet-5-5` does not prefer the matched base in either order, and 0 otherwise.

reward_i = hard_i * q_i * trigram_ratio_i ** 2 * sigmoid(z_i / T) * length_score_i

The gate, the 203 length target, the 100 to 300 word window, the 700-token hit, and `z_i` are as in `results/grpo-training-config.md`. The judge is the JSON schema, no tools, no temperature, in both orders. A rollout is judged only when `hard_i` is 1, so a gated-out rollout costs no call. `N` is no longer used.

The wording "win or tie, both orders" has two readings. This note uses the strict one. The rollout must come out as a win or a tie in each order, so a single order that names the base sets q_i to 0. A pair where the two orders name different answers because the judge picked one slot twice therefore gives q_i = 0. The lenient reading sets q_i to 0 only when both orders name the base. The dry run computes both from the same calls. Only the strict q_i is used for training.

A call that does not parse sets q_i to 0, so a failed call never opens the gate. If a gated-in rollout has no gated-in matched base, its reward is 0.
&nbsp;
## Dry run

T = 1. The 20 groups are `validation-00001` through `validation-00020`. The rollouts, the pool, and the matched bases are the 80 rollouts and 160 pool answers of `results/grpo-length-matched-dry-run.md`. Those rollout-against-match pairs were already judged in both orders, so this dry run read the stored verdicts and sent no new request. The stored token counts give the cost.

Of the 80 rollouts, 51 are gated in and 29 are not. All 51 have a matched base. There were 0 parse failures. Over the 51 gated-in rollouts, both orders named the rollout 26 times, both orders named the base 18 times, the orders split 5 times, and 2 pairs had a tie in one or both orders with no vote for the base.

Strict q_i is 1 for 28 of the 51 gated-in rollouts, which is 0.549, or 28 of all 80 rollouts, which is 0.350. Lenient q_i is 1 for 33 of 51, which is 0.647.

A real spread is a reward range above 1e-4 with a maximum above 0, as in the earlier dry runs. Strict has a real spread in 15 of 20 groups. The bar is at least 15, so it passes with no margin. Lenient has 16 of 20.

The spread is small in most groups. At a range above 1e-3 the count is 11 of 20, and above 1e-2 it is 5 of 20. The frozen base rarely scores high on the detector, so most nonzero rewards are near 1e-3 and a few are near 0.7. In 8 groups q_i differs inside the group, so the gate changes the ranking there. The flat groups under strict are `validation-00010`, `validation-00015`, `validation-00017`, `validation-00018`, and `validation-00019`. In the first three, every gated-in rollout lost to its base. In `validation-00018` only one rollout was gated in and it lost, and in `validation-00019` none was gated in.

The GRPO advantage is the reward minus the group mean, divided by the group standard deviation plus 1e-4. A group with a range near 1e-3 has a standard deviation near 4e-4, so it keeps most of its advantage. A group with a range below 1e-4 keeps almost none.

| prompt | r0 | r1 | r2 | r3 | q r0 r1 r2 r3 | range | real |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| validation-00001 | 0.579740 | 0.000000 | 0.706741 | 0.005194 | 1 - 1 1 | 0.706741 | yes |
| validation-00002 | 0.000000 | 0.000000 | 0.000933 | 0.000000 | 0 0 1 0 | 0.000933 | yes |
| validation-00003 | 0.000000 | 0.000000 | 0.000000 | 0.000667 | - - - 1 | 0.000667 | yes |
| validation-00004 | 0.000000 | 0.010541 | 0.267378 | 0.000000 | 0 1 1 0 | 0.267378 | yes |
| validation-00005 | 0.000000 | 0.000000 | 0.000932 | 0.000000 | 0 0 1 0 | 0.000932 | yes |
| validation-00006 | 0.000000 | 0.734499 | 0.138513 | 0.000000 | 0 1 1 0 | 0.734499 | yes |
| validation-00007 | 0.001666 | 0.000000 | 0.001416 | 0.003155 | 1 0 1 1 | 0.003155 | yes |
| validation-00008 | 0.000678 | 0.000000 | 0.000754 | 0.000000 | 1 - 1 0 | 0.000754 | yes |
| validation-00009 | 0.000000 | 0.000000 | 0.003016 | 0.000000 | - - 1 - | 0.003016 | yes |
| validation-00010 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 0 0 0 | 0.000000 | no |
| validation-00011 | 0.003903 | 0.665268 | 0.003796 | 0.003867 | 1 1 1 1 | 0.661471 | yes |
| validation-00012 | 0.000000 | 0.003497 | 0.000000 | 0.000779 | - 1 - 1 | 0.003497 | yes |
| validation-00013 | 0.000000 | 0.000000 | 0.001064 | 0.000000 | - - 1 - | 0.001064 | yes |
| validation-00014 | 0.000647 | 0.000000 | 0.000000 | 0.042628 | 1 0 - 1 | 0.042628 | yes |
| validation-00015 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 0 - - | 0.000000 | no |
| validation-00016 | 0.000000 | 0.000000 | 0.001808 | 0.002007 | - - 1 1 | 0.002007 | yes |
| validation-00017 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 - - 0 | 0.000000 | no |
| validation-00018 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - 0 - - | 0.000000 | no |
| validation-00019 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - - - - | 0.000000 | no |
| validation-00020 | 0.000000 | 0.000000 | 0.000000 | 0.001566 | - - 0 1 | 0.001566 | yes |

A dash is a gated-out rollout. Its reward is 0 and it is not judged.
&nbsp;
## Cost per step

Each gated-in rollout needs 2 calls. The mean is 5.1 calls and $0.01125 per step over the 20 groups. The largest group is 8 calls and $0.0178. A group with no gated-in rollout is $0. These are the stored token counts at $2 per million input tokens and $10 per million output tokens. The old reward judged every pair among the gated-in rollouts and the base, up to 20 calls a step. The T = 2 run averaged $0.0302 per step for the reward judge.

An 80-step run at this rate is about $0.90 for the reward judge. The evaluation judges are the same as the sweep's: about $1.0 for each 50-prompt Opus check, $2.1 for the step-80 Opus check on 100 prompts, and $1.1 for `gpt-5.5`. The full run is expected to cost near $7.
&nbsp;
## Decision

The dry run has a real spread in 15 of 20 groups, which meets the bar. The T = 1 run followed with seed 42, 80 steps planned, from scratch, the 200-word prompt, and the same checks and stop rules as the sweep. The reward, the dry run, and the run share the one q_i definition above. The run stopped at step 40. The record is in the next section.
&nbsp;
## T = 1 run

The run stopped at step 40 on the rule "under 45% at two checks in a row". The Opus win rate was 19 of 47 at step 20 (0.4043) and 18 of 43 at step 40 (0.4186). The gate-fire rise streak peaked at 2 and never fired. Seed 2 did not start. Weights are `results/grpo-length-200-qgate-t1-weights` on the lab machine. The step log is `results/grpo-length-200-qgate-t1/run.log`.

Quality and human score at each check, trained against the length-matched base. The matched base is the gated-in pool sample closest in word count to each trained answer, so for long trained answers it is a base answer near 300 words. That is why the base means differ from the sweep.

| step | Opus agreed pairs | within 20% length | sigmoid(z / 1), trained | sigmoid(z / 1), base | DistilBERT, trained | DistilBERT, base | trained answers gated out | trained mean words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 19/47 | 2/3 | 0.031439 | 0.106336 | 0.072059 | 0.174049 | 47 of 50 | 508 |
| 40 | 18/43 | 2/3 | 0.044962 | 0.106397 | 0.083364 | 0.174740 | 48 of 50 | 510 |

Mean `z` was -5.31 for trained and -4.82 for base at step 20, and -5.15 against -4.80 at step 40. Both detector means for the trained answers are below the base's, not above. Only 3 pairs at each check fall inside the length band, so the band rate of 2 of 3 is not informative. Ties were 3 at step 20 and 7 at step 40.

The cause is length. By step 5 the policy wrote about 440 words, and from step 8 most steps were four rollouts at 450 to 600 words. Over the 40 steps, 136 of 160 rollouts were gated out. That is 134 over 300 words and 77 at the 700-token cap, which overlap, plus 2 under 100. The window rates were 24 of 40 in steps 1 to 10, then 35, 38, and 39 of 40. At the checks, 47 and 48 of 50 trained answers were over 300 words, with a median near 540. The two step-40 answers I read are long essays with `###` and bold headings.

The reward then gave no signal. In 28 of 40 steps all four rollouts were gated out, so no pair was judged, every reward was 0, and the loss was 0. Only 7 steps had a reward range above 1e-4, steps 1, 2, 3, 6, 7, 14, and 23. Only 24 rollouts were judged in the whole run, and 10 of them had q = 1. The sweep's T = 1 run, which had the same seed and the same first rollouts, had 36 of 40 steps with a range and one step with all four gated out. The old reward gave the surviving rollouts a win fraction against each other and the base. This reward judges only a rollout against its base, so a group with one or two survivors and q = 0 is flat.

Two causes are plausible and neither was tested. The reward values are near 1e-3, and the advantage divides by the group standard deviation plus 1e-4, so a noise-level difference in one group can give a full-size update at a learning rate of 1e-5. And a step with zero advantage still calls the AdamW step, which keeps moving the weights on stored momentum. Either could push the policy toward longer text during steps 4 to 8. Once all four rollouts were gated out, nothing pulled it back.

The dry run did not predict this. It used the frozen base's rollouts, of which 51 of 80 were gated in. The trained policy's rollouts were gated in 24 of 160 times.

Cost: $3.049356 in total. The reward judge was $0.108496 over 48 calls with 0 parse failures, and Opus was $2.940860 over 200 calls with 0 parse failures. `gpt-5.5` was not called.
&nbsp;
## Confound: zero-advantage steps were not skipped

The T = 1 run of this reward did not skip zero-advantage steps. Its training loop called `optimizer.step()` on every step, so on a step with all advantages zero AdamW still applied its stored momentum and weight decay and moved the weights. By the repo criterion, 33 of its 40 steps had advantages all zero. The earlier count of 28 steps with no judged rollout leaves out 5 steps whose rewards were all equal. So most of the run's weight changes after the first steps came from momentum. The drift to answers over 300 words and the stop at step 40 are confounded on this point, and the logs cannot separate the cause. The run was not repeated with the skip. Details and the later runs that skip these steps are in `results/grpo-slope-reward.md`.
