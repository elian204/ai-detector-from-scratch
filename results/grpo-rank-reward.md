&nbsp;
# Rank reward

The slope run, `results/grpo-slope-reward.md`, beat its base on quality and did not raise either detector score. Its detector term, 0.3 * sigmoid(z), was near 0 for almost every rollout, so it did not separate rollouts. This note replaces the sigmoid with a rank inside the group, so the detector term always spans its full range of 0 to 0.3. It records a dry run on two sets of 20 groups. The run that followed is at the end of the note.
&nbsp;
## Reward

q_i, length_score_i, trigram_ratio_i, hard_i, the matched base, and the judge are unchanged from the slope reward. The change is the detector term.

r_i = the rank of z_i among the group's four rollouts, scaled to [0, 1]. The lowest z gets 0 and the highest gets 1. Tied z values share the average rank. z_i is the raw human-class logit of `qwen3-variable` before the saved temperature.

reward_i = hard_i * length_score_i * trigram_ratio_i ** 2 * (q_i + 0.3 * r_i)

The rank is taken over all four rollouts of the group, including any that the gate zeroes. A gated-out rollout still has reward 0. The rank is scaled as (average rank - 1) / (4 - 1), so with no ties the four values are 0, 1/3, 2/3, and 1. The code in `/tmp/grpo_len200_rank_train.py` is:

```python
def rank_scores(values):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        end = pos
        while end + 1 < n and values[order[end + 1]] == values[order[pos]]:
            end += 1
        for k in range(pos, end + 1):
            ranks[order[k]] = (pos + end) / 2.0
        pos = end + 1
    return [rank / (n - 1) for rank in ranks]

def slope_reward(hard, q, tri, rank, length):
    if hard == 0:
        return 0.0
    return hard * length * (tri ** 2) * (q + 0.3 * rank)
```

The rank ignores how far apart the z values are. A group whose four z values all sit near -5 gets the same 0, 1/3, 2/3, 1 as one whose values span -8 to 3. So the detector term is now a within-group ordering, and it cannot reward a rollout for raising z across groups. A group with identical z values gives every rollout r = 0.5. The advantage, the zero-advantage skip, and the stop rules are as in the slope run. The training script skips backward and `optimizer.step()` on a zero-advantage step.
&nbsp;
## Dry run

The 20 groups are `validation-00001` through `validation-00020`, four rollouts each, temperature 0.8, `top_p` 0.9, 700 tokens, the 200-word prompt. Set one is the 80 saved base-model rollouts. Set two is 80 new rollouts from the seed-42 slope run step-80 checkpoint, `results/grpo-length-200-slope-t1-weights`, generated with seed 42. The matched bases are the same as in the slope dry run. The set-one pairs came from the call cache. The 160 set-two calls were new.

The dry run passes. Both sets have a real spread, a range above 1e-4 with a maximum above 0, in 20 of 20 groups. The bar was 15.
&nbsp;
### Summary

| Item | Set one, base | Set two, slope step 80 |
| --- | --- | --- |
| Groups with a real spread, of 20 | 20 | 20 |
| Groups with spread above 0.01, of 20 | 20 | 20 |
| Smallest and median spread | 0.150 and 0.638 | 0.144 and 0.574 |
| Groups with a nonzero advantage, so a gradient, of 20 | 20 (1.00) | 20 (1.00) |
| Smallest largest-absolute-advantage in a group | 1.18 | 1.17 |
| Parse failures, of 80 | 0 | 0 |
| Gate fires, of 80 | 0 | 0 |
| Verdicts: rollout, base, split, tie | 42, 26, 9, 3 | 44, 23, 11, 2 |
| q = 1, 0.5, 0 | 42, 12, 26 | 44, 13, 23 |
| Groups where q differs inside the group, of 20 | 16 | 12 |
| Mean reward | 0.470 | 0.616 |
| Words, median and mean | 188 and 214.8 | 214.5 and 246.7 |
| Rollouts over 300 words, of 80 | 19 | 14 |
| USD per step, mean (min to max) | 0.0194 (0.0125 to 0.0301) | 0.0204 (0.0152 to 0.0284) |
| New API requests | 0 | 160 |

Every group has a spread above 0.01, which the slope dry run did not reach on set one. The two weak groups there, where all four rollouts had q = 0, now spread by up to 0.3 * length * trigram squared from the rank, so the smallest spread is 0.15.

The step-80 checkpoint beats its matched bases in 44 of 80 rollouts and loses in 23, against 42 and 26 for the base rollouts. That agrees with the Opus win rate of the slope run at step 80.
&nbsp;
### Does the detector term change the top rollout

A group's top is the rollout with the highest reward. Two comparisons are made, and the tie handling differs.

Against the reward without the detector term, hard_i * length_score_i * trigram_ratio_i ** 2 * q_i. Here the top is the highest value of that product. A group where every q_i is 0 gives that product 0 for all four rollouts, so it has a four-way tie, and the rank alone picks the top. Those groups are counted separately and not as a "change". A group counts as changed when the top of the full reward is not among the tied top of the product.

Against q alone. The q-alone top is the set of gated-in rollouts with the highest q_i. q takes only the values 0, 0.5, and 1, so ties are common. A group with one rollout at the highest q has a unique q-alone top. A group with several has a tied q-alone top, and length, trigram ratio, and the detector term decide among them.

| Item | Set one | Set two |
| --- | --- | --- |
| Top of the full reward differs from the top without the detector term | 2 of 20 | 3 of 20 |
| Groups with q = 0 for all four, where the rank alone picks the top | 2 | 3 |
| Groups where the detector term picks or changes the top, total | 4 of 20 | 6 of 20 |
| Groups where any pair of rollouts changes order | 8 of 20 | 14 of 20 |
| q alone has a unique top | 3 | 3 |
| In those, the full-reward top is that rollout | 3 of 3 | 3 of 3 |
| q alone has a tied top | 17 | 17 |
| In those, the full-reward top is inside the tied set | 17 of 17 | 17 of 17 |
| Full-reward top is the highest-z rollout, all groups | 7 of 20 | 11 of 20 |

The detector term rarely overrides q. When q has a unique top, the full reward always keeps it. Where several rollouts share the top q, which is most groups, the full-reward top is always one of them, and the rank, the length score, and the trigram ratio choose which. The rank changes the top in 10% of groups on the base rollouts and 15% on the checkpoint rollouts, and it changes some ordering in 40% and 70%. The highest-z rollout is the full-reward top in 35% of groups on the base rollouts and 55% on the checkpoint rollouts, against 25% if the top were random among four. The detector term has more effect on the checkpoint rollouts than on the base rollouts. This dry run does not show why.
&nbsp;
### Length among rollouts with the same q

For every pair of gated-in rollouts in a group with the same q and different word counts, the longer one's reward is compared with the shorter one's.

| Pairs | Set one | Set two |
| --- | --- | --- |
| Same q, different length | 52 | 74 |
| Longer ranks below shorter | 27 (52%) | 45 (61%) |
| Shorter is over 203 words | 14 | 27 |
| In those, longer ranks below shorter | 11 (79%) | 21 (78%) |

Above the 203-word target the longer answer ranks below the shorter one in about four of five pairs. Below the target the rank term, which does not depend on length, can put the shorter answer first.
&nbsp;
### Rewards per group

In each row the entries are the four rollouts in order. `v1` is `validation-00001`. Top is "same" when the full-reward top is also a top of the product without the detector term, "changed" when it is not, and "all q=0, detector picks" when every q is 0.

### Set one, base rollouts

| Group | Rewards | q | Rank r | Words | Spread | Real | Top |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.7602 0.0916 0.4847 0.9064 | 1 0 0.5 1 | 1.00 0.67 0.33 0.00 | 252 95 168 184 | 0.8148 | yes | same |
| v2 | 0.7043 0.8217 0.9619 0.2542 | 1 1 1 0 | 0.33 0.67 0.00 1.00 | 134 139 209 180 | 0.7077 | yes | same |
| v3 | 0.3648 0.0897 0.0656 0.9703 | 0.5 0 0 1 | 1.00 0.67 0.33 0.00 | 342 91 301 201 | 0.9047 | yes | same |
| v4 | 0.0000 0.6447 0.7747 0.1218 | 0 1 1 0 | 0.00 0.33 1.00 0.67 | 201 121 123 174 | 0.7747 | yes | same |
| v5 | 0.5657 0.0000 0.7685 0.2601 | 0.5 0 1 0 | 0.33 0.00 0.67 1.00 | 207 213 138 182 | 0.7685 | yes | same |
| v6 | 0.0000 0.9669 0.7790 0.0995 | 0 1 1 0 | 0.00 1.00 0.33 0.67 | 159 153 251 103 | 0.9669 | yes | same |
| v7 | 0.6501 0.2660 0.7138 0.7383 | 1 0.5 1 1 | 0.67 0.00 0.33 1.00 | 116 108 280 300 | 0.4723 | yes | changed |
| v8 | 0.5983 0.3092 0.6961 0.1611 | 1 0.5 1 0 | 0.00 0.50 0.50 1.00 | 284 397 293 109 | 0.5350 | yes | same |
| v9 | 0.0457 0.3152 1.1408 0.0000 | 0 0.5 1 0 | 0.33 1.00 0.67 0.00 | 403 430 197 464 | 1.1408 | yes | same |
| v10 | 0.0000 0.1655 0.0635 0.2231 | 0 0 0 0 | 0.00 0.67 0.33 1.00 | 220 172 137 153 | 0.2231 | yes | all q=0, detector picks |
| v11 | 0.7446 0.8776 0.7272 0.7424 | 1 1 1 1 | 0.67 1.00 0.00 0.33 | 130 147 260 137 | 0.1503 | yes | changed |
| v12 | 0.6476 0.8038 0.5633 0.9705 | 1 1 1 1 | 0.00 0.67 1.00 0.33 | 301 138 92 226 | 0.4072 | yes | same |
| v13 | 0.0478 0.3309 0.6255 0.5498 | 0 0.5 1 1 | 0.33 1.00 0.00 0.67 | 97 90 131 93 | 0.5777 | yes | same |
| v14 | 0.6077 0.4819 0.3138 1.2038 | 1 0.5 0.5 1 | 0.00 0.33 0.67 1.00 | 299 238 91 192 | 0.8900 | yes | same |
| v15 | 0.5911 0.4714 0.4739 0.0311 | 1 0.5 1 0 | 0.67 0.17 1.00 0.17 | 100 178 74 301 | 0.5601 | yes | same |
| v16 | 0.2932 0.2845 0.4380 0.9501 | 1 1 0.5 1 | 0.00 1.00 0.67 0.33 | 484 379 241 233 | 0.6655 | yes | same |
| v17 | 0.1626 0.0409 0.0000 0.1084 | 0 0 0 0 | 1.00 0.33 0.00 0.67 | 110 83 62 110 | 0.1626 | yes | all q=0, detector picks |
| v18 | 0.5111 0.2039 0.5036 0.7383 | 1 0 1 1 | 0.00 1.00 0.33 0.67 | 358 138 416 311 | 0.5344 | yes | same |
| v19 | 0.4250 0.1676 0.5245 0.7771 | 1 0 1 1 | 0.33 1.00 0.00 0.67 | 496 351 383 301 | 0.6096 | yes | same |
| v20 | 0.0000 0.7098 0.4846 0.9225 | 0 1 0.5 1 | 0.00 0.67 1.00 0.33 | 362 313 127 225 | 0.9225 | yes | same |

### Set two, seed-42 slope run step-80 checkpoint

| Group | Rewards | q | Rank r | Words | Spread | Real | Top |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.9308 0.9514 0.8279 0.9724 | 1 1 1 1 | 0.00 1.00 0.67 0.33 | 214 267 286 217 | 0.1445 | yes | changed |
| v2 | 0.0951 0.5206 0.0000 0.2926 | 0 0.5 0 0 | 0.33 0.67 0.00 1.00 | 193 153 143 200 | 0.5206 | yes | same |
| v3 | 0.0000 0.1139 0.2577 0.0621 | 0 0 0 0 | 0.00 0.67 1.00 0.33 | 419 344 196 294 | 0.2577 | yes | all q=0, detector picks |
| v4 | 0.8521 0.9161 1.2190 1.0078 | 1 1 1 1 | 0.00 0.67 1.00 0.33 | 177 161 206 188 | 0.3669 | yes | same |
| v5 | 0.7882 0.1894 1.0275 0.2837 | 1 0 1 0 | 0.00 0.67 0.33 1.00 | 160 206 209 198 | 0.8381 | yes | same |
| v6 | 0.9438 0.5912 0.7724 1.0684 | 1 0.5 0.5 1 | 0.00 0.67 1.00 0.33 | 211 221 198 209 | 0.4772 | yes | same |
| v7 | 1.0284 0.6994 0.6827 1.1783 | 1 1 1 1 | 0.67 0.00 0.33 1.00 | 178 146 323 192 | 0.4956 | yes | same |
| v8 | 1.0893 0.9491 0.4122 0.9350 | 1 1 1 1 | 0.33 1.00 0.00 0.67 | 205 252 454 239 | 0.6770 | yes | same |
| v9 | 0.3931 0.9317 0.6209 0.0000 | 0.5 1 0.5 0 | 0.33 1.00 0.67 0.00 | 141 245 214 478 | 0.9317 | yes | same |
| v10 | 0.1941 0.2498 0.0561 0.0000 | 0 0 0 0 | 0.67 1.00 0.33 0.00 | 197 169 116 512 | 0.2498 | yes | all q=0, detector picks |
| v11 | 0.5413 0.8398 0.8817 1.0174 | 0.5 1 1 1 | 0.67 0.33 0.00 1.00 | 163 159 185 249 | 0.4760 | yes | changed |
| v12 | 0.0000 0.1951 0.2734 0.0795 | 0 0 0 0 | 0.00 0.67 1.00 0.33 | 435 198 191 229 | 0.2734 | yes | all q=0, detector picks |
| v13 | 0.9014 0.1439 0.2638 1.0457 | 1 0 0 1 | 0.00 0.67 1.00 0.33 | 185 276 216 195 | 0.9018 | yes | same |
| v14 | 1.2296 0.8925 0.8614 0.2493 | 1 1 1 0.5 | 1.00 0.67 0.33 0.00 | 192 155 163 375 | 0.9802 | yes | same |
| v15 | 0.5727 0.8021 0.1963 0.8241 | 1 1 0 1 | 0.67 0.00 1.00 0.33 | 413 249 289 164 | 0.6278 | yes | changed |
| v16 | 0.7071 0.8420 0.7183 0.5959 | 1 1 0.5 0.5 | 0.33 0.00 1.00 0.67 | 299 237 222 212 | 0.2462 | yes | same |
| v17 | 0.6931 0.9003 0.0000 0.3819 | 0.5 1 0 0.5 | 1.00 0.33 0.00 0.67 | 226 246 338 368 | 0.9003 | yes | same |
| v18 | 0.4945 1.1719 0.7208 0.8354 | 0.5 1 1 1 | 0.67 1.00 0.33 0.00 | 277 191 284 243 | 0.6774 | yes | same |
| v19 | 1.0607 0.3652 0.8723 0.9451 | 1 1 1 1 | 1.00 0.00 0.33 0.67 | 215 492 165 243 | 0.6954 | yes | same |
| v20 | 0.9664 0.8353 0.0380 0.2082 | 1 1 0 0.5 | 1.00 0.67 0.33 0.00 | 269 268 480 449 | 0.9284 | yes | same |

&nbsp;
## Run

The dry run passed on both sets, so the run started. Seed 42, T = 1 in the logs but unused by this reward, `Qwen/Qwen3-0.6B-Base`, group size 4, `max_new_tokens` 700, temperature 0.8, `top_p` 0.9, AdamW at 1e-5, gradient clip 1.0, no KL, bfloat16, gradient checkpointing, the 200-word prompt. Zero-advantage steps skip backward and `optimizer.step()`. The script is `/tmp/grpo_len200_rank_train.py`, the tmux session was `grpo-rank-t1`, and the log is `results/grpo-length-200-rank-t1/run.log`. The checks and stop rules are those of the slope run. Weights are in `results/grpo-length-200-rank-t1-weights` on the lab machine and are not committed.

The run stopped at step 40. The recorded reason is an Opus win rate under 40%. That rate is 13 of 48 agreed pairs, which is 0.271. Step 20 was 24 of 45, or 0.533, so the two-check streak rule did not apply. No other rule fired. The gate-fire rise streak peaked at 1, 1 step had no gradient, and the median length over 10 steps peaked at 269.5 words. Reward-judge parse failures are 0 over 306 calls, and Opus parse failures are 0. `gpt-5.5` was not called. Over the 40 steps the 160 rollouts were judged against their matched base as rollout 65, base 75, split 10, tie 3, and 7 were gated out.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 24/45 | 21/37 | 0.140022 | 0.144758 | 0.149680 | 0.128783 |
| 40 | Opus | 13/48 | 9/33 | 0.350149 | 0.080392 | 0.400764 | 0.104174 |

Win rates on all agreed pairs are 53.3% at step 20 and 27.1% at step 40. Within 20% length they are 56.8% and 27.3%. Opus ties were 5 and 2. The 95% Wilson interval at step 40 is 0.17 to 0.41.
&nbsp;
## Detector scores rose, and quality fell

At step 20 the run looks like the slope run, with quality near even and the detectors level with the base. By step 40 the trained answers score much higher on both detectors than their matched bases. The Qwen3 sigmoid is 0.350 against 0.080, and DistilBERT is 0.401 against 0.104. The share of trained evaluation answers with z above 0 is 17 of 50, against 4 of 50 for the base. Opus prefers the base in 35 of 48 agreed pairs.

The text got worse in a specific way. The mean trigram ratio of the training rollouts fell from 0.97 to 0.99 in the first 20 steps to 0.915 in steps 21 to 30 and 0.891 in steps 31 to 40, and the mean word count rose to 284. In the step-40 evaluation answers the trigram ratio is 0.925 for trained and 0.969 for base, 10 of 50 trained answers are under 0.9 against 4 of 50 base answers, and 4 of 50 trained answers are 700-token runaways. The two most repetitive trained answers run to 641 and 628 words. Gate fires went from 0 in the first 20 steps to 2 in steps 21 to 30 and 5 in steps 31 to 40.

The rank term gives the detector a full 0.3 in every group, where the sigmoid gave about 0.01 in most. That reopened the route the T = 1 sweep took, to raise the detector score and lose on quality. q did not hold it back: the mean q of judged rollouts fell from 0.54 in steps 1 to 10 to 0.35 in steps 31 to 40, so the policy was moving toward detector-higher answers that lose to the base, and q was not enough to stop it. The rank cannot say how far apart the z values are, so even a group where all four answers are poor on the detector still gets a 0 to 1 spread, and the policy is pushed on every group.

Compared at step 40 with the earlier runs:

| Run | Opus agreed pairs | Trained sigmoid | Base sigmoid | Trained DistilBERT | Base DistilBERT |
| --- | ---: | ---: | ---: | ---: | ---: |
| T = 1 sweep | 18/46 | 0.363 | 0.173 | 0.627 | 0.233 |
| Slope, seed 42 | 22/39 | 0.121 | 0.111 | 0.140 | 0.145 |
| Slope, seed 2 | 27/45 | 0.062 | 0.147 | 0.121 | 0.217 |
| Rank, seed 42 | 13/48 | 0.350 | 0.080 | 0.401 | 0.104 |

The rank run's detector gap at step 40 is 0.27 in sigmoid against 0.19 for the T = 1 sweep, and 0.30 in DistilBERT against 0.39. It loses more clearly on quality, 27% against 39%. The slope runs hold quality and do not move the detectors. No reward so far does both. The T = 1 sweep also ran without the zero-advantage skip, so that comparison carries the confound noted in `results/grpo-slope-reward.md`.
&nbsp;
## Training trajectory

| steps | mean q of judged rollouts | mean words | mean sigmoid | mean DistilBERT | mean trigram ratio | mean reward | gate fires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 to 10 | 0.537 | 163 | 0.196 | 0.295 | 0.969 | 0.375 | 0 |
| 11 to 20 | 0.500 | 125 | 0.014 | 0.134 | 0.992 | 0.373 | 0 |
| 21 to 30 | 0.450 | 192 | 0.248 | 0.334 | 0.915 | 0.287 | 2 |
| 31 to 40 | 0.350 | 284 | 0.230 | 0.440 | 0.891 | 0.316 | 5 |
&nbsp;
## Cost

| Part | Calls | USD |
| --- | ---: | ---: |
| Reward judge, `claude-sonnet-5-5` | 306 | 0.638 |
| Opus checks, `claude-opus-5-5` | 200 | 2.053 |
| `gpt-5.5` | 0 | 0.000 |
| Total | 506 | 2.692 |

The run took about 1 hour 35 minutes of wall time for 40 steps. Seed 2 was not started for this reward.
