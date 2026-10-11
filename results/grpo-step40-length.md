&nbsp;
# Step 40 length

No new training. The texts are the step-40 generations in `results/grpo-sonnet-json/eval-step-0040.jsonl`, fifty validation answers. The base opponent for `validation-00001` through `validation-00020` is the saved index-0 answer. The base opponent for `validation-00021` through `validation-00050` is the frozen draw held for that prompt. Every pair already had a saved Opus outcome and a saved `gpt-5.5` outcome, so this note uses those verdicts.
&nbsp;
## Detector scores

The soft score is sigmoid(z / 4). z is the raw human-class logit from the frozen `qwen3-variable` classifier, index 0, before the temperature division. DistilBERT P(human) is 1 minus the P(AI) from `score_many`. Both detectors ran on CPU. The step-40 texts were scored again. The largest absolute difference from the saved per-text scores is 0, so the means match the logged figures.

| set | sigmoid(z / 4) | DistilBERT P(human) |
| --- | ---: | ---: |
| 50 base answers | 0.2655 | 0.2160 |
| step 40 | 0.1971 | 0.0924 |

Step 40 is less human-like than the base answers on both detectors. The base soft score is 0.2655 and the step-40 soft score is 0.1971. The base DistilBERT score is 0.2160 and the step-40 DistilBERT score is 0.0924.
&nbsp;
## Win rates

A win rate is trained wins divided by the pairs whose two orders named the same answer. A tie stays out of the denominator. Word counts are whitespace words.

On all agreed pairs, Opus is 26 of 39, which is 66.7%, with 13 base wins and 11 ties. `gpt-5.5` is 29 of 41, which is 70.7%, with 12 base wins and 9 ties. The returned model string on those calls was `gpt-5.5-2026-04-23`.

On the 18 pairs where the trained answer has at most 1.2 times as many words as the base answer, Opus is 5 of 11, which is 45.5%, with 6 base wins and 7 ties. `gpt-5.5` is 7 of 14, which is 50%, with 7 base wins and 4 ties.

The length bin is trained words minus base words. The 50 differences were sorted and cut after the 17th and 34th values, into groups of 17, 17, and 16. The first cut is 27 and the second cut is 157.

| bin | edge | observed range | n | Opus | gpt-5.5 |
| --- | --- | --- | ---: | --- | --- |
| 1 | difference ≤ 27 | −320 to 27 | 17 | 5/10 | 7/13 |
| 2 | 27 < difference ≤ 157 | 33 to 157 | 17 | 11/14 | 12/15 |
| 3 | difference > 157 | 165 to 380 | 16 | 10/15 | 10/13 |

Opus ties in the three bins are 7, 3, and 1. `gpt-5.5` ties are 4, 2, and 3. No pair was unparsed. The wins add up to the all-pairs counts: Opus 5 + 11 + 10 = 26, and `gpt-5.5` 7 + 12 + 10 = 29.
&nbsp;
## Length gate

Across steps 1 through 40, each step has 4 rollouts. A rollout is hard-zeroed for length when it is under 150 words, over 375 words, or a 700-token hit. A rollout that matches more than one of those is counted once.

The mean count per step is 2.975. That is 119 length hard-zeros over 40 steps. The fraction of all rollout-steps is 119 of 160, which is 74.4%.

Of those rollouts, 113 are under 150 words, 6 are over 375, and 1 hit 700 tokens. The 700-token hit is also over 375, a step-40 rollout of 580 words, so the three reason counts sum to 120 and the union is 119. 113 of 160 rollouts, 70.6%, are under 150 words. 26 of 40 steps have a mean word count under 150. Most of the run is under 150 words.

The proposal for the next run is a floor of 100 words. The current config keeps its floor of 150.
&nbsp;
## Next experiment

This experiment has not started. It waits for Eli's OK.

T is 4, 2, and 1. Each value gets 2 seeds. That is 6 runs of 80 steps. The stop rules are the same as the JSON run: stop if the Opus win rate is under 40%, or under 45% at two checks in a row, or the gate-fire count rises for 20 consecutive steps. Every check reports the Opus win rate, the `gpt-5.5` win rate, and both detector scores.
