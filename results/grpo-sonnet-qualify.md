&nbsp;
# Sonnet qualification

No training. The 20 pairs in `results/grpo-pilot-pairs.md` were scored as they stand. `claude-sonnet-5-5` passed the held-out set and the reference miss bars. The dry run then spread on 7 of the 20 saved groups. That is not most of the groups, so the 40-step run did not start.
&nbsp;
## Step 20 hand check

The trained side of each pair is the step-20 answer in the pair sheet. The other side is the saved base answer. The side map stays out of this repository.

`gpt-4.1-mini-2025-04-14` scored both orders with the A/B probability method. The signal is the trained answer over the base answer. A signal above 7 is a win for the trained answer. A signal below −7 is a win for the base answer. Absolute signal at most 7 is a tie. The response `model` field was `gpt-4.1-mini-2025-04-14` on all 40 calls.

`claude-opus-5-5` wrote A or B in both orders. The same side on both orders is a win. A disagreement is a tie. The response `model` field was `claude-opus-5-5` on all 40 calls. Those calls used `max_tokens` 1024.

The two judges disagree on pairs 1, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, and 18. On those pairs the mini judge preferred the trained side on 1, 4, 7, 8, 10, 11, 12, 14, 15, and 16. It did not on 13 and 18, where its signal was a tie. Opus preferred the base answer on every disagreement except pair 7, where its two orders disagreed.

| pair | mini | signal | opus | mini trained | opus trained |
| ---: | --- | ---: | --- | --- | --- |
| 1 | trained | 16.375 | base | yes | no |
| 2 | trained | 18.452 | trained | yes | yes |
| 3 | trained | 21.544 | trained | yes | yes |
| 4 | trained | 8.850 | base | yes | no |
| 5 | trained | 20.250 | trained | yes | yes |
| 6 | trained | 23.128 | trained | yes | yes |
| 7 | trained | 15.500 | tie | yes | no |
| 8 | trained | 10.089 | base | yes | no |
| 9 | base | -20.923 | base | no | no |
| 10 | trained | 12.983 | base | yes | no |
| 11 | trained | 13.625 | base | yes | no |
| 12 | trained | 19.310 | base | yes | no |
| 13 | tie | 1.125 | base | no | no |
| 14 | trained | 19.125 | base | yes | no |
| 15 | trained | 13.917 | base | yes | no |
| 16 | trained | 26.441 | base | yes | no |
| 17 | trained | 21.691 | trained | yes | yes |
| 18 | tie | 4.750 | base | no | no |
| 19 | trained | 27.316 | trained | yes | yes |
| 20 | trained | 10.956 | trained | yes | yes |

They agree on pairs 2, 3, 5, 6, 9, 17, 19, and 20. Pair 9 is the agreement on the base answer. The other seven agreements are on the trained answer.
&nbsp;
### Truncated trained answers

Each trained answer was cut to its first 200 whitespace words. All 20 are longer than 200 words, so every cut removed text. The frozen `qwen3-variable` score is sigmoid(z / 4), with z the raw human-class logit before the saved temperature. DistilBERT is P(human). GPU 3 was already in use, so both passes ran on CPU.

The full-answer mean of sigmoid(z / 4) is 0.3804. The 200-word mean is 0.6264. The human score stays up. It rises. DistilBERT P(human) goes from 0.4996 to 0.7819.
&nbsp;
## Qualify claude-sonnet-5-5

The prompt was exactly "Which of these two answers to the same question is better? Reply with A or B." `max_tokens` was 32. Temperature 0 was sent on the first call. The API rejected it with the message that temperature is deprecated for this model. That call was retried once without temperature, and every later call omitted temperature.

The response `model` field was `claude-sonnet-5-5` on every Sonnet call in this note, including the dry run.

A letter is read from the first nonempty line after markdown bold is stripped. The line has to open with A or B. A reply that explains first is not a letter. The same side on both orders is a win. A disagreement, or a missing letter, is not a win. On the reference pairs, a miss is both orders agreeing on the wrong side. A missing letter is not a miss.
&nbsp;
### Held-out 30

The pairs are `results/grpo-judge-methods-heldout-gold.json`. The corrected text is the right side. All 30 pairs were correct: both orders agreed on the corrected side. A disagreement would have failed that pair. There were none.
&nbsp;
### Reference pairs

The letters are the PDF slots in `results/grpo-reference-labels.md`. The scored set is the 20 pairs in the three bands below. Pairs 13, 14, 16, 18, and 21 were not sent.

High and medium-high pairs 3, 6, 7, 8, 10, 11, 15, 17, 19, 20, 22, and 25 have no misses. Pair 25's swapped order had no letter, so that pair is not counted as a miss and not counted as an agreement. Medium pairs 1, 5, 9, 12, 23, and 24 have no misses. Pairs 1, 5, 9, and 23 had a missing letter. Low pairs 2 and 4 are reported and do not enter the miss bar. Pair 4's two letters are A on the display order and A on the swapped order.

| pair | band | reference | display | swapped | result |
| ---: | --- | --- | --- | --- | --- |
| 1 | medium | B | none | A | no letter |
| 2 | low | B | none | A | no letter |
| 3 | high | A | A | B | correct |
| 4 | low | A | A | A | disagree |
| 5 | medium | A | none | A | no letter |
| 6 | high | A | A | B | correct |
| 7 | high | B | B | A | correct |
| 8 | high | B | B | A | correct |
| 9 | medium | B | none | none | no letter |
| 10 | high | A | A | B | correct |
| 11 | high | A | A | B | correct |
| 12 | medium | B | B | A | correct |
| 15 | high | A | A | B | correct |
| 17 | high | B | B | A | correct |
| 19 | high | B | B | A | correct |
| 20 | high | B | B | A | correct |
| 22 | high | B | B | A | correct |
| 23 | medium | A | none | none | no letter |
| 24 | medium | A | A | B | correct |
| 25 | high | B | B | none | no letter |
&nbsp;
### Identical answers

The 10 texts are the identical-answer rows in `results/grpo-judge-bias.md`, the saved base answers at those prompt ids and indexes. The same text was placed in both slots, in both orders. A tie is the two orders returning different letters. Zero of the 10 returned the same letter both times, because neither order opened with A or B. In the body, each of the 10 says the two texts match and then names A.
&nbsp;
### Cost

One comparison is one pair in both orders. Across the 60 qualification comparisons, the mean is $0.003607. Those 120 calls used 89,054 input tokens and 3,834 output tokens, $0.216448 in total.

The price table is the Anthropic pricing table at https://platform.claude.com/docs/en/about-claude/pricing, the Claude Sonnet 5.5 standard row: $2 per million input tokens and $10 per million output tokens. Cache reads and cache writes were zero on every call.
&nbsp;
## Dry run

The 20 saved groups are `results/grpo-base-detector-scores/base-answers.jsonl`, `validation-00001` through `validation-00020`, indexes 0 through 3. Index 0 is the unpaid base opponent. Indexes 1, 2, and 3 are rewarded. z is the saved raw human logit. The trigram ratio is distinct lowercased whitespace-word trigrams over positions. The length score is min(words, 250) / max(words, 250).

hard_i is 0 under 150 words, over 375 words, under 30 words, or if the text hit max_new_tokens. Words from 150 through 375 stay in. Every saved `hit_1616` flag is false, and these answers were not generated under a 700-token cap, so the word rules are the whole gate. A stub under 30 words is already under 150.

A win is a Sonnet agreement against another gated-in answer, including the base answer when that answer is gated in. N is that opponent count. If N is 0, the reward is 0. Identical texts are not a win for either side.

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N

There were 39 comparisons, 78 calls. Fourteen agreed on a winner. Five returned two letters that named different sides. Twenty had no opening letter, so they awarded no win.

A real spread is the largest of the three rewards minus the smallest, above 1e-4, with the largest above 0. Seven of the 20 groups have one. That is not most of the groups. The 40-step run did not start.

| prompt | index 1 | index 2 | index 3 | spread |
| --- | ---: | ---: | ---: | ---: |
| validation-00001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00002 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00003 | 0.050890 | 0.000000 | 0.000000 | 0.050890 |
| validation-00004 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00005 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00006 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00007 | 0.000000 | 0.074854 | 0.137322 | 0.137322 |
| validation-00008 | 0.000000 | 0.000000 | 0.495696 | 0.495696 |
| validation-00009 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00010 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00011 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00012 | 0.000000 | 0.037944 | 0.000000 | 0.037944 |
| validation-00013 | 0.143740 | 0.000000 | 0.000000 | 0.143740 |
| validation-00014 | 0.112294 | 0.000000 | 0.000000 | 0.112294 |
| validation-00015 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00016 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00017 | 0.106094 | 0.000000 | 0.000000 | 0.106094 |
| validation-00018 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00019 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| validation-00020 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The seven groups are `validation-00003`, `validation-00007`, `validation-00008`, `validation-00012`, `validation-00013`, `validation-00014`, and `validation-00017`.

The dry run cost $0.181064 for 78 calls, 78,052 input tokens and 2,496 output tokens, at the same Sonnet 5.5 rates. That is $0.004643 per comparison. A training step with 4 rollouts plus the base answer, all gated in, is 10 comparisons, $0.04643 at this dry-run mean. The qualification mean of $0.003607 would put the same 10 comparisons at $0.03607.
