&nbsp;
# Preflight

No GRPO and no training. The 40-step pilot was not started.
&nbsp;
## Confidence bands

The signals are the ones already in `results/grpo-reference-labels.md`. The 25 pairs were not rescored. The judge there is `gpt-4.1-mini-2025-04-14`, both orders, signal = (m1 + m2) / 2.

The exclusion set is pairs 13, 14, 16, 18, and 21. Pairs 9, 13, and 25 are not this exclusion set. The excluded pairs are outside the bands below.

High and Med-High: 3, 6, 7, 8, 10, 11, 15, 17, 19, 20, 22, 25. Medium: 1, 5, 9, 12, 23, 24. Low: 2, 4.

At threshold 14, a match is signal greater than 14, a tie is absolute signal at most 14, and a miss is signal less than −14. At threshold 7, a match is signal greater than 7, a tie is absolute signal at most 7, and a miss is signal less than −7.

| pair | band | signal | at 14 | at 7 |
| ---: | --- | ---: | --- | --- |
| 3 | High and Med-High | 8.000 | tie | match |
| 6 | High and Med-High | 16.748 | match | match |
| 7 | High and Med-High | 23.691 | match | match |
| 8 | High and Med-High | 27.631 | match | match |
| 10 | High and Med-High | 14.568 | match | match |
| 11 | High and Med-High | 20.481 | match | match |
| 15 | High and Med-High | 23.941 | match | match |
| 17 | High and Med-High | 22.941 | match | match |
| 19 | High and Med-High | 27.631 | match | match |
| 20 | High and Med-High | 15.125 | match | match |
| 22 | High and Med-High | 26.503 | match | match |
| 25 | High and Med-High | 7.869 | tie | match |
| 1 | Medium | 20.875 | match | match |
| 5 | Medium | 1.500 | tie | tie |
| 9 | Medium | 15.375 | match | match |
| 12 | Medium | 22.941 | match | match |
| 23 | Medium | 6.875 | tie | tie |
| 24 | Medium | 18.691 | match | match |
| 2 | Low | 17.066 | match | match |
| 4 | Low | -23.941 | miss | miss |

High and Med-High, 12 pairs: at 14, 10 matches, 2 ties, 0 misses. The ties are pair 3, signal 8.000, and pair 25, signal 7.869. At 7, 12 matches, 0 ties, 0 misses.

Medium, 6 pairs: at 14, 4 matches, 2 ties, 0 misses. The ties are pair 5, signal 1.500, and pair 23, signal 6.875. At 7, 4 matches, 2 ties, 0 misses. Pair 23 stays a tie because 6.875 is not greater than 7.

Low, 2 pairs: at 14 and at 7, 1 match and 1 miss. The match is pair 2, signal 17.066. The miss is pair 4, signal −23.941.

The acceptance bar is no misses on High and Med-High, at most one miss on Medium, and Low misses reported only. That bar is met at 14 and at 7. High and Med-High have no misses at either threshold, and Medium has no misses at either threshold.
&nbsp;
## Gate on the reference side

The gate is the frozen loop, padding, and stub rules in `results/grpo-junk-gate.md`. A 416 `hit_token_cap` does not count. A cap hit means the completion reached 1616 tokens. None of the 25 reference sides reached 1616 tokens, so the cap rule fires on none of them.

The side below is the PDF slot named by the reference label. The gate flags that side on 12 of the 25 pairs. Padding and stub fire on none of them.

| pair | side | flagged | rules |
| ---: | --- | --- | --- |
| 1 | B | no |  |
| 2 | B | no |  |
| 3 | A | no |  |
| 4 | A | no |  |
| 5 | A | no |  |
| 6 | A | yes | loop, 6-gram 4 |
| 7 | B | yes | loop, trigram 0.5614 |
| 8 | B | no |  |
| 9 | B | yes | loop, 6-gram 4; loop, trigram 0.6765 |
| 10 | A | yes | loop, trigram 0.6776 |
| 11 | A | no |  |
| 12 | B | yes | loop, sentence 3 |
| 13 | B | yes | loop, 6-gram 5; loop, trigram 0.6973 |
| 14 | B | no |  |
| 15 | A | yes | loop, 6-gram 5; loop, trigram 0.5838 |
| 16 | A | no |  |
| 17 | B | yes | loop, 6-gram 6; loop, trigram 0.5985 |
| 18 | A | yes | loop, 6-gram 4; loop, trigram 0.6456 |
| 19 | B | no |  |
| 20 | B | yes | loop, 6-gram 7 |
| 21 | B | no |  |
| 22 | B | no |  |
| 23 | A | yes | loop, 6-gram 5; loop, trigram 0.5365 |
| 24 | A | no |  |
| 25 | B | yes | loop, 6-gram 4 |

After dropping pairs 13, 14, 16, 18, and 21, the gate flags 10 of the remaining 20 reference sides. The dropped sides that were flagged are pair 13 side B and pair 18 side A.

Five reference sides stopped at 416 tokens and fire no loop, padding, or stub rule, so they are not flagged: pair 1 side B, pair 2 side B, pair 8 side B, pair 16 side A, and pair 19 side B.
&nbsp;
## Reward dry run

The 80 answers are `results/grpo-base-detector-scores/base-answers.jsonl`, 20 prompts with sample indexes 0, 1, 2, and 3. In each group the lowest index is the frozen base opponent and is not rewarded. Indexes 1, 2, and 3 are the rollouts. No text was compared with itself. No two answers in a group were the same text.

The gate is the same 1616 rule as the section above. It fires on none of the 80 answers. None of them reached 1616 tokens. The base opponent passes the gate in every group, so it is included in wins and in N. N is 3 for every rollout.

The detector is the saved `qwen3-variable` P(human) from the frozen-base scoring run in `results/grpo-base-detector-scores.md`, times the symmetric length score at target 250: min(word count, 250) / max(word count, 250). Word count is whitespace words. P(human) of 0 makes the detector 0.

The judge is `gpt-4.1-mini-2025-04-14` only, both orders, temperature 0, the A/B probability method in `results/grpo-openai-judge.md`. A win is signal greater than 14. The calls were the pairs of gated-in answers that include a rewarded rollout with detector above 0. That is 41 pairs and 82 forwards. The response `model` field was `gpt-4.1-mini-2025-04-14` on every call. 65 calls had both letters in the top list, 17 used the chosen-only fallback, and 12 of those fallback calls were clamped.

reward = gate × detector × wins / N. The gate is 1 on all 80 answers. A rollout with detector 0 has reward 0. A rollout with wins 0 has reward 0. A tie is absolute signal at most 14 and is not a win. A loss is signal less than −14 and is not a tie.

Every rewarded rollout has reward 0 in 15 of the 20 groups. That fraction is 15/20.

The gate accounts for none of those 15 groups. Nine groups are detector 0 on all three rewarded rollouts: `validation-00002`, `validation-00003`, `validation-00005`, `validation-00007`, `validation-00009`, `validation-00010`, `validation-00017`, `validation-00019`, and `validation-00020`.

No group is all ties on all three rewarded rollouts. The other six groups mix detector 0 with a rollout that has detector above 0 and no wins.

| prompt | what is zero |
| --- | --- |
| validation-00001 | index 1 detector 0; index 2 loss; index 3 all ties |
| validation-00006 | index 1 loss; index 2 all ties; index 3 detector 0 |
| validation-00011 | index 1 detector 0; index 2 detector 0; index 3 all ties |
| validation-00012 | index 1 loss; index 2 detector 0; index 3 detector 0 |
| validation-00013 | index 1 detector 0; index 2 all ties; index 3 detector 0 |
| validation-00015 | index 1 detector 0; index 2 loss; index 3 detector 0 |

Five groups have a rollout with reward above 0. The signal against the base opponent is included. A value at most 14 is not a win against the base.

| prompt | index | wins | N | reward | signal vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation-00004 | 1 | 1 | 3 | 0.0008 | 13.875 |
| validation-00008 | 3 | 1 | 3 | 0.2697 | -2.125 |
| validation-00014 | 2 | 2 | 3 | 0.4532 | 17.854 |
| validation-00016 | 1 | 1 | 3 | 0.0012 | 23.191 |
| validation-00018 | 2 | 1 | 3 | 0.0077 | 14.688 |
