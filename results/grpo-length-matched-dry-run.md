&nbsp;
# Length-matched dry run

No training ran. The T sweep is unstarted. This note records the length target, the new gate, the length-matched opponent, the reward dry run, and the evaluation plan. The 100-prompt judge check has not been run.
&nbsp;
## Length target

Whitespace words are `text.split()`. The saved 80 answers are indexes 0 through 3 of `validation-00001` through `validation-00020` in `results/grpo-base-detector-scores/base-answers.jsonl`. Their median is 187.0. The two central counts are 186 and 188.

The new 160 answers are eight draws per prompt from frozen `Qwen/Qwen3-0.6B-Base`, saved in `results/grpo-length-matched/base-pool.jsonl`. `max_new_tokens` is 700, temperature is 0.8, `top_p` is 0.9, and the seed is 42, set after the model load and before the first prompt. The eight draws for a prompt are one batch. Their median is 227.5. The two central counts are 227 and 228.

The saved median and the new median differ by 40.5 words. The nearest integers are 187 and 228. The saved-80 median is the target when those medians are within 10 words or the nearest integers match. Neither holds, so the target is the median of all 240 answers. Both central counts of that combined list are 203, and the median is 203.0. The length target is 203.

The prompt text is unchanged. It still asks for approximately 250 words. The value 203 is the `length_score` target.
&nbsp;
## Hard gate

`hard_i` is 0 when the answer has fewer than 100 words, more than 300 words, fewer than 30 words, or at least 700 generated tokens. Counts from 100 through 300 stay in. A stub under 30 words is already under 100, so that clause adds no extra zero on these samples. Inside the band,

length_score = min(words, 203) / max(words, 203).

On the 160 pool samples, 63 are hard zeros. That is 63 of 160, a gate-fire rate of 0.39375. Twelve are under 100 words, 51 are over 300 words, and none are under 30 words. One sample also hit 700 generated tokens: `validation-00019` pool index 0, 535 words, which is already in the 51 over 300. The 63 count is unique samples. Ninety-seven pool samples are inside the gate. Every one of the 20 prompts has at least one gated pool sample.

On the 80 saved rollouts, 23 are hard zeros under this gate: 17 over 300 words and 6 under 100 words. None reached 700 generated tokens. Those answers were drawn under the older 1616 cap.
&nbsp;
## Opponent

The dry-run rollouts are the saved indexes 0, 1, 2, and 3. They are not members of the pool of eight. For each rollout, the opponent is the gated-in pool sample whose word count is closest in absolute difference. A tie on that difference takes the lowest `pool_index`. If a prompt had no pool sample inside the gate, its rollouts would have no base opponent. That did not happen on these 20 prompts.

A rollout outside the gate has reward 0. Its closest pool sample is listed below and was not sent to the judge.
&nbsp;
## Reward

The reward judge is `claude-sonnet-5-5` with the JSON schema that passed. The schema is an object whose only property is `winner`, a string enum of A, B, and tie. `winner` is required and `additionalProperties` is false. The call sends no tools, no `tool_choice`, and no temperature. `max_tokens` is 1024. The prompt is "Which of these two answers to the same question is better? If they are equally good, choose tie."

A win is both orders naming the same answer. If either order says tie, the orders name different answers, or either call still fails to parse after one retry, the pair is a tie. Identical texts would not be a win, and the pair would still count in N. None of these pairs were identical.

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N

`wins_i` counts the other gated-in rollouts and that rollout's matched base. N is the same count. If N is 0 or `hard_i` is 0, the reward is 0. z_i is the saved raw human-class logit for that rollout, index 0, before the temperature division. Those 80 logits were reused and the rollouts were not rescored. Recomputing the product from the saved logits, the word counts, and the recorded wins matches the rewards below with a maximum absolute error of 0.
&nbsp;
## Dry-run spreads

A real spread is max reward minus min reward above 1e-4, with the max reward above 0. The four rewards are indexes 0, 1, 2, and 3. Eighteen of the 20 groups have a real spread. The two without a spread are `validation-00003` and `validation-00019`.

| prompt | index 0 | index 1 | index 2 | index 3 | spread | real |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| validation-00001 | 0.369170 | 0.099131 | 0.000000 | 0.247914 | 0.369170 | yes |
| validation-00002 | 0.000000 | 0.157584 | 0.000000 | 0.045675 | 0.157584 | yes |
| validation-00003 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00004 | 0.159317 | 0.064971 | 0.000000 | 0.240910 | 0.240910 | yes |
| validation-00005 | 0.136717 | 0.024962 | 0.064599 | 0.000000 | 0.136717 | yes |
| validation-00006 | 0.234966 | 0.034545 | 0.000000 | 0.101169 | 0.234966 | yes |
| validation-00007 | 0.000000 | 0.027034 | 0.092185 | 0.059011 | 0.092185 | yes |
| validation-00008 | 0.000000 | 0.090409 | 0.000000 | 0.565050 | 0.565050 | yes |
| validation-00009 | 0.046561 | 0.000000 | 0.143991 | 0.000000 | 0.143991 | yes |
| validation-00010 | 0.144619 | 0.000000 | 0.095455 | 0.048641 | 0.144619 | yes |
| validation-00011 | 0.190286 | 0.000000 | 0.000000 | 0.000000 | 0.190286 | yes |
| validation-00012 | 0.000000 | 0.067117 | 0.093459 | 0.065699 | 0.093459 | yes |
| validation-00013 | 0.070867 | 0.118013 | 0.000000 | 0.000000 | 0.118013 | yes |
| validation-00014 | 0.067445 | 0.000000 | 0.000000 | 0.054829 | 0.067445 | yes |
| validation-00015 | 0.000000 | 0.000000 | 0.000000 | 0.139019 | 0.139019 | yes |
| validation-00016 | 0.275906 | 0.170305 | 0.000000 | 0.000000 | 0.275906 | yes |
| validation-00017 | 0.000000 | 0.130657 | 0.000000 | 0.051589 | 0.130657 | yes |
| validation-00018 | 0.077621 | 0.000000 | 0.000000 | 0.000000 | 0.077621 | yes |
| validation-00019 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00020 | 0.000000 | 0.000000 | 0.000000 | 0.057440 | 0.057440 | yes |

`validation-00003`: index 0 has 196 words, is inside the gate, and won 0 of 1 against pool index 5 (140 words); index 1 has 354 words and is over 300; index 2 has 301 words and is over 300; index 3 has 79 words and is under 100. `validation-00019`: index 0 has 70 words and is under 100; index 1 has 331 words and is over 300; index 2 has 274 words, is inside the gate, and won 0 of 1 against pool index 5 (269 words); index 3 has 380 words and is over 300.

| prompt | words 0–3 | wins/N | opponent pool:words |
| --- | --- | --- | --- |
| validation-00001 | 199, 226, 153, 220 | 3/4, 2/4, 0/4, 2/4 | 0:186, 0:186, 2:130, 0:186 |
| validation-00002 | 120, 192, 408, 172 | 0/3, 3/3, 0/0, 1/3 | 3:130, 1:217, 6:248, 0:157 |
| validation-00003 | 196, 354, 301, 79 | 0/1, 0/0, 0/0, 0/0 | 5:140, 3:285, 3:285, 4:100 |
| validation-00004 | 103, 149, 334, 124 | 3/3, 1/3, 0/0, 2/3 | 6:112, 3:125, 4:297, 3:125 |
| validation-00005 | 194, 269, 235, 261 | 4/4, 1/4, 2/4, 0/4 | 2:182, 4:270, 0:239, 7:265 |
| validation-00006 | 162, 104, 125, 105 | 2/4, 1/4, 0/4, 4/4 | 0:172, 4:136, 4:136, 4:136 |
| validation-00007 | 238, 130, 203, 243 | 0/4, 1/4, 2/4, 2/4 | 3:219, 4:121, 0:201, 3:219 |
| validation-00008 | 145, 197, 489, 211 | 0/3, 1/3, 0/0, 3/3 | 2:158, 1:201, 7:228, 1:201 |
| validation-00009 | 149, 384, 183, 144 | 1/3, 0/0, 3/3, 0/3 | 5:266, 5:266, 5:266, 5:266 |
| validation-00010 | 184, 154, 170, 169 | 4/4, 0/4, 3/4, 1/4 | 7:180, 1:162, 1:162, 1:162 |
| validation-00011 | 125, 126, 114, 122 | 4/4, 0/4, 0/4, 0/4 | 0:141, 0:141, 5:107, 5:107 |
| validation-00012 | 352, 152, 200, 289 | 0/0, 1/3, 2/3, 2/3 | 6:278, 2:137, 1:203, 6:278 |
| validation-00013 | 141, 199, 293, 92 | 2/3, 2/3, 0/3, 0/0 | 7:138, 4:196, 4:196, 0:109 |
| validation-00014 | 106, 352, 366, 126 | 1/2, 0/0, 0/0, 1/2 | 1:109, 5:293, 5:293, 0:114 |
| validation-00015 | 68, 90, 111, 186 | 0/0, 0/0, 0/2, 2/2 | 1:139, 1:139, 1:139, 2:162 |
| validation-00016 | 188, 118, 527, 324 | 1/2, 2/2, 0/0, 0/0 | 5:190, 5:190, 2:288, 2:288 |
| validation-00017 | 154, 184, 72, 102 | 0/3, 3/3, 0/0, 2/3 | 7:149, 7:149, 3:124, 3:124 |
| validation-00018 | 145, 256, 379, 349 | 1/2, 0/2, 0/0, 0/0 | 2:150, 0:253, 5:284, 5:284 |
| validation-00019 | 70, 331, 274, 380 | 0/0, 0/0, 0/1, 0/0 | 2:231, 1:294, 5:269, 1:294 |
| validation-00020 | 211, 378, 423, 241 | 0/2, 0/0, 0/0, 1/2 | 0:246, 2:278, 2:278, 0:246 |

The judge made 119 comparisons, 238 calls, both orders. Parse failures were 0, and no call was retried. Every returned model string was `claude-sonnet-5-5`. Cache reads and cache writes were 0. Input tokens were 233,498 and output tokens were 3,740.

Sonnet 5.5 standard prices are $2 per 1M input tokens and $10 per 1M output tokens. The dry-run cost is 233498 / 1,000,000 * 2 + 3740 / 1,000,000 * 10 = $0.504396.
&nbsp;
## Cost of a training step and a check

A training step of 4 rollouts, with all four inside the gate and each one holding a matched base, is 6 rollout pairs plus 4 base pairs, which is 10 comparisons and 20 calls. The measured mean is 981.084 input tokens and 15.714 output tokens per call. That step is 20 / 238 of the dry-run cost, $0.042386.

The dry run itself gated out some rollouts, so its 20 groups cost $0.025220 each. The $0.042386 figure is the full-gate step.

A check is 100 validation prompts, `validation-00001` through `validation-00100`. It has not been run. The estimate below uses the measured step-40 prices on 50 prompts and doubles them.

The step-40 Opus check of 50 prompts cost $1.182668 across 101 calls. Opus 5.5 prices are $4 per 1M input tokens and $20 per 1M output tokens. The token split for that check was not saved, so the 100-prompt estimate doubles the measured dollar cost: $2.365336.

`gpt-5.5` on those same 50 texts used 63,127 input tokens and 9,688 output tokens, model `gpt-5.5-2026-04-23`. The page prices are $5 per 1M input tokens, $0.50 per 1M cached input tokens, and $30 per 1M output tokens. Cached input was not stored, so this estimate prices every input token at $5. Reasoning tokens sit inside the output count and are priced at the output rate. The prompts are far below 272K tokens, so the long-context multiplier does not apply. Fifty prompts cost 63127 / 1,000,000 * 5 + 9688 / 1,000,000 * 30 = $0.606275. One hundred prompts cost $1.212550.

One 100-prompt check, Opus plus `gpt-5.5`, is $2.365336 + $1.212550 = $3.577886.
&nbsp;
## Evaluation plan

The check stays unrun. When it runs, each trained answer is paired with the gated-in frozen base sample whose word count is closest, by the same rule as this dry run. The judges are `claude-opus-5-5` and `gpt-5.5`, both orders, agreement only. Opus writes A or B to "Which of these two answers to the same question is better? Reply with A or B." `gpt-5.5` uses the Responses API, reasoning effort low, `store` false, and the same letter rule. A pair counts when both orders name the same answer. The win rate is trained wins divided by trained wins plus base wins. Ties and unparsed pairs stay out of the denominator. If the agreed count is 0, that rate is not treated as under 40%.

The check reports that win rate on all agreed pairs, and again on the pairs whose trained word count is between 0.8 and 1.2 times the matched base word count. It also reports the mean of sigmoid(z / 4) and the mean DistilBERT P(human) for the trained answers and for the length-matched base answers. DistilBERT P(human) is 1 minus P(AI) from `score_many`. These judges stay out of the reward.
&nbsp;
## T sweep

The T sweep is unstarted. It waits for Eli's OK after this dry run. The plan is T in {4, 2, 1}, two seeds each, 80 steps, with the stop rules from the JSON run: Opus win rate under 40%, or under 45% at two checks in a row, or the gate-fire count rising for 20 consecutive steps. Every check reports the Opus win rate, the `gpt-5.5` win rate, and both detector scores.

The pool file, the Sonnet call cache, and the summary json stay on the lab machine. They are not part of the note commit. Weights were not trained and were not committed.
