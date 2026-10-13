&nbsp;
# Sonnet JSON schema

Step 2 passed and step 3 passed. The reward judge is `claude-sonnet-5-5` with a JSON schema. No tools were sent. No `tool_choice` was sent. Temperature was not sent. The 40-step run uses this format.
&nbsp;
## Probe

The prompt was exactly "Which of these two answers to the same question is better? If they are equally good, choose tie." The probe question was "Name a primary color." The two answers were "Red is a primary color." and "Blue is a primary color."

The request set `output_config.format.type` to `json_schema`. The schema is an object. The only property is `winner`, a string enum of `A`, `B`, and `tie`. `winner` is required. `additionalProperties` is false. `max_tokens` was 1024.

The API returned HTTP 200. The text block was `{"winner":"tie"}`. The stop reason was `end_turn`. The returned model string was `claude-sonnet-5-5`. The call was not a parse failure. The free-text fallback was not used. That fallback would have required a last line of `Verdict: A`, `Verdict: B`, or `Verdict: tie`.

The probe used 284 input tokens and 12 output tokens. Cache read and cache write were 0. The cost was $0.000688.
&nbsp;
## Identical answers

The same 10 base answers were placed in both slots, in both orders. A pair is a tie when either order says tie, the orders name different answers, or either call still fails to parse after one retry. All 10 pairs were ties. All 20 calls returned `winner` = `tie`. None picked a side. Parse failures were 0. Every returned model string was `claude-sonnet-5-5`.

| prompt | index | words | order 1 | order 2 | pair |
| --- | ---: | ---: | --- | --- | --- |
| validation-00015 | 0 | 68 | tie | tie | tie |
| validation-00006 | 3 | 105 | tie | tie | tie |
| validation-00011 | 0 | 125 | tie | tie | tie |
| validation-00004 | 1 | 149 | tie | tie | tie |
| validation-00002 | 3 | 172 | tie | tie | tie |
| validation-00008 | 1 | 197 | tie | tie | tie |
| validation-00005 | 2 | 235 | tie | tie | tie |
| validation-00012 | 3 | 289 | tie | tie | tie |
| validation-00003 | 1 | 354 | tie | tie | tie |
| validation-00016 | 2 | 527 | tie | tie | tie |
&nbsp;
## Held-out 30

All 30 held-out pairs had both orders on the corrected side. Display order returned `A` and the swapped order returned `B` on every pair. Parse failures were 0. Every returned model string was `claude-sonnet-5-5`.
&nbsp;
## Reference bands

Pairs 13, 14, 16, 18, and 21 were not sent. A miss is both orders agreeing on the wrong side. A disagreement is not a miss. High and medium-high pairs 3, 6, 7, 8, 10, 11, 15, 17, 19, 20, 22, and 25 had no misses. Medium pairs 1, 5, 9, 12, 23, and 24 had no misses. Pair 9 returned `B` then `B`, which is a disagreement, not a miss.

Low pair 2 has reference `B`. The two winners were `A` then `B`. Both orders named the wrong side, so pair 2 is a miss. Low misses do not fail the bar.

Low pair 4 has reference `A`. The two winners were `B` then `A`. Both orders named the wrong side, so pair 4 is a miss. Low misses do not fail the bar.

| pair | band | reference | display | swapped | result |
| --- | --- | --- | --- | --- | --- |
| 1 | medium | B | B | A | correct |
| 2 | low | B | A | B | miss |
| 3 | high | A | A | B | correct |
| 4 | low | A | B | A | miss |
| 5 | medium | A | A | B | correct |
| 6 | high | A | A | B | correct |
| 7 | high | B | B | A | correct |
| 8 | high | B | B | A | correct |
| 9 | medium | B | B | B | disagree |
| 10 | high | A | A | B | correct |
| 11 | high | A | A | B | correct |
| 12 | medium | B | B | A | correct |
| 15 | high | A | A | B | correct |
| 17 | high | B | B | A | correct |
| 19 | high | B | B | A | correct |
| 20 | high | B | B | A | correct |
| 22 | high | B | B | A | correct |
| 23 | medium | A | A | B | correct |
| 24 | medium | A | A | B | correct |
| 25 | high | B | B | A | correct |

Step 2 passes. Held-out is 30 of 30. The high and medium miss bars hold. Parse failures are 0. All 10 identical pairs are ties.
&nbsp;
## Dry run

The dry run used the saved answers for `validation-00001` through `validation-00020`. Index 0 is the unpaid base. Rewards are indexes 1, 2, and 3. `hard_i` is 0 under 150 words, over 375 words, under 30 words, or on a max-token hit. Words from 150 through 375 stay in. The product is `hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N`. A win is both orders naming the same answer. If either order says tie, the orders name different answers, or either call still fails to parse after one retry, the pair is a tie. Parse failures were 0.

A real spread is max minus min above 1e-4, with max above 0. Fifteen of the 20 groups have a real spread. The five without a spread are `validation-00004`, `validation-00006`, `validation-00009`, `validation-00011`, and `validation-00015`. Each of those three rewards is 0.

| prompt | index 1 | index 2 | index 3 | spread | real |
| --- | ---: | ---: | ---: | ---: | --- |
| validation-00001 | 0.066512 | 0.000000 | 0.157623 | 0.157623 | yes |
| validation-00002 | 0.127958 | 0.000000 | 0.000000 | 0.127958 | yes |
| validation-00003 | 0.101780 | 0.000000 | 0.000000 | 0.101780 | yes |
| validation-00004 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00005 | 0.040988 | 0.046864 | 0.000000 | 0.046864 | yes |
| validation-00006 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00007 | 0.000000 | 0.074854 | 0.068661 | 0.074854 | yes |
| validation-00008 | 0.000000 | 0.000000 | 0.495696 | 0.495696 | yes |
| validation-00009 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00010 | 0.000000 | 0.068897 | 0.052662 | 0.068897 | yes |
| validation-00011 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00012 | 0.000000 | 0.075888 | 0.040455 | 0.075888 | yes |
| validation-00013 | 0.143740 | 0.000000 | 0.000000 | 0.143740 | yes |
| validation-00014 | 0.112294 | 0.000000 | 0.000000 | 0.112294 | yes |
| validation-00015 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00016 | 0.000000 | 0.000000 | 0.127164 | 0.127164 | yes |
| validation-00017 | 0.106094 | 0.000000 | 0.000000 | 0.106094 | yes |
| validation-00018 | 0.000000 | 0.000000 | 0.100232 | 0.100232 | yes |
| validation-00019 | 0.000000 | 0.132780 | 0.000000 | 0.132780 | yes |
| validation-00020 | 0.000000 | 0.000000 | 0.131474 | 0.131474 | yes |

Step 3 passes. Parse failures are 0, and 15 of 20 groups have a spread.
&nbsp;
## Cost

Rates are the Claude Sonnet 5.5 standard prices on the Anthropic pricing page, https://platform.claude.com/docs/en/about-claude/pricing. Input is $2 per million tokens. Output is $10 per million tokens. Cache read is $0.10, the 5-minute cache write is $2.50, and the 1-hour cache write is $4. These are not batch prices. Cache read and cache write were 0 on the probe, on step 2, and on step 3. The billed totals match input tokens times $2 plus output tokens times $10.

| piece | calls | input tokens | output tokens | USD |
| --- | ---: | ---: | ---: | ---: |
| probe | 1 | 284 | 12 | 0.000688 |
| step 2 | 120 | 115094 | 1603 | 0.246218 |
| step 3 | 78 | 94978 | 1933 | 0.209286 |

Step 3 is 39 both-order comparisons, so one comparison is $0.005366. A training step whose five answers all stay in the gate has 10 comparisons, about $0.0537.
&nbsp;
## Run

The 40-step rerun uses this JSON verdict as the only reward judge. The policy is `Qwen/Qwen3-0.6B-Base`. The seed is 42. The group size is 4. `max_new_tokens` is 700. Sampling temperature is 0.8 and `top_p` is 0.9. The optimizer is AdamW at 1e-5 with gradient clipping at 1.0, bfloat16, gradient checkpointing, and no KL term. The loss is the mean of minus the normalized advantage times the sequence log probability. Advantages are taken over the four rollout rewards. The frozen base answer is an opponent in the wins and in N, and it is not in the advantage group.

`reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N`. `z_i` is the raw human-class logit from the frozen `qwen3-variable` classifier, index 0, before the temperature division. The length score is `min(words, 250) / max(words, 250)`. If N is 0, the reward is 0.

The evaluation judge is `claude-opus-5-5`. It writes A or B. The prompt is "Which of these two answers to the same question is better? Reply with A or B." Both orders are scored. A pair counts only when both orders name the same answer. Opus is not in the reward. Every 10 steps, 50 validation prompts are scored. Sonnet scores those same texts with the JSON schema. The log records the Opus win rate versus the base, the Sonnet win rate versus the base, the mean of sigmoid(z / 4), the mean DistilBERT P(human), the mean word count, the mean trigram ratio, the gate-fire counts by reason, and 5 sample texts.

Saved index-0 bases are the opponents for `validation-00001` through `validation-00020`. The opponents for `validation-00021` through `validation-00050` are drawn once from the frozen initial policy at 700 tokens and then held fixed. The training seed is set again before step 1, so that draw does not move the training stream.

Stop if the Opus win rate is under 40%, or under 45% at two checks in a row, or the gate-fire count on the four rollouts rises for 20 consecutive steps. A check with no agreed pairs does not count as under 40%. After the run, the final 50 texts are scored once with `gpt-5.5` if that id is accepted, otherwise `gpt-5.4`, with reasoning, both orders, agreement only.

GPU 3 already held this account's breakfast synthesis job. This run capped its own CUDA allocator at half of the card and left that job in place. Detectors ran on CPU. Weights are in `results/grpo-sonnet-json-weights` and are not committed. The log is `results/grpo-sonnet-json/run.log`. The tmux session was `grpo-sonnet-json-train`.
&nbsp;
## Result

The run finished all 40 steps. No stop rule fired. The longest streak of a rising gate-fire count was 3 steps. Reward-judge parse failures were 0 across 112 calls, all returned as `claude-sonnet-5-5`. Cache read and cache write on those calls were 0. The reward-judge cost was $0.320332.

A win rate is trained wins divided by the pairs whose two orders named the same answer. Ties and unparsed pairs stay out of the denominator. Opus is `claude-opus-5-5`. Sonnet on the same texts uses the JSON schema. The five sample texts at each check are `validation-00001` through `validation-00005`. The full generations are in `results/grpo-sonnet-json/eval-step-0010.jsonl` through `eval-step-0040.jsonl`, and those files are not committed.

| step | Opus | Sonnet | sigmoid(z/4) | DistilBERT | words | trigram | under 150 | over 375 | 700-token hit | hard zero |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 20/42 | 19/40 | 0.2206 | 0.0986 | 129.0 | 0.9907 | 40 | 1 | 0 | 41 |
| 20 | 23/47 | 21/44 | 0.2218 | 0.0743 | 135.6 | 0.9921 | 36 | 0 | 0 | 36 |
| 30 | 23/46 | 14/37 | 0.2645 | 0.1151 | 116.8 | 0.9885 | 46 | 1 | 0 | 47 |
| 40 | 26/39 | 26/39 | 0.1971 | 0.0924 | 281.8 | 0.9759 | 6 | 11 | 1 | 17 |

At step 10, Opus had 22 base wins and 8 ties. At step 20, 24 base wins, 2 ties, and 1 unparsed pair. At step 30, 23 base wins and 4 ties. At step 40, 13 base wins and 11 ties. Sonnet ties were 10, 6, 13, and 11. Opus had one remaining parse failure, at step 20, after the retry. Sonnet parse failures at the checks were 0. No check was under 40%, and no two checks in a row were under 45%.

The final 50 texts are the step-40 generations. `gpt-5.5` accepted the id and returned `gpt-5.5-2026-04-23`. The call used the Responses API with reasoning effort low. Reasoning tokens were 8,990 of 9,688 output tokens. Input tokens were 63,127. Parse failures were 0. On those same texts the Opus win rate is 26 of 39, which is 66.7%. The gpt-5.5 win rate is 29 of 41, which is 70.7%, with 12 base wins and 9 ties.

Twenty shuffled unlabeled pairs are in `results/grpo-pilot-sonnet-pairs.md`. The shuffle seed is 20261008. The map from A and B to the trained side is only in the store internal directory, and it is not committed.
&nbsp;
## Confound: zero-advantage steps were not skipped

The training loop of this run called `optimizer.step()` on every step and had no skip. Its saved step log has 24 of 40 steps with all four rewards equal to 0 and a loss of exactly 0, so on those steps the optimizer ran on a zero gradient and moved the weights through AdamW momentum and weight decay. Its result is confounded on this point in the same way. The steps are 1, 3, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, and 35. The script of this run has one `optimizer.step()` call, inside the step loop, and no skip flag. The Opus and `gpt-5.5` checks above are read with this in mind. The later runs that skip these steps are in `results/grpo-slope-reward.md`.
