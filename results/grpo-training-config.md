&nbsp;
# Training config

The reward below is the reward used for the pilot in `results/grpo-pilot-40.md`.
&nbsp;
## Policy

The policy is `Qwen/Qwen3-0.6B-Base`. The trainer is the stage-18 GRPO script, `scripts/18_reinforcement-learning/05_train_grpo_human_writing.py`. The group size is 4. That script's `--num-rollouts` default is 8, and this config uses 4.
&nbsp;
## Token budget

The response token budget is 1616. A cap hit is that 1616 limit. The 416 cap is not used.
&nbsp;
## Detector

The detector is the frozen `qwen3-variable` classifier, the verifier that trainer already calls. P(AI) is the probability it returns. The length target is 250 whitespace words. The length score is the existing symmetric score, `min(word count, 250) / max(word count, 250)`.

detector_i = (1 − P_AI_i) * length_score_i

The reward uses T = 4 in sigmoid(z_i / 4), where z_i is the raw human-class logit. T is 4, not 1.
&nbsp;
## Gate

The gate is the frozen rule gate in `results/grpo-junk-gate.md`. The thresholds are unchanged. gate_i is 0 when the gate fires, including a 1616 cap hit, and 1 otherwise.
&nbsp;
## Judge

The judge is `gpt-4.1-mini-2025-04-14`, both orders, temperature 0, the A/B probability method in `results/grpo-openai-judge.md`. signal = (m1 + m2) / 2. The accepted tie threshold is 7. A win is signal greater than 7. Absolute signal at most 7 gives the win to neither side.
&nbsp;
## Comparison group

The comparison group is the gated-in rollouts plus one frozen base-model answer to the same prompt. The base answer is sampled once from the initial policy and then held fixed. It is not trained. It is an opponent only.

wins_i is the number of other completions j that are not hard zeros and have signal(i over j) greater than 7. The frozen base answer is included when it is not a hard zero. N is the number of other completions that are not hard zeros, and it includes the base answer.
&nbsp;
## Reward

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N

hard_i is 0 for a stub under 30 words or a completion that reached 1616 tokens, and 1 otherwise. z_i is the raw human-class logit. T is 4. A win is signal greater than 7. The frozen base answer is included in wins and in N. If hard_i is 0, the reward is 0. If N is 0, the reward is 0 and the quotient is not taken.
&nbsp;
## Reference result

On the 22 pairs in `results/grpo-reference-labels.md` that are not both-bad, this judge has 15 matches, 6 ties, and 1 miss against the Claude Opus reference labels. The both-bad pairs, 9, 13, and 25, are excluded from that count. The one miss is pair 4.
&nbsp;
## API cost, an estimate

Rates are the standard text prices recorded in `results/grpo-openai-judge.md` from the OpenAI model page on 8 October 2026: `gpt-4.1-mini` at $0.40 per 1M input tokens and $1.60 per 1M output tokens. This estimate does not use the cached-input rate.

Each answer is counted as 1616 tokens, the response budget used as a stand-in. That is not a measured API token count. The question and the fixed instruction are counted as 40 tokens. One forward is two answers, so 3,272 input tokens, plus 1 output token, which is what the 50 reference calls returned.

If all four rollouts pass the gate, the comparison group has five answers. That is 10 pairs, and both orders make 20 forwards. Input is 65,440 tokens and output is 20 tokens. The estimate is 65,440 / 1,000,000 * 0.40 + 20 / 1,000,000 * 1.60 = $0.026 per step.

&nbsp;
## Monitoring

Every 10 steps, log the mean trigram ratio, the stub count, the 1616-runaway count, the DistilBERT score, the evaluation-judge win rate versus the saved base answer, and 5 sample texts. Also log the tie count and the cost of that check. The mean trigram ratio is tracked every step.

The evaluation judge is `claude-opus-5-5`. It writes a verdict in both orders. A pair counts only when both orders name the same answer. A pair whose orders disagree is a tie. Each check is 20 trained answers against the saved base answers to the same prompts. This judge is not the reward judge.

Stop if the mean trigram ratio falls for 20 consecutive steps, or if the evaluation win rate versus the base answer drops below 50%.

The pilot is 40 steps. The reward judge is `gpt-4.1-mini-2025-04-14`. The response token budget is 1616. The pilot ran on GPU 3 and stopped at step 20. The evaluation win rate at that check was 7 of 18 counted pairs, which is below 50%. The trigram-ratio stop did not fire. The record is `results/grpo-pilot-40.md`.

&nbsp;
## Rerun stop rule

This rule was written before the Sonnet check. It did not run. The check is the next sections.

Each check uses 50 Claude comparisons. Stop if the win rate is under 40%, or under 45% at two checks in a row.
&nbsp;
## Sonnet reward

The reward tested in `results/grpo-sonnet-qualify.md` keeps the pilot product and changes the gate and the win. hard_i is 0 under 150 words, over 375 words, under 30 words, or on a max_new_tokens hit. Words from 150 through 375 stay in. A win is both orders of `claude-sonnet-5-5` naming the same answer. The prompt is "Which of these two answers to the same question is better? Reply with A or B." `max_tokens` is 32. Temperature 0 is deprecated on this model, so the qualification calls omitted temperature. The response model string was `claude-sonnet-5-5` on every call.

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N

The frozen base answer stays in wins and in N when it is gated in. If N is 0, the reward is 0. A run on this reward would use a 700-token response budget. The dry run kept the saved answers and applied the word rules, because those answers were not capped at 700.

Held-out was 30 of 30. The high and medium-high reference pairs had no misses, and the medium pairs had no misses. The dry run spread on 7 of 20 groups. The 40-step run did not start.
&nbsp;
## Stop rule

This rule is not running. The dry run spread on 7 of 20 groups, so the run that would use it did not start.

Each check would use 50 validation prompts. The evaluation judge would be `claude-opus-5-5`, in both orders, and it would stay out of the reward. Stop if the Opus win rate versus the saved base answer is under 40%, or under 45% at two checks in a row, or the gate-fire count rises for 20 consecutive steps.

Each check would also log the Sonnet win rate versus the base answer on those same texts, the mean of sigmoid(z / 4), the mean DistilBERT score, the mean word count, the mean trigram ratio, the gate-fire counts by reason, and 5 sample texts.
&nbsp;
## Judge format

The requested reward-judge format is a forced tool call on `claude-sonnet-5-5`, not a free-text letter. One tool, `select_winner`. One field, `winner`, with values A, B, and tie. The prompt is "Which of these two answers to the same question is better? If they are equally good, choose tie." Temperature is omitted. `max_tokens` is 256. A reply that is not that tool call is a parse failure. Both orders are scored. A win is both orders naming the same answer. If either order says tie, the orders name different answers, or either call fails to parse, the pair is a tie.

`tool_choice` type `tool` was sent once. The API returned HTTP 400: tool_choice type "tool" and "any" are not supported for this model. The Anthropic tool-use page says Claude Sonnet 5.5 returns that 400 for type `tool` and type `any`. The identical-answer check, the held-out 30, the reference bands, and the dry run were not sent. The 40-step run did not start. The record is `results/grpo-sonnet-tool.md`.

&nbsp;
## JSON schema judge

The reward judge that passed is `claude-sonnet-5-5` with a JSON schema. The request sets `output_config.format.type` to `json_schema`. The schema is an object with one property, `winner`, a string enum of A, B, and tie. `winner` is required. `additionalProperties` is false. The call sends no tools, no `tool_choice`, and no temperature. `max_tokens` is 1024. The prompt is "Which of these two answers to the same question is better? If they are equally good, choose tie."

One probe returned HTTP 200, `{"winner":"tie"}`, and model `claude-sonnet-5-5`. The free-text last-line fallback was not used. The identical-answer check was a tie on all 10 pairs. Held-out was 30 of 30. High and medium-high pairs had no misses. Medium pairs had no misses. Low pair 4 returned B then A, and low pair 2 returned A then B. Those two low pairs are misses, and low misses do not fail the bar. Parse failures were 0. The dry run had a real spread on 15 of 20 groups and 0 parse failures. The record is `results/grpo-sonnet-json.md`.

The 40-step run uses this JSON verdict as the only reward judge. The response budget is 700 tokens. `hard_i` is 0 under 150 words, over 375 words, under 30 words, or on a 700-token hit. The evaluation judge is `claude-opus-5-5`, written A or B, both orders, and it stays out of the reward. This run stops if the Opus win rate is under 40%, or under 45% at two checks in a row, or the gate-fire count rises for 20 consecutive steps. Each check uses 50 validation prompts. The forced-tool format in the previous section was rejected and is not used. The run finished all 40 steps. The final Opus win rate versus the base was 26 of 39. `gpt-5.5` on the same texts was 29 of 41. Weights stayed on the lab machine.

&nbsp;
## Length-matched setup

The length target is 203 words. `hard_i` is 0 under 100 words, over 300 words, under 30 words, or on a 700-token hit. Words from 100 through 300 stay in. Inside that band, `length_score` is min(words, 203) / max(words, 203). The reward judge is the JSON schema on `claude-sonnet-5-5`. The call sends no tools and no temperature.

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / T) * length_score_i * wins_i / N

The opponent for a rollout is the gated-in sample among eight frozen base answers whose word count is closest. A tie takes the lowest pool index.

An earlier dry run, commit `a149083`, used a prompt that asked for approximately 250 words. The current training prompt asks for approximately 200 words. The 200-word dry run is `results/grpo-length-matched-dry-run.md`. It passed: a real spread on 18 of 20 groups, 0 parse failures, and a gate-fire rate of 47 of 160 pool answers, which is 0.29375. The length target stayed 203.

The one-seed sweep followed that pass. T is 4, then 2, then 1. Seed 42. Each run is 80 steps from scratch. T is the divisor in `sigmoid(z / T)`. At steps 20, 40, and 60 the check is 50 validation prompts with `claude-opus-5-5` only. At step 80 the check is 100 validation prompts with Opus and `gpt-5.5`. Both orders, agreement only, against the length-matched base. Each check reports the win rate on all agreed pairs and on pairs within 20 percent of the matched length, plus the mean of `sigmoid(z / T)` and the mean DistilBERT score for the trained answers and for the matched base answers. A run stops if Opus is under 40%, or under 45% at two checks in a row, or the gate-fire count rises for 20 consecutive steps. A second seed is not part of this sweep.

The sweep has finished. T = 4 stopped at step 40, T = 2 finished 80 steps, and T = 1 stopped at step 40. Seed 2 did not start. The record is `results/grpo-length-200-t-sweep.md`.

The gates in the sections above, including 150 through 375 and the 1616 cap, are the gates those earlier runs used.

The quality-gate reward, q_i times the product with no win fraction, is written in `results/grpo-quality-gate-reward.md`. Its dry run at T = 1 had a real spread in 15 of 20 groups. The T = 1 run stopped at step 40 with the Opus win rate at 19 of 47 and then 18 of 43. The policy drifted to answers over 300 words, and 28 of 40 steps had no judged rollout and no gradient.

The slope reward replaces the quality-gate reward for the next run. It is written in `results/grpo-slope-reward.md`.

q_i is 1 if `claude-sonnet-5-5` picks the rollout over the matched base in both orders, 0 if it picks the base in both orders, and 0.5 otherwise, which covers a tie, a split, and a parse failure. Parse failures are counted. s_i = sigmoid(z_i / T) with T = 1, where z_i is the raw human-class logit before the saved temperature.

reward_i = hard_i * length_score_i * trigram_ratio_i ** 2 * (q_i + 0.3 * s_i)

hard_i is 0 only for a stub under 30 words or a 700-token runaway. There is no 100 to 300 word window. length_score_i = min(w, 203) / max(w, 203). The prompt asks for approximately 200 words. The matched base is the gated-in pool sample closest in word count, and only the stub and runaway rules gate the pool. The judge is the JSON verdict as before: no tools, no temperature, both orders. Group size 4, `max_new_tokens` 700, temperature 0.8, `top_p` 0.9, seed 42, AdamW at 1e-5, no KL.

A step whose advantages are all zero skips the optimizer entirely. The earlier sweep and quality-gate scripts called `optimizer.step()` on every step, so a zero-advantage step still moved the weights through AdamW momentum and weight decay. The slope run follows the repo `--skip-zero-advantage-updates` behavior: no backward, no clip, no `optimizer.step()`. The check is in `results/grpo-slope-reward.md`.

The dry run of this reward on two sets of 20 groups, base rollouts and rollouts of the T = 1 step-40 checkpoint, had a real spread in 20 of 20 groups on both sets, 0 parse failures, and a gradient in 20 of 20 groups on both sets.

The slope run finished 80 steps with no stop rule, no zero-gradient step, and 0 reward-judge parse failures. Opus won 49 of 83 agreed pairs at step 80 and `gpt-5.5` won 51 of 80. The detector means did not rise above the length-matched base. The cost was $7.95. The record is in `results/grpo-slope-reward.md`.
