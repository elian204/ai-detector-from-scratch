&nbsp;
# Training config, not run

No GRPO was started. This config is for review. The judge is not accepted for a run until this config is reviewed.
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
&nbsp;
## Gate

The gate is the frozen rule gate in `results/grpo-junk-gate.md`. The thresholds are unchanged. gate_i is 0 when the gate fires, including a 1616 cap hit, and 1 otherwise.
&nbsp;
## Judge

The judge is `gpt-4.1-mini-2025-04-14`, both orders, temperature 0, the A/B probability method in `results/grpo-openai-judge.md`. signal = (m1 + m2) / 2. The accepted tie threshold is 7. A win is signal greater than 7. Absolute signal at most 7 gives the win to neither side.
&nbsp;
## Comparison group

The comparison group is the gated-in rollouts plus one frozen base-model answer to the same prompt. The base answer is sampled once from the initial policy and then held fixed. It is not trained. It is an opponent only.

wins_i is the number of other gated-in completions j with signal(i over j) greater than 7. The base answer is one of those completions when it passes the gate. N is the number of other gated-in completions, and it includes the base answer.
&nbsp;
## Reward

reward_i = gate_i * detector_i * wins_i / N

If the gate fires, gate_i is 0 and the reward is 0. If N is 0, the reward is 0 and the quotient is not taken. There is no KL term and no trigram term.
&nbsp;
## Reference result

On the 22 pairs in `results/grpo-reference-labels.md` that are not both-bad, this judge has 15 matches, 6 ties, and 1 miss against the Claude Opus reference labels. The both-bad pairs, 9, 13, and 25, are excluded from that count. The one miss is pair 4. The judge is not accepted for a run until this config is reviewed.
&nbsp;
## API cost, an estimate

Rates are the standard text prices recorded in `results/grpo-openai-judge.md` from the OpenAI model page on 8 October 2026: `gpt-4.1-mini` at $0.40 per 1M input tokens and $1.60 per 1M output tokens. This estimate does not use the cached-input rate.

Each answer is counted as 1616 tokens, the response budget used as a stand-in. That is not a measured API token count. The question and the fixed instruction are counted as 40 tokens. One forward is two answers, so 3,272 input tokens, plus 1 output token, which is what the 50 reference calls returned.

If all four rollouts pass the gate, the comparison group has five answers. That is 10 pairs, and both orders make 20 forwards. Input is 65,440 tokens and output is 20 tokens. The estimate is 65,440 / 1,000,000 * 0.40 + 20 / 1,000,000 * 1.60 = $0.026 per step.

&nbsp;
## Monitoring

Every 10 steps, log the gate-fire rate, the DistilBERT score, the win rate versus the base answer by the evaluation judge, and 5 sample texts. The evaluation judge is Claude via API. It writes a verdict in both orders, and only agreeing verdicts count. It is not in the reward. The key is not set up, and Claude was not called.

Stop if the gate-fire rate rises for 20 steps, or if the win rate versus the base answer falls below 50%.

No run was started.
