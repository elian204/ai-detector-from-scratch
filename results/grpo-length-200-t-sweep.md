&nbsp;
# Length-matched T sweep

Seed 42. One run for T = 4, then one for T = 2, then one for T = 1. Each run loaded `Qwen/Qwen3-0.6B-Base` from scratch. Group size 4. `max_new_tokens` 700. Temperature 0.8. `top_p` 0.9. The prompt asks for approximately 200 words. The `length_score` target is 203. T is the divisor in `sigmoid(z / T)`. The reward judge is `claude-sonnet-5-5` with the JSON schema, no tools and no temperature. The evaluation judge is `claude-opus-5-5`. It is not in the reward. Seed 2 did not start.

The dry run that cleared the bars is `results/grpo-length-matched-dry-run.md`. The step log is `results/grpo-length-200/sweep.log`. Weights stayed on the lab machine.
&nbsp;
## Checks

Steps 20, 40, and 60 score 50 validation prompts with Opus only. Step 80 scores 100 validation prompts with Opus and `gpt-5.5`. Both orders are scored. A pair counts when both orders name the same answer. Ties and unparsed pairs stay out of the denominator. The length band is trained words divided by matched-base words, from 0.8 through 1.2. The matched base is the gated-in pool sample closest in word count.

`gpt-5.5` used the Responses API, reasoning effort low, and store false. The returned model id was `gpt-5.5-2026-04-23`. The `gpt-5.4` fallback was not used. Detector means at step 80 are the same texts for both judges.
&nbsp;
## T = 4

The run stopped at step 40. The recorded reason is an Opus win rate under 40%. That rate is 17 of 44, which is 0.386364. The gate-fire rise streak peaked at 2. The under-45% streak stayed at 0, because step 20 was 19 of 40. Reward-judge parse failures are 0. Opus parse failures are 1. `gpt-5.5` was not called. The total is $3.068520. Weights are `results/grpo-length-200-t4-weights`.

| step | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 19/40 | 13/29 | 0.259578 | 0.286557 | 0.162038 | 0.236049 |
| 40 | 17/44 | 14/33 | 0.292563 | 0.269687 | 0.206323 | 0.173113 |
&nbsp;
## T = 2

The run finished 80 steps. No stop rule fired. The gate-fire rise streak peaked at 2. Reward-judge parse failures are 0. Opus parse failures are 5. `gpt-5.5` parse failures are 0. The total is $8.565829. Weights are `results/grpo-length-200-t2-weights`.

| step | judge | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | Opus | 26/43 | 16/28 | 0.145708 | 0.225814 | 0.246375 | 0.250888 |
| 40 | Opus | 23/42 | 19/35 | 0.129257 | 0.192768 | 0.192837 | 0.183753 |
| 60 | Opus | 22/41 | 21/37 | 0.150549 | 0.244242 | 0.135108 | 0.234949 |
| 80 | Opus | 49/90 | 37/72 | 0.199070 | 0.236935 | 0.173432 | 0.285208 |
| 80 | gpt-5.5 | 50/83 | 38/68 | 0.199070 | 0.236935 | 0.173432 | 0.285208 |
&nbsp;
## T = 1

The run stopped at step 40. The recorded reason is an Opus win rate under 40%. That rate is 18 of 46, which is 0.391304. Step 20 was 18 of 42, which is 0.428571, so that check was already under 45%. The under-40% rule is the one the run recorded, because a rate under 40% is checked before the two-check streak. The gate-fire rise streak peaked at 2. Reward-judge parse failures are 0. Opus parse failures are 1. `gpt-5.5` was not called. The total is $2.992922. Weights are `results/grpo-length-200-t1-weights`.

| step | agreed pairs | within 20% length | trained sigmoid | base sigmoid | trained DistilBERT | base DistilBERT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 18/42 | 12/27 | 0.366649 | 0.131796 | 0.611090 | 0.200572 |
| 40 | 18/46 | 10/31 | 0.362741 | 0.173465 | 0.626696 | 0.232660 |
&nbsp;
## Cost

| T | steps | stop | USD |
| --- | ---: | --- | ---: |
| 4 | 40 | Opus under 40% | 3.068520 |
| 2 | 80 | none | 8.565829 |
| 1 | 40 | Opus under 40% | 2.992922 |
| sum |  |  | 14.627271 |

Seed 2 did not start.
