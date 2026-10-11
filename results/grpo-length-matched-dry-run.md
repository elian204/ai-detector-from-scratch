&nbsp;
# Length-matched dry run

The training prompt asks for approximately 200 words. The `length_score` target stays 203. This dry run passed all three bars. The one-seed sweep starts after this note: T = 4, then T = 2, then T = 1, seed 42, 80 steps each, from scratch. A second seed is not in the plan.

Commit `a149083` recorded the earlier dry run, whose prompt asked for approximately 250 words. Those samples are not the pool or the rollouts here.
&nbsp;
## Samples

The prompt is "Write a standalone answer to the question below in approximately 200 words. Return only the answer." Frozen `Qwen/Qwen3-0.6B-Base` drew the answers at `max_new_tokens` 700, temperature 0.8, and `top_p` 0.9. Seed 42 was set after the model load. For `validation-00001` through `validation-00020`, the eight pool answers were drawn first and saved in `results/grpo-length-200/base-pool.jsonl`. The four dry-run rollouts for each prompt were drawn next, from that same seed stream, and saved in `results/grpo-length-200/rollouts.jsonl`. None of the 80 rollout texts is a copy of a pool text for the same prompt. They are not the old 250-word samples.

The pool median is 171.5. Its two central counts are 170 and 173. The rollout median is 188.0. Its two central counts are 184 and 192. The reward uses 203 as the length target.
&nbsp;
## Bars

`hard_i` is 0 under 100 words, over 300 words, under 30 words, or at 700 generated tokens. Counts from 100 through 300 stay in. Inside that window, `length_score` is min(words, 203) / max(words, 203).

On the 160 pool answers, 47 are hard zeros. The rate is 47 / 160 = 0.29375. Seventeen are under 100 words, 30 are over 300 words, and none are under 30 words. One sample also hit 700 generated tokens: `validation-00019` pool index 4, 554 words, which is already one of the 30 over 300. The bar is a gate-fire rate of at most 30%. This rate passes.

A real spread is max reward minus min reward above 1e-4, with the max reward above 0. Eighteen of the 20 groups have a real spread. The bar is at least 15. This count passes. The flat groups are `validation-00018` and `validation-00019`.

Parse failures are 0. That bar passes.

The reward judge is `claude-sonnet-5-5` with the JSON schema, no tools and no temperature. A win is both orders naming the same answer. The product is `hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N`. Wins count the other gated-in rollouts and that rollout's matched base. The matched base is the gated-in pool sample whose word count is closest. A tie on the absolute difference takes the lowest pool index. z_i is the raw human-class logit. A check of the saved `validation-00001` index-0 logit returned 1.1171875, the same value as the stored logit, with absolute error 0.

| prompt | index 0 | index 1 | index 2 | index 3 | spread | real |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| validation-00001 | 0.149365 | 0.000000 | 0.166734 | 0.195794 | 0.195794 | yes |
| validation-00002 | 0.083658 | 0.065653 | 0.144317 | 0.000000 | 0.144317 | yes |
| validation-00003 | 0.000000 | 0.000000 | 0.000000 | 0.135263 | 0.135263 | yes |
| validation-00004 | 0.000000 | 0.157622 | 0.217722 | 0.054317 | 0.217722 | yes |
| validation-00005 | 0.000000 | 0.000000 | 0.026168 | 0.106966 | 0.106966 | yes |
| validation-00006 | 0.000000 | 0.557130 | 0.146070 | 0.091852 | 0.557130 | yes |
| validation-00007 | 0.025831 | 0.022313 | 0.086533 | 0.121922 | 0.099609 | yes |
| validation-00008 | 0.092783 | 0.000000 | 0.031915 | 0.000000 | 0.092783 | yes |
| validation-00009 | 0.000000 | 0.000000 | 0.182459 | 0.000000 | 0.182459 | yes |
| validation-00010 | 0.000000 | 0.106850 | 0.026137 | 0.064858 | 0.106850 | yes |
| validation-00011 | 0.034131 | 0.375505 | 0.154226 | 0.072891 | 0.341374 | yes |
| validation-00012 | 0.000000 | 0.071026 | 0.000000 | 0.129749 | 0.129749 | yes |
| validation-00013 | 0.000000 | 0.000000 | 0.105614 | 0.000000 | 0.105614 | yes |
| validation-00014 | 0.093011 | 0.000000 | 0.000000 | 0.098503 | 0.098503 | yes |
| validation-00015 | 0.046680 | 0.000000 | 0.000000 | 0.000000 | 0.046680 | yes |
| validation-00016 | 0.000000 | 0.000000 | 0.000000 | 0.155561 | 0.155561 | yes |
| validation-00017 | 0.057812 | 0.000000 | 0.000000 | 0.000000 | 0.057812 | yes |
| validation-00018 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00019 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no |
| validation-00020 | 0.000000 | 0.000000 | 0.081456 | 0.072196 | 0.081456 | yes |

`validation-00018`: indexes 0, 2, and 3 are over 300 words. Index 1 has 138 words, is inside the gate, and won 0 of 1 against pool index 6 (142 words). `validation-00019`: all four rollouts are over 300 words, so every reward is 0.

| prompt | words 0–3 | wins/N | opponent pool:words |
| --- | --- | --- | --- |
| validation-00001 | 252, 95, 168, 184 | 1/3, 0/0, 1/3, 3/3 | 7:192, 6:100, 1:180, 2:182 |
| validation-00002 | 134, 139, 209, 180 | 2/4, 1/4, 4/4, 0/4 | 1:243, 1:243, 1:243, 1:243 |
| validation-00003 | 342, 91, 301, 201 | 0/0, 0/0, 0/0, 1/1 | 5:278, 2:156, 5:278, 6:183 |
| validation-00004 | 201, 121, 123, 174 | 0/4, 4/4, 3/4, 1/4 | 0:120, 0:120, 0:120, 0:120 |
| validation-00005 | 207, 213, 138, 182 | 0/4, 0/4, 1/4, 3/4 | 3:190, 3:190, 7:151, 4:182 |
| validation-00006 | 159, 153, 251, 103 | 0/4, 4/4, 2/4, 1/4 | 3:160, 2:155, 1:264, 4:103 |
| validation-00007 | 116, 108, 280, 300 | 1/4, 1/4, 3/4, 4/4 | 2:122, 2:122, 1:184, 1:184 |
| validation-00008 | 284, 397, 293, 109 | 3/3, 0/0, 1/3, 0/3 | 6:294, 6:294, 6:294, 0:111 |
| validation-00009 | 403, 430, 197, 464 | 0/0, 0/0, 1/1, 0/0 | 1:181, 1:181, 1:181, 1:181 |
| validation-00010 | 220, 172, 137, 153 | 0/4, 3/4, 1/4, 2/4 | 7:167, 7:167, 5:140, 3:150 |
| validation-00011 | 130, 147, 260, 137 | 1/4, 3/4, 4/4, 2/4 | 7:137, 7:137, 3:270, 7:137 |
| validation-00012 | 301, 138, 92, 226 | 0/0, 1/2, 0/0, 2/2 | 7:229, 3:140, 3:140, 1:226 |
| validation-00013 | 97, 90, 131, 93 | 0/0, 0/0, 1/1, 0/0 | 3:100, 3:100, 5:139, 3:100 |
| validation-00014 | 299, 238, 91, 192 | 3/3, 0/3, 0/0, 1/3 | 1:234, 1:234, 7:118, 3:219 |
| validation-00015 | 100, 178, 74, 301 | 1/2, 0/2, 0/0, 0/0 | 4:141, 2:178, 4:141, 0:292 |
| validation-00016 | 484, 379, 241, 233 | 0/0, 0/0, 0/2, 2/2 | 4:266, 4:266, 1:253, 1:253 |
| validation-00017 | 110, 83, 62, 110 | 1/2, 0/0, 0/0, 0/2 | 1:101, 1:101, 1:101, 1:101 |
| validation-00018 | 358, 138, 416, 311 | 0/0, 0/1, 0/0, 0/0 | 4:264, 6:142, 4:264, 4:264 |
| validation-00019 | 496, 351, 383, 301 | 0/0, 0/0, 0/0, 0/0 | 2:290, 2:290, 2:290, 2:290 |
| validation-00020 | 362, 313, 127, 225 | 0/0, 0/0, 1/2, 1/2 | 1:229, 1:229, 5:141, 1:229 |

The judge made 107 comparisons and 214 calls. Every returned model string was `claude-sonnet-5-5`. Cache reads and cache writes were 0. Input tokens were 216,160 and output tokens were 3,016. At $2 and $10 per 1M tokens, the cost is 216160 / 1,000,000 * 2 + 3016 / 1,000,000 * 10 = $0.462480.
&nbsp;
## Sweep that follows

The pass starts three runs, in order, from the base `Qwen/Qwen3-0.6B-Base` each time. Seed 42. Group size 4. `max_new_tokens` 700. T is 4, then 2, then 1. T is the divisor in `sigmoid(z / T)`. The reward judge is `claude-sonnet-5-5` JSON and stays out of the evaluation. The evaluation judge is `claude-opus-5-5`. At steps 20, 40, and 60 the check is 50 validation prompts scored with Opus only. At step 80 the check is 100 validation prompts scored with Opus and `gpt-5.5`. Both orders, agreement only, against the length-matched base. Each check reports the win rate on all agreed pairs and on pairs whose trained word count is between 0.8 and 1.2 times the matched base, plus the mean of `sigmoid(z / T)` and the mean DistilBERT score for the trained answers and for the matched base answers.

A run stops if the Opus win rate on all agreed pairs is under 40%, or under 45% at two checks in a row, or the gate-fire count rises for 20 consecutive steps. The note after the runs reports the USD total of each run. Weights stay on the lab machine. The pool files stay uncommitted.
