&nbsp;
# Per-token log ratio on policy text, step 20

Forward passes only, on GPU 0. No training. Both `step-00020` weight files were on disk. The plotted value is `log π_step20(token | prefix) − log π_base(token | prefix)` on the trainer’s completion slice. `π_base` is frozen `Qwen/Qwen3-0.6B-Base` in float32. `π_step20` is that run’s own step-00020 checkpoint, also float32. Step 60 was not used. Both lines in a panel are answers the policy wrote.

The latest mixed group before step 20 that has a long loop (repetition_score ≤ 0.35 and at least 30 words) and a more-normal answer (repetition_score ≥ 0.7, at least two sentences, not mostly tags) is step 17 in both runs. The prompt is `train-03203`: “What should a thorough explanation of health disease studies cover?” Where a group had more than one long loop, the plot uses the one with the lower repetition score.

`rollouts.jsonl` does not store token ids, so both answers were re-tokenized. Counts against the logged `gen_tokens`:

| run | answer | index | logged gen | re-tokenized | rep |
| --- | --- | ---: | ---: | ---: | ---: |
| β = 0.05 | loop | 0 | 1616 | 1611 | 0.288 |
| β = 0.05 | normal | 3 | 233 | 229 | 0.723 |
| β = 0.5 | loop | 3 | 1616 | 1602 | 0.031 |
| β = 0.5 | normal | 1 | 271 | 268 | 0.805 |

The figure is `results/kl-token-ratio-policy.png`.
&nbsp;
## Means

Mean of the first 20 completion tokens, then the mean of every token after that. These are under the step-20 weights, not the logged step-17 kl.

| checkpoint | text | first 20 | the rest |
| --- | --- | ---: | ---: |
| β = 0.05, step 17 | loop, rollout 0 | −0.055 | −0.023 |
| β = 0.05, step 17 | normal, rollout 3 | −0.055 | 0.285 |
| β = 0.5, step 17 | loop, rollout 3 | −0.159 | −0.008 |
| β = 0.5, step 17 | normal, rollout 1 | −0.394 | 0.657 |

On the β = 0.05 panel the first 15 ratios match because the loop and the normal answer share that opening. On the β = 0.5 panel only the first 3 match.
&nbsp;
## What the normal answer does

The policy-written normal answer does pay a positive ratio, and it is spread through the body rather than piled on the first 20 tokens. Those first 20 are slightly negative in both runs.

Under β = 0.05, 200 of the 209 later tokens are positive. The windows after token 20 stay positive (tokens 21–50 average 0.664, 51–100 average 0.233, 101–200 average 0.263). The median of that tail is only 0.011, so the mean of 0.285 is pulled up by a right tail: 35 of those tokens are above 0.5.

Under β = 0.5 the positive ratio is plainer. Tokens 21–50 average 0.389, 51–100 average 0.895, 101–200 average 0.634, and 201–268 average 0.635. The median of the tail is 0.139. 218 of 248 later tokens are positive, and 86 of them are above 0.5.

The loops do not do this. Their tail means are −0.023 and −0.008, and the median of each tail is 0. A few tokens spike, but the repeated continuation is not sitting above the base the way the more-normal answer is.
