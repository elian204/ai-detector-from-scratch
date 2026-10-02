&nbsp;
# Does the loop’s opening spike grow?

Forward passes only, on GPU 0. No training. The texts are the step-17 answers already used in the policy-written figure, prompt `train-03203`. β = 0.05 uses rollout 0 as the loop and rollout 3 as the normal answer. β = 0.5 uses rollout 3 as the loop and rollout 1 as the normal answer. Each run’s own `step-00020`, `step-00050`, and `step-00060` weights were on disk. `π_base` is frozen `Qwen/Qwen3-0.6B-Base`. The ratio is `log π_checkpoint − log π_base` on the trainer’s completion slice. The texts were re-tokenized, same as before.

Re-tokenized length against logged `gen_tokens`: β = 0.05 loop 1611 vs 1616, normal 229 vs 233; β = 0.5 loop 1602 vs 1616, normal 268 vs 271.

The β = 0.05 loop and normal answer share their first 20 ratios at every checkpoint, so those two first-20 means match. The β = 0.5 pair shares only the first three tokens.

The figure is `results/kl-entry-spike.png`. It shows the first 40 tokens of each loop at the three checkpoints.
&nbsp;
## Means

| run | text | step | first 20 | the rest | mean of all |
| --- | --- | ---: | ---: | ---: | ---: |
| β = 0.05 | loop | 20 | −0.055 | −0.023 | −0.023 |
| β = 0.05 | loop | 50 | −1.536 | −0.195 | −0.211 |
| β = 0.05 | loop | 60 | −2.160 | −1.916 | −1.919 |
| β = 0.05 | normal | 20 | −0.055 | 0.285 | 0.256 |
| β = 0.05 | normal | 50 | −1.536 | −0.126 | −0.249 |
| β = 0.05 | normal | 60 | −2.160 | −0.250 | −0.417 |
| β = 0.5 | loop | 20 | −0.159 | −0.008 | −0.010 |
| β = 0.5 | loop | 50 | −1.900 | −0.298 | −0.318 |
| β = 0.5 | loop | 60 | −3.525 | −0.251 | −0.291 |
| β = 0.5 | normal | 20 | −0.394 | 0.657 | 0.579 |
| β = 0.5 | normal | 50 | −2.024 | 0.485 | 0.298 |
| β = 0.5 | normal | 60 | −3.175 | 0.261 | 0.004 |
&nbsp;
## Opening spike

A positive opening spike does not appear. At step 20 the first-20 means are near zero. By steps 50 and 60 they are largely negative: the later checkpoint finds the start of this step-17 text less likely than the frozen base does.

On the β = 0.05 loop the body falls with the opening. The rest goes from −0.023 at step 20 to −1.916 at step 60, so step 60 is a low ratio across the answer, not a spike with a flat tail. One token in the opening reaches about −20 and pulls the first-20 mean down, but the drop is not only that token: tokens below −1 go from 1 of 20 at step 20 to 7 of 20 at step 60. The normal answer’s body also loses the positive ratio it had at step 20 (0.285, then −0.126, then −0.250).

On the β = 0.5 loop the opening does separate from the tail. The first 20 go from −0.159 to −1.900 to −3.525, while the rest stays near −0.25. At step 60, 13 of those 20 tokens are below −1. That is an opening dip, and its sign is the opposite of the positive spike on the step-60 loop texts. The normal answer’s body stays positive (0.657, then 0.485, then 0.261) while its opening goes to −3.175.
&nbsp;
## Step-20 mean against the logged step-17 kl

The logged kl is the step-17 policy on the original token ids. The profile mean is the step-20 checkpoint on the re-tokenized text.

| answer | step-20 mean | logged step-17 kl |
| --- | ---: | ---: |
| β = 0.05 loop | −0.023 | 0.064 |
| β = 0.05 normal | 0.256 | 0.362 |
| β = 0.5 loop | −0.010 | 0.015 |
| β = 0.5 normal | 0.579 | 0.390 |

The two loops disagree in sign. Both numbers are small. The near-zero signs are not reliable: three more updates and a re-tokenization of a few tokens are enough to flip them. The normal answers keep the same sign. The levels still differ, by 0.11 and 0.19.
