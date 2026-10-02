&nbsp;
# Does the winner’s opening spike grow?

Forward passes only, on GPU 0. No training. All three checkpoints exist for both runs. The ratio is `log π_checkpoint − log π_base` on the trainer’s completion slice. `π_base` is frozen `Qwen/Qwen3-0.6B-Base`. The strings were re-tokenized, because the logs do not store token ids.

The `<content>` loop first appears at step 56, rollout 3, prompt `train-04431`. All 72 nonempty lines are exactly `<content>`. Nothing earlier has even 10 such lines. It is the same family as the step-60 winner, and shorter: logged `gen_tokens` 256, re-tokenized 216. The 1616-token version first shows up at step 58.

“Return again” repeated the way the step-60 winner does first appears at step 60, rollout 1, prompt `train-00770`, 191 times. Logged and re-tokenized length are both 1616. Step 56 has the phrase once, inside a different loop, so that string was not used. The repeated winner is scored here under the earlier checkpoints as well.

The figure is `results/kl-winner-entry.png`. It shows the first 40 tokens at steps 20, 50, and 60.
&nbsp;
## Means

| string | first seen | checkpoint | first 20 | the rest |
| --- | ---: | ---: | ---: | ---: |
| `<content>`, 72 lines | step 56 | 20 | −0.181 | −0.037 |
| `<content>`, 72 lines | step 56 | 50 | 0.140 | −0.077 |
| `<content>`, 72 lines | step 56 | 60 | 0.871 | −0.009 |
| “Return again” × 191 | step 60 | 20 | 0.583 | 0.002 |
| “Return again” × 191 | step 60 | 50 | 0.711 | 0.003 |
| “Return again” × 191 | step 60 | 60 | 2.896 | 0.011 |
&nbsp;
## Opening spike

A positive opening spike does grow on both winning strings, and the tail stays near 0.

On the early `<content>` string the opening is slightly negative at step 20 (no token above 1). At step 50 the first-20 mean is 0.140, with 2 tokens above 1. At step 60 it is 0.871, with 6 tokens above 1, and the first token is 5.01. The rest of the 216 tokens stays near 0 (−0.037, then −0.077, then −0.009).

On “Return again” the shape is already there at step 20: first-20 mean 0.583, five tokens above 1, and the rest is 0.002. It grows. Step 50 is 0.711. Step 60 is 2.896, with 12 of the first 20 tokens above 1 and a peak of 13.0 at token 4. The other 1,596 tokens stay near 0 (0.002, then 0.003, then 0.011).

The step-60 numbers on “Return again” match the earlier plot of that same winner. The `<content>` numbers are for the short step-56 string, not the 495-line step-60 answer. Re-tokenization shortens that short string from 256 logged tokens to 216.
