&nbsp;
# Per-token log ratio at step 60

Forward passes only, on GPU 0. No training. For each completion token the plotted value is `log π_step60(token | prefix) − log π_base(token | prefix)`, on the trainer’s completion slice `selected[prompt_length - 1 :]`. `π_base` is frozen `Qwen/Qwen3-0.6B-Base` in float32. `π_step60` is the step-00060 checkpoint of that run, also float32. Both checkpoints were on disk, so the figure has two panels.

The prompt is `train-00770` in both runs: “How should someone evaluate competing explanations or approaches related to player career progression and team performance analysis?” The same normal answer is scored under both checkpoints.

The loops are the stored step-60 texts, re-tokenized, because `rollouts.jsonl` does not keep token ids. The β = 0.05 loop is rollout 0, 496 words, 495 copies of `<content>`, logged `gen_tokens` 1616. Re-tokenization is 1487 completion tokens, so that panel is not the original id sequence. The mean of its plotted series is −0.019, against a logged kl of 0.030. The β = 0.5 loop is rollout 1, the reward winner, “Return again” 191 times, logged and re-tokenized length both 1616. The mean of that series is 0.047, against a logged kl of 0.056.

The normal answer is one new sample from the frozen base on that prompt: temperature 0.8, top-p 0.9, max 1616, seed 42, `render_prompt`, target 250. It is not a loop (201 words, 227 generated tokens, repetition score 0.990). Those generated ids were scored. It was not drawn again.

```text
When evaluating competing explanations or approaches related to player career progression and team performance analysis, it is crucial to consider several key factors to ensure a comprehensive and nuanced understanding.
```

The figure is `results/kl-token-ratio.png`. The axis is limited to −6 through 14 so the loop’s opening is visible. A few normal tokens lie below that frame: −19.4 under β = 0.05 and −10.9 under β = 0.5. The β = 0.5 loop reaches +13.0, which is inside the frame.
&nbsp;
## Means

Mean of the first 20 completion tokens, then the mean of every token after that.

| checkpoint | text | first 20 | the rest |
| --- | --- | ---: | ---: |
| β = 0.05 | `<content>` loop | 1.022 | −0.033 |
| β = 0.05 | normal prose | −1.410 | −2.604 |
| β = 0.5 | “Return again” loop | 2.896 | 0.011 |
| β = 0.5 | normal prose | −1.926 | −0.533 |
&nbsp;
## What the series does

Both loops are the first pattern. The ratio is concentrated in the opening and near 0 after it.

On the `<content>` loop, 6 of the first 20 tokens have an absolute ratio above 1, and the largest is +5.79 at position 4. None of the other 1467 tokens reach 0.5 in absolute value. The median of that tail is −0.0005.

On the “Return again” loop, 12 of the first 20 tokens have an absolute ratio above 1, and position 4 is +13.0. The bump fades rather than stopping at token 20: tokens 21–50 average +0.190, tokens 51–100 average +0.085, and after token 100 the windows sit between +0.025 and +0.002. Only 4 of the 1596 later tokens have an absolute ratio above 1. The repeated continuation is not paying a per-token KL.

The normal answer is not that pattern. Under β = 0.05 the rest is more negative than the opening (−2.604 against −1.410), and 103 of the 207 later tokens have an absolute ratio above 1. Under β = 0.5 the opening is the deep part (−1.926, including −10.9 and −9.4 on the first two tokens) and the rest is still −0.533, with 46 of 207 later tokens above 1 in absolute value. The step-60 policy gives this prose a lower probability than the frozen base does, across the answer, not only at the start.
