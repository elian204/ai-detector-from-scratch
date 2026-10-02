&nbsp;
# Does the frozen base loop on its own?

No training. Generation only, on GPU 0, which was idle (0 MiB used) when the job started. The model is frozen `Qwen/Qwen3-0.6B-Base` in float32, `eval()`, no gradient. Decoding matches the GRPO trainer: `do_sample=True`, temperature 0.8, top-p 0.9, `max_new_tokens` 1616, EOS cut included. The prompt is `render_prompt` from `scripts/18_reinforcement-learning/05_train_grpo_human_writing.py`: “Write a standalone answer… approximately 250 words. Return only the answer.” Seed 42, set with `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` after the model was on the GPU and before the first sample. Eight unique questions, four samples each, 32 samples. The whole 60-step set was not sampled.

A sample is a loop if `repetition_score` ≤ 0.35 or the single most common word-trigram is at least half of the trigrams. `repetition_score` is the trainer function: unique whitespace word-trigrams divided by all word-trigrams, and 1.0 when the text has fewer than three words. Mean log π is the mean of `completion_token_logprobs` on the completion slice.

Both KL metrics files have step 1 `kl` equal to `0.0` (`results/grpo-250w-kl/metrics.csv` and `results/grpo-250w-kl-beta0.5/metrics.csv`). Step 1 is also `zero_advantage` True. That is the policy still matching the frozen base, before any update.
&nbsp;
## Prompts

Questions already in the β = 0.05 list were not sampled again. The β = 0.5 steps 6, 8, 9, and 60 use the same questions as those steps of the β = 0.05 run. Two extra questions come from β = 0.5 steps 10 and 16.

| source | step | id | question |
| --- | ---: | --- | --- |
| grpo-250w-kl | 6 | train-02704 | What should a thorough explanation of social reform literature cover? |
| grpo-250w-kl | 7 | train-02983 | What background, current approaches, and unresolved questions are relevant to indigenous storytelling traditions? |
| grpo-250w-kl | 8 | train-03461 | What background, current approaches, and unresolved questions are relevant to disease outbreak prediction? |
| grpo-250w-kl | 9 | train-04174 | What should a thorough explanation of environmental regulations cover? |
| grpo-250w-kl | 11 | train-02842 | Which principles, challenges, and practical implications are most important when discussing analysis of particle properties? |
| grpo-250w-kl | 60 | train-00770 | How should someone evaluate competing explanations or approaches related to player career progression and team performance analysis? |
| grpo-250w-kl-beta0.5 | 10 | train-03311 | What common misconceptions affect discussions of software management, and how can they be corrected? |
| grpo-250w-kl-beta0.5 | 16 | train-04311 | What background, current approaches, and unresolved questions are relevant to historical political surveillance periods? |
&nbsp;
## The base’s own samples

0 of 32 are loops. 0 of 32 hit 1616. Repetition scores run from 0.840 to 1.000 (mean 0.968). The most common word-trigram is never more than 2.0% of the trigrams in a sample. Word counts run from 104 to 399 (mean 228). Generated tokens run from 128 to 508 (mean 273).

There is no loop set, so there is no loop-versus-normal log-prob split on the base’s own text. The mean per-token log π_base of all 32 samples is −0.688.

The lowest repetition score is still ordinary prose (`train-00770`, sample 2, 152 words, 172 tokens, repetition 0.840, mean log π −0.618):

```text
When evaluating competing explanations or approaches related to player career progression and team performance analysis, it is essential to consider various factors that contribute to these outcomes. These factors may include statistical measures, team dynamics, individual performance, and external influences such as coaching, management, and media coverage. It is important to weigh the strengths and weaknesses of each approach and make informed decisions based on the available evidence.
```

A sample near the 250-word target (`train-02842`, sample 2, 249 words, 293 tokens, repetition 0.976, mean log π −0.524):

```text
When discussing the analysis of particle properties, several key principles, challenges, and practical implications come into play. Firstly, it is crucial to understand the fundamental properties of particles, such as size, shape, surface area, and chemical composition, as these characteristics significantly influence their behavior in various environments. For instance, the size of a particle can affect its surface area, which in turn impacts its reactivity and interactions with other substances.
```

| prompt | i | words | gen | hit 1616 | rep | mode | loop | mean log π |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| train-02704 | 0 | 391 | 456 | no | 0.918 | 0.010 | no | −0.768 |
| train-02704 | 1 | 164 | 205 | no | 0.957 | 0.012 | no | −0.730 |
| train-02704 | 2 | 116 | 134 | no | 1.000 | 0.009 | no | −0.759 |
| train-02704 | 3 | 126 | 139 | no | 0.968 | 0.016 | no | −0.776 |
| train-02983 | 0 | 127 | 144 | no | 1.000 | 0.008 | no | −0.811 |
| train-02983 | 1 | 219 | 250 | no | 0.986 | 0.009 | no | −0.834 |
| train-02983 | 2 | 163 | 186 | no | 0.994 | 0.012 | no | −0.715 |
| train-02983 | 3 | 197 | 228 | no | 0.995 | 0.010 | no | −0.706 |
| train-03461 | 0 | 257 | 309 | no | 0.984 | 0.012 | no | −0.601 |
| train-03461 | 1 | 206 | 239 | no | 0.971 | 0.020 | no | −0.487 |
| train-03461 | 2 | 254 | 301 | no | 0.972 | 0.008 | no | −0.608 |
| train-03461 | 3 | 104 | 128 | no | 1.000 | 0.010 | no | −0.839 |
| train-04174 | 0 | 231 | 272 | no | 0.948 | 0.013 | no | −0.669 |
| train-04174 | 1 | 386 | 480 | no | 0.977 | 0.010 | no | −0.594 |
| train-04174 | 2 | 399 | 466 | no | 0.879 | 0.010 | no | −0.586 |
| train-04174 | 3 | 193 | 224 | no | 0.990 | 0.010 | no | −0.599 |
| train-02842 | 0 | 376 | 508 | no | 0.952 | 0.011 | no | −0.478 |
| train-02842 | 1 | 297 | 401 | no | 0.969 | 0.017 | no | −0.678 |
| train-02842 | 2 | 249 | 293 | no | 0.976 | 0.016 | no | −0.524 |
| train-02842 | 3 | 248 | 287 | no | 0.967 | 0.012 | no | −0.631 |
| train-00770 | 0 | 119 | 140 | no | 1.000 | 0.009 | no | −0.802 |
| train-00770 | 1 | 144 | 175 | no | 1.000 | 0.007 | no | −0.871 |
| train-00770 | 2 | 152 | 172 | no | 0.840 | 0.013 | no | −0.618 |
| train-00770 | 3 | 380 | 468 | no | 0.981 | 0.005 | no | −0.566 |
| train-03311 | 0 | 289 | 353 | no | 0.993 | 0.007 | no | −0.517 |
| train-03311 | 1 | 179 | 206 | no | 0.977 | 0.011 | no | −0.651 |
| train-03311 | 2 | 247 | 286 | no | 0.927 | 0.016 | no | −0.590 |
| train-03311 | 3 | 150 | 175 | no | 0.993 | 0.014 | no | −0.723 |
| train-04311 | 0 | 118 | 134 | no | 1.000 | 0.009 | no | −0.879 |
| train-04311 | 1 | 208 | 247 | no | 1.000 | 0.005 | no | −0.802 |
| train-04311 | 2 | 288 | 354 | no | 0.990 | 0.007 | no | −0.891 |
| train-04311 | 3 | 317 | 373 | no | 0.886 | 0.010 | no | −0.706 |
&nbsp;
## Logged policy answers, scored under the base

These are the already logged answers in the first five mixed groups of each KL run. They were not regenerated. `rollouts.jsonl` stores text, so each answer was re-tokenized: the same prompt with special tokens, then the stored text without special tokens. Mean log π_base is the mean completion-token logprob of that encoding. A few completion lengths differ from the logged `gen_tokens` by a handful of tokens.

Labels match the earlier mixed-group note. Degenerate means repetition ≤ 0.35, mostly tags, or fewer than 30 words. More normal means repetition ≥ 0.7, at least two sentences, and not a tag dump. The rest are neither. None of these forty answers was labeled degenerate for tags.

| run | kind | n | mean log π_base |
| --- | --- | ---: | ---: |
| β = 0.05 | degenerate | 6 | −0.437 |
| β = 0.05 | normal | 9 | −0.824 |
| β = 0.05 | neither | 5 | −0.666 |
| β = 0.5 | degenerate | 12 | −0.115 |
| β = 0.5 | normal | 5 | −1.226 |
| β = 0.5 | neither | 3 | −0.666 |

The β = 0.05 degenerate mean is pulled down by one 15-word stub at step 7 (mean log π_base −2.114, repetition 1.0, 17 logged tokens). The other five degenerate answers in that run are the long low-repetition texts, and their mean log π_base is −0.102. All 12 degenerate answers in the β = 0.5 groups are long repeats, mean −0.115. Across both runs the 17 long repeats average −0.112, and the 14 more-normal answers average −0.967.

A higher number here is more probable under the frozen base. The policy’s loops sit closer to 0 than the policy’s ordinary answers.
&nbsp;
## Conclusion

The frozen base does not loop often under this decoder. In 32 samples it produced no loop and never hit 1616. Staying close to it still cannot rule loops out. The loops the policy already wrote have a higher mean per-token log π_base than the more normal answers in the same early groups. Those loops are likely under the base even though this sampler did not emit them, so a penalty for leaving the base does not penalize them.
