&nbsp;
# OpenAI judge qualification

No GRPO and no training. No model weights were downloaded. The API key was read from a local env file, loaded, and that file was deleted. The key is not written here. The 20 prose-versus-prose pairs were not scored.

The qualification set is the old 30 pairs in `results/grpo-judge-length-matched-gold.json`, the held-out 30 in `results/grpo-judge-methods-heldout-gold.json`, and the 20 prose-versus-junk pairs in `results/grpo-judge-bias.md`, same logs and same rollout indices. Each pair is scored in both orders. The prompt is `Which of these two answers to the same question is better? Reply with A or B.` The preferred side is the corrected essay, or the prose side on a junk pair.

m1 is the margin with the preferred side in slot B. m2 is the margin with the preferred side in slot A. The margin is log P(preferred slot) minus log P(other slot). signal = (m1 + m2) / 2. position bias = (m2 − m1) / 2. A probability pair passes only if signal is at least 3. Absolute signal below 3 is a tie and is not a pass. Signal at or below −3 is a miss.

The written run has no logprobs. A pass is both orders naming the preferred side. If the two orders name different answers, the pair is a disagreement and a fail. If both orders name the other side, the pair is a miss. There is no 3-nat rule on that run.

The pass gate is every held-out pair and every prose-versus-junk pair. The old 30 are reported and are not the gate. No run passed the gate, so no snapshot is pinned.
&nbsp;
## Price table

Rates are the standard text-token prices on the OpenAI model pages, retrieved 8 October 2026. The pages are `https://developers.openai.com/api/docs/models/gpt-5.4-nano`, `https://developers.openai.com/api/docs/models/gpt-5.4-mini`, and `https://developers.openai.com/api/docs/models/gpt-4.1-mini`. The gpt-4.1-mini page header price is $0.40 input and $1.60 output, which matches the text-token lines below. These are not batch rates. Dollars use the measured token counts from each response. Cached input was 0 on every call, so the cached rate was not charged. Reasoning tokens are already inside the output token count and are billed at the output rate.

| model | input / 1M | cached input / 1M | output / 1M |
| --- | ---: | ---: | ---: |
| `gpt-5.4-nano` | $0.20 | $0.02 | $1.25 |
| `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| `gpt-4.1-mini` | $0.40 | $0.10 | $1.60 |
&nbsp;
## Request mode

Probability calls use chat completions, temperature 0, and `top_logprobs` 5 on the first output token. When that list contains both A and B, the mode is `top_logprobs` and the margin uses those two log probabilities. When the list contains only the chosen letter, the mode is `chosen_only`: P(other) = 1 − P(chosen), with both probabilities clamped into [1e-12, 1 − 1e-12] so the log is finite. The request for `top_logprobs` did not error. The fallback is the case where the returned list has only the chosen letter.

`gpt-4.1-mini` rejected `reasoning_effort` with HTTP 400, so those 160 calls omit it. `gpt-5.4-nano` and the probability run of `gpt-5.4-mini` send `reasoning_effort` none. The written run uses the Responses API with reasoning effort low. That setting rejects `temperature`, so those 160 calls omit it. Each written reply was the single letter A or B.
&nbsp;
## Runs

The snapshot is the `model` field on the API response, and it was the same on every call of that run.

| run | snapshot | input | output | reasoning | USD | old pass | held-out pass | junk pass | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt-5.4-nano`, reasoning none | `gpt-5.4-nano-2026-03-17` | 84484 | 640 | 0 | 0.017697 | 7/30 | 1/30 | 6/20 | no |
| `gpt-5.4-mini`, reasoning none | `gpt-5.4-mini-2026-03-17` | 84484 | 712 | 0 | 0.066567 | 30/30 | 30/30 | 15/20 | no |
| `gpt-4.1-mini` | `gpt-4.1-mini-2025-04-14` | 84644 | 160 | 0 | 0.034114 | 30/30 | 30/30 | 17/20 | no |
| `gpt-5.4-mini`, reasoning low | `gpt-5.4-mini-2026-03-17` | 84484 | 5780 | 4660 | 0.089373 | 30/30 | 30/30 | 15/20 | no |

Probability modes, counted per call. A clamped call is a `chosen_only` call whose chosen probability was within 1e-12 of 0 or 1.

| run | top_logprobs | chosen_only | clamped |
| --- | ---: | ---: | ---: |
| gpt-5.4-nano | 50 | 110 | 0 |
| gpt-5.4-mini | 12 | 148 | 60 |
| gpt-4.1-mini | 22 | 138 | 82 |

| run | old pass | old tie | old miss | held-out pass | held-out tie | held-out miss | junk pass | junk tie | junk miss | junk disagree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-nano`, reasoning none | 7 | 23 | 0 | 1 | 29 | 0 | 6 | 13 | 1 | 0 |
| `gpt-5.4-mini`, reasoning none | 30 | 0 | 0 | 30 | 0 | 0 | 15 | 4 | 1 | 0 |
| `gpt-4.1-mini` | 30 | 0 | 0 | 30 | 0 | 0 | 17 | 1 | 2 | 0 |
| `gpt-5.4-mini`, reasoning low | 30 | 0 | 0 | 30 | 0 | 0 | 15 | 0 | 0 | 5 |

The written run has no ties. Its five junk failures are disagreements: pairs 1, 2, 15, 17, and 20. Both orders returned a letter, and the letters named different answers. No written pair had both orders name the other side.

`gpt-5.4-mini` with reasoning none passed all 30 held-out pairs and 15 of 20 junk pairs. The junk ties are pairs 5, 7, 17, and 20. The junk miss is pair 15, signal −3.000.

`gpt-4.1-mini` passed all 30 held-out pairs and 17 of 20 junk pairs. The junk tie is pair 1. The junk misses are pairs 5 and 15.

`gpt-5.4-nano` passed 7 of the old 30, 1 of the held-out 30, and 6 of 20 junk pairs. Its junk miss is pair 19, signal −3.000. The other junk failures are ties.
&nbsp;
## No snapshot is pinned

A model is pinned only if it passes every held-out pair and every prose-versus-junk pair. None of the four runs did. Passing the old 30 is not the gate. No snapshot id is pinned.
&nbsp;
## Per-pair probability signals

There is no winning run. The signals below are the full qualification, so the gate can be checked pair by pair.
&nbsp;
### `gpt-5.4-nano-2026-03-17`, reasoning none, Old 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -7.931 | 7.949 | 0.009 | 7.940 | tie |
| 2 | -5.584 | 7.203 | 0.809 | 6.394 | tie |
| 3 | -6.377 | 8.560 | 1.091 | 7.468 | tie |
| 4 | -6.840 | 7.303 | 0.232 | 7.071 | tie |
| 5 | -4.549 | 7.664 | 1.558 | 6.107 | tie |
| 6 | -3.688 | 6.863 | 1.588 | 5.275 | tie |
| 7 | -6.555 | 7.769 | 0.607 | 7.162 | tie |
| 8 | -3.688 | 5.848 | 1.080 | 4.768 | tie |
| 9 | -2.688 | 5.822 | 1.567 | 4.255 | tie |
| 10 | 0.063 | 6.683 | 3.373 | 3.310 | pass |
| 11 | -5.385 | 6.274 | 0.444 | 5.830 | tie |
| 12 | 0.688 | 5.663 | 3.175 | 2.488 | pass |
| 13 | -6.804 | 6.850 | 0.023 | 6.827 | tie |
| 14 | -3.996 | 8.079 | 2.042 | 6.037 | tie |
| 15 | -5.056 | 6.983 | 0.964 | 6.019 | tie |
| 16 | -3.625 | 9.341 | 2.858 | 6.483 | tie |
| 17 | -3.125 | 4.973 | 0.924 | 4.049 | tie |
| 18 | 1.625 | 5.426 | 3.526 | 1.901 | pass |
| 19 | -0.438 | 6.617 | 3.090 | 3.527 | pass |
| 20 | 3.606 | 6.217 | 4.912 | 1.305 | pass |
| 21 | -1.125 | 4.603 | 1.739 | 2.864 | tie |
| 22 | -3.688 | 5.014 | 0.663 | 4.351 | tie |
| 23 | -0.562 | 8.866 | 4.152 | 4.714 | pass |
| 24 | -3.812 | 8.545 | 2.366 | 6.179 | tie |
| 25 | -2.438 | 7.582 | 2.572 | 5.010 | tie |
| 26 | -4.028 | 8.200 | 2.086 | 6.114 | tie |
| 27 | -3.188 | 6.279 | 1.546 | 4.733 | tie |
| 28 | -5.916 | 7.221 | 0.652 | 6.569 | tie |
| 29 | 0.313 | 6.733 | 3.523 | 3.210 | pass |
| 30 | -3.688 | 8.366 | 2.339 | 6.027 | tie |
&nbsp;
### `gpt-5.4-nano-2026-03-17`, reasoning none, Held-out 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -6.858 | 9.190 | 1.166 | 8.024 | tie |
| 2 | -3.375 | 5.912 | 1.268 | 4.643 | tie |
| 3 | -2.625 | 4.995 | 1.185 | 3.810 | tie |
| 4 | -1.313 | 5.948 | 2.318 | 3.630 | tie |
| 5 | -7.133 | 8.049 | 0.458 | 7.591 | tie |
| 6 | -4.127 | 7.104 | 1.489 | 5.616 | tie |
| 7 | -1.875 | 6.401 | 2.263 | 4.138 | tie |
| 8 | -3.978 | 5.952 | 0.987 | 4.965 | tie |
| 9 | -1.938 | 5.934 | 1.998 | 3.936 | tie |
| 10 | -0.438 | 6.373 | 2.968 | 3.405 | tie |
| 11 | -6.042 | 7.889 | 0.924 | 6.965 | tie |
| 12 | -4.491 | 6.919 | 1.214 | 5.705 | tie |
| 13 | -2.812 | 6.428 | 1.808 | 4.620 | tie |
| 14 | -1.938 | 6.355 | 2.209 | 4.146 | tie |
| 15 | -5.074 | 8.239 | 1.583 | 6.656 | tie |
| 16 | -4.332 | 8.590 | 2.129 | 6.461 | tie |
| 17 | -5.141 | 6.863 | 0.861 | 6.002 | tie |
| 18 | -6.133 | 7.381 | 0.624 | 6.757 | tie |
| 19 | -2.688 | 8.025 | 2.669 | 5.356 | tie |
| 20 | -4.771 | 6.439 | 0.834 | 5.605 | tie |
| 21 | -1.750 | 6.542 | 2.396 | 4.146 | tie |
| 22 | -3.188 | 5.607 | 1.210 | 4.397 | tie |
| 23 | -0.500 | 5.972 | 2.736 | 3.236 | tie |
| 24 | -3.375 | 6.367 | 1.496 | 4.871 | tie |
| 25 | -0.625 | 7.960 | 3.668 | 4.293 | pass |
| 26 | -5.360 | 8.408 | 1.524 | 6.884 | tie |
| 27 | -3.926 | 5.951 | 1.013 | 4.938 | tie |
| 28 | -4.550 | 7.395 | 1.423 | 5.972 | tie |
| 29 | -4.837 | 7.355 | 1.259 | 6.096 | tie |
| 30 | -3.563 | 3.188 | -0.188 | 3.375 | tie |
&nbsp;
### `gpt-5.4-nano-2026-03-17`, reasoning none, Prose versus junk

| pair | run | step | prompt | m1 | m2 | signal | bias | result |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 416 | 5 | `train-01696` | -6.044 | 6.089 | 0.023 | 6.067 | tie |
| 2 | 416 | 8 | `train-03461` | -4.478 | 5.602 | 0.562 | 5.040 | tie |
| 3 | 416 | 10 | `train-03311` | 1.125 | 8.172 | 4.649 | 3.524 | pass |
| 4 | 416 | 15 | `train-04078` | 0.312 | 5.625 | 2.969 | 2.656 | tie |
| 5 | 416 | 16 | `train-04311` | -1.500 | 5.966 | 2.233 | 3.733 | tie |
| 6 | 416 | 17 | `train-03203` | -3.375 | 6.012 | 1.319 | 4.694 | tie |
| 7 | 416 | 20 | `train-03040` | -3.938 | 6.176 | 1.119 | 5.057 | tie |
| 8 | trigram | 9 | `train-04174` | -1.250 | 4.328 | 1.539 | 2.789 | tie |
| 9 | trigram | 16 | `train-04311` | 4.875 | 3.963 | 4.419 | -0.456 | pass |
| 10 | KL | 8 | `train-03461` | -0.688 | 6.047 | 2.680 | 3.367 | tie |
| 11 | KL | 9 | `train-04174` | 1.750 | 5.905 | 3.828 | 2.078 | pass |
| 12 | KL | 11 | `train-02842` | 5.282 | 8.166 | 6.724 | 1.442 | pass |
| 13 | KL | 12 | `train-04764` | -0.688 | 5.555 | 2.434 | 3.121 | tie |
| 14 | KL | 13 | `train-04918` | 3.125 | 4.475 | 3.800 | 0.675 | pass |
| 15 | KL | 17 | `train-03203` | -2.938 | 3.313 | 0.188 | 3.125 | tie |
| 16 | 1616 | 8 | `train-03461` | 3.562 | 8.129 | 5.846 | 2.283 | pass |
| 17 | 1616 | 10 | `train-03311` | -4.000 | 0.563 | -1.719 | 2.281 | tie |
| 18 | 1616 | 15 | `train-04078` | -5.858 | 6.109 | 0.125 | 5.984 | tie |
| 19 | 1616 | 17 | `train-03203` | -5.438 | -0.562 | -3.000 | 2.438 | miss |
| 20 | 1616 | 18 | `train-04745` | 0.312 | 2.688 | 1.500 | 1.188 | tie |
&nbsp;
### `gpt-5.4-mini-2026-03-17`, reasoning none, Old 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 9.341 | 12.477 | 10.909 | 1.568 | pass |
| 2 | 7.648 | 10.174 | 8.911 | 1.263 | pass |
| 3 | 10.279 | 10.867 | 10.573 | 0.294 | pass |
| 4 | 27.631 | 11.378 | 19.505 | -8.126 | pass |
| 5 | 12.477 | 10.079 | 11.278 | -1.199 | pass |
| 6 | 10.174 | 10.685 | 10.429 | 0.255 | pass |
| 7 | 10.685 | 27.631 | 19.158 | 8.473 | pass |
| 8 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 9 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 10 | 9.912 | 10.867 | 10.389 | 0.478 | pass |
| 11 | 9.643 | 27.631 | 18.637 | 8.994 | pass |
| 12 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 13 | 10.174 | 11.090 | 10.632 | 0.458 | pass |
| 14 | 9.643 | 27.631 | 18.637 | 8.994 | pass |
| 15 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 16 | 10.685 | 11.090 | 10.888 | 0.203 | pass |
| 17 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 18 | 9.769 | 10.867 | 10.318 | 0.549 | pass |
| 19 | 9.586 | 27.631 | 18.609 | 9.022 | pass |
| 20 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 21 | 9.643 | 27.631 | 18.637 | 8.994 | pass |
| 22 | 9.838 | 27.631 | 18.734 | 8.897 | pass |
| 23 | 10.279 | 27.631 | 18.955 | 8.676 | pass |
| 24 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 25 | 27.631 | 10.685 | 19.158 | -8.473 | pass |
| 26 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 27 | 10.279 | 11.378 | 10.829 | 0.549 | pass |
| 28 | 27.631 | 9.643 | 18.637 | -8.994 | pass |
| 29 | 27.631 | 8.813 | 18.222 | -9.409 | pass |
| 30 | 10.174 | 9.838 | 10.006 | -0.168 | pass |
&nbsp;
### `gpt-5.4-mini-2026-03-17`, reasoning none, Held-out 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 10.174 | 27.631 | 18.903 | 8.728 | pass |
| 2 | 10.279 | 27.631 | 18.955 | 8.676 | pass |
| 3 | 9.586 | 10.279 | 9.933 | 0.347 | pass |
| 4 | 12.477 | 27.631 | 20.054 | 7.577 | pass |
| 5 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 6 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 7 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 8 | 10.079 | 27.631 | 18.855 | 8.776 | pass |
| 9 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 10 | 10.685 | 10.079 | 10.382 | -0.303 | pass |
| 11 | 10.174 | 27.631 | 18.903 | 8.728 | pass |
| 12 | 9.769 | 27.631 | 18.700 | 8.931 | pass |
| 13 | 9.912 | 27.631 | 18.771 | 8.860 | pass |
| 14 | 9.992 | 10.279 | 10.136 | 0.144 | pass |
| 15 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 16 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 17 | 11.378 | 9.643 | 10.511 | -0.867 | pass |
| 18 | 11.090 | 27.631 | 19.361 | 8.270 | pass |
| 19 | 27.631 | 11.783 | 19.707 | -7.924 | pass |
| 20 | 9.992 | 9.643 | 9.818 | -0.174 | pass |
| 21 | 27.631 | 10.079 | 18.855 | -8.776 | pass |
| 22 | 9.643 | 27.631 | 18.637 | 8.994 | pass |
| 23 | 9.769 | 27.631 | 18.700 | 8.931 | pass |
| 24 | 9.643 | 9.643 | 9.643 | 0.000 | pass |
| 25 | 9.769 | 27.631 | 18.700 | 8.931 | pass |
| 26 | 9.838 | 10.079 | 9.958 | 0.121 | pass |
| 27 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 28 | 11.090 | 12.477 | 11.783 | 0.693 | pass |
| 29 | 8.433 | 9.643 | 9.038 | 0.605 | pass |
| 30 | 8.120 | 12.477 | 10.298 | 2.178 | pass |
&nbsp;
### `gpt-5.4-mini-2026-03-17`, reasoning none, Prose versus junk

| pair | run | step | prompt | m1 | m2 | signal | bias | result |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 416 | 5 | `train-01696` | 9.769 | -0.750 | 4.509 | -5.259 | pass |
| 2 | 416 | 8 | `train-03461` | 7.609 | 9.769 | 8.689 | 1.080 | pass |
| 3 | 416 | 10 | `train-03311` | 27.631 | 10.867 | 19.249 | -8.382 | pass |
| 4 | 416 | 15 | `train-04078` | 4.252 | 7.389 | 5.820 | 1.568 | pass |
| 5 | 416 | 16 | `train-04311` | 2.250 | 3.500 | 2.875 | 0.625 | tie |
| 6 | 416 | 17 | `train-03203` | 3.996 | 2.500 | 3.248 | -0.748 | pass |
| 7 | 416 | 20 | `train-03040` | 2.500 | 2.500 | 2.500 | 0.000 | tie |
| 8 | trigram | 9 | `train-04174` | 8.692 | 6.247 | 7.470 | -1.223 | pass |
| 9 | trigram | 16 | `train-04311` | 27.631 | 5.752 | 16.692 | -10.939 | pass |
| 10 | KL | 8 | `train-03461` | 6.265 | 0.250 | 3.258 | -3.008 | pass |
| 11 | KL | 9 | `train-04174` | 27.631 | 11.090 | 19.361 | -8.270 | pass |
| 12 | KL | 11 | `train-02842` | 8.839 | 10.079 | 9.459 | 0.620 | pass |
| 13 | KL | 12 | `train-04764` | 7.459 | 9.432 | 8.446 | 0.987 | pass |
| 14 | KL | 13 | `train-04918` | 10.685 | 27.631 | 19.158 | 8.473 | pass |
| 15 | KL | 17 | `train-03203` | -3.000 | -3.000 | -3.000 | 0.000 | miss |
| 16 | 1616 | 8 | `train-03461` | 9.432 | 5.254 | 7.343 | -2.089 | pass |
| 17 | 1616 | 10 | `train-03311` | -0.750 | 0.750 | 0.000 | 0.750 | tie |
| 18 | 1616 | 15 | `train-04078` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 19 | 1616 | 17 | `train-03203` | 4.757 | 6.955 | 5.856 | 1.099 | pass |
| 20 | 1616 | 18 | `train-04745` | 6.776 | -1.500 | 2.638 | -4.138 | tie |
&nbsp;
### `gpt-4.1-mini-2025-04-14`, Old 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 27.631 | 14.655 | 21.143 | -6.488 | pass |
| 2 | 27.631 | 14.978 | 21.304 | -6.327 | pass |
| 3 | 27.631 | 12.661 | 20.146 | -7.485 | pass |
| 4 | 27.631 | 13.132 | 20.381 | -7.250 | pass |
| 5 | 27.631 | 11.591 | 19.611 | -8.020 | pass |
| 6 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 7 | 27.631 | 13.678 | 20.655 | -6.976 | pass |
| 8 | 27.631 | 13.260 | 20.445 | -7.186 | pass |
| 9 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 10 | 27.631 | 25.125 | 26.378 | -1.253 | pass |
| 11 | 27.631 | 10.955 | 19.293 | -8.338 | pass |
| 12 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 13 | 27.375 | 27.631 | 27.503 | 0.128 | pass |
| 14 | 27.631 | 14.978 | 21.304 | -6.327 | pass |
| 15 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 16 | 27.631 | 14.978 | 21.304 | -6.327 | pass |
| 17 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 18 | 24.125 | 15.457 | 19.791 | -4.334 | pass |
| 19 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 20 | 27.631 | 14.215 | 20.923 | -6.708 | pass |
| 21 | 27.631 | 14.411 | 21.021 | -6.610 | pass |
| 22 | 27.631 | 11.469 | 19.550 | -8.081 | pass |
| 23 | 27.631 | 12.661 | 20.146 | -7.485 | pass |
| 24 | 27.631 | 14.411 | 21.021 | -6.610 | pass |
| 25 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 26 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 27 | 27.631 | 12.457 | 20.044 | -7.587 | pass |
| 28 | 27.631 | 10.169 | 18.900 | -8.731 | pass |
| 29 | 27.631 | 14.215 | 20.923 | -6.708 | pass |
| 30 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
&nbsp;
### `gpt-4.1-mini-2025-04-14`, Held-out 30

| pair | m1 | m2 | signal | bias | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 14.978 | 15.457 | 15.218 | 0.240 | pass |
| 2 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 3 | 27.631 | 11.617 | 19.624 | -8.007 | pass |
| 4 | 27.631 | 13.579 | 20.605 | -7.026 | pass |
| 5 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 6 | 27.631 | 14.411 | 21.021 | -6.610 | pass |
| 7 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 8 | 27.631 | 14.411 | 21.021 | -6.610 | pass |
| 9 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 10 | 27.631 | 13.911 | 20.771 | -6.860 | pass |
| 11 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 12 | 27.631 | 12.869 | 20.250 | -7.381 | pass |
| 13 | 27.631 | 13.489 | 20.560 | -7.071 | pass |
| 14 | 27.631 | 13.788 | 20.709 | -6.922 | pass |
| 15 | 27.631 | 13.911 | 20.771 | -6.860 | pass |
| 16 | 27.631 | 13.132 | 20.381 | -7.250 | pass |
| 17 | 27.631 | 13.260 | 20.445 | -7.186 | pass |
| 18 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 19 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 20 | 27.631 | 13.331 | 20.481 | -7.150 | pass |
| 21 | 27.631 | 12.588 | 20.110 | -7.521 | pass |
| 22 | 27.631 | 12.315 | 19.973 | -7.658 | pass |
| 23 | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 24 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 25 | 27.631 | 13.911 | 20.771 | -6.860 | pass |
| 26 | 27.631 | 14.978 | 21.304 | -6.327 | pass |
| 27 | 27.631 | 10.086 | 18.858 | -8.773 | pass |
| 28 | 27.631 | 15.457 | 21.544 | -6.087 | pass |
| 29 | 22.625 | 12.398 | 17.511 | -5.114 | pass |
| 30 | 27.631 | 11.018 | 19.324 | -8.307 | pass |
&nbsp;
### `gpt-4.1-mini-2025-04-14`, Prose versus junk

| pair | run | step | prompt | m1 | m2 | signal | bias | result |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 416 | 5 | `train-01696` | 18.750 | -17.500 | 0.625 | -18.125 | tie |
| 2 | 416 | 8 | `train-03461` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 3 | 416 | 10 | `train-03311` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 4 | 416 | 15 | `train-04078` | 19.250 | 16.750 | 18.000 | -1.250 | pass |
| 5 | 416 | 16 | `train-04311` | -5.000 | -12.000 | -8.500 | -3.500 | miss |
| 6 | 416 | 17 | `train-03203` | 17.500 | 10.000 | 13.750 | -3.750 | pass |
| 7 | 416 | 20 | `train-03040` | 22.000 | 15.000 | 18.500 | -3.500 | pass |
| 8 | trigram | 9 | `train-04174` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 9 | trigram | 16 | `train-04311` | 12.489 | 15.457 | 13.973 | 1.484 | pass |
| 10 | KL | 8 | `train-03461` | 27.631 | 10.750 | 19.191 | -8.441 | pass |
| 11 | KL | 9 | `train-04174` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 12 | KL | 11 | `train-02842` | 27.631 | 14.978 | 21.304 | -6.327 | pass |
| 13 | KL | 12 | `train-04764` | 27.631 | 27.631 | 27.631 | 0.000 | pass |
| 14 | KL | 13 | `train-04918` | 15.750 | 27.631 | 21.691 | 5.941 | pass |
| 15 | KL | 17 | `train-03203` | -7.250 | -16.500 | -11.875 | -4.625 | miss |
| 16 | 1616 | 8 | `train-03461` | 14.978 | 12.489 | 13.733 | -1.245 | pass |
| 17 | 1616 | 10 | `train-03311` | 8.470 | 1.438 | 4.954 | -3.516 | pass |
| 18 | 1616 | 15 | `train-04078` | 13.407 | 13.911 | 13.659 | 0.252 | pass |
| 19 | 1616 | 17 | `train-03203` | 16.000 | 20.250 | 18.125 | 2.125 | pass |
| 20 | 1616 | 18 | `train-04745` | 12.000 | -4.250 | 3.875 | -8.125 | pass |
&nbsp;
## Per-pair written verdicts

`gpt-5.4-mini-2026-03-17`, reasoning low. m1 is the letter when the preferred side is in slot B, so B is the preferred letter in that order. m2 is the letter when the preferred side is in slot A, so A is the preferred letter in that order. A pass needs B then A.

&nbsp;
### Written verdict, Old 30

| pair | m1 letter | m2 letter | result |
| ---: | --- | --- | --- |
| 1 | B | A | pass |
| 2 | B | A | pass |
| 3 | B | A | pass |
| 4 | B | A | pass |
| 5 | B | A | pass |
| 6 | B | A | pass |
| 7 | B | A | pass |
| 8 | B | A | pass |
| 9 | B | A | pass |
| 10 | B | A | pass |
| 11 | B | A | pass |
| 12 | B | A | pass |
| 13 | B | A | pass |
| 14 | B | A | pass |
| 15 | B | A | pass |
| 16 | B | A | pass |
| 17 | B | A | pass |
| 18 | B | A | pass |
| 19 | B | A | pass |
| 20 | B | A | pass |
| 21 | B | A | pass |
| 22 | B | A | pass |
| 23 | B | A | pass |
| 24 | B | A | pass |
| 25 | B | A | pass |
| 26 | B | A | pass |
| 27 | B | A | pass |
| 28 | B | A | pass |
| 29 | B | A | pass |
| 30 | B | A | pass |
&nbsp;
### Written verdict, Held-out 30

| pair | m1 letter | m2 letter | result |
| ---: | --- | --- | --- |
| 1 | B | A | pass |
| 2 | B | A | pass |
| 3 | B | A | pass |
| 4 | B | A | pass |
| 5 | B | A | pass |
| 6 | B | A | pass |
| 7 | B | A | pass |
| 8 | B | A | pass |
| 9 | B | A | pass |
| 10 | B | A | pass |
| 11 | B | A | pass |
| 12 | B | A | pass |
| 13 | B | A | pass |
| 14 | B | A | pass |
| 15 | B | A | pass |
| 16 | B | A | pass |
| 17 | B | A | pass |
| 18 | B | A | pass |
| 19 | B | A | pass |
| 20 | B | A | pass |
| 21 | B | A | pass |
| 22 | B | A | pass |
| 23 | B | A | pass |
| 24 | B | A | pass |
| 25 | B | A | pass |
| 26 | B | A | pass |
| 27 | B | A | pass |
| 28 | B | A | pass |
| 29 | B | A | pass |
| 30 | B | A | pass |
&nbsp;
### Written verdict, Prose versus junk

| pair | run | step | prompt | m1 letter | m2 letter | result |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | 416 | 5 | `train-01696` | B | B | disagree |
| 2 | 416 | 8 | `train-03461` | A | A | disagree |
| 3 | 416 | 10 | `train-03311` | B | A | pass |
| 4 | 416 | 15 | `train-04078` | B | A | pass |
| 5 | 416 | 16 | `train-04311` | B | A | pass |
| 6 | 416 | 17 | `train-03203` | B | A | pass |
| 7 | 416 | 20 | `train-03040` | B | A | pass |
| 8 | trigram | 9 | `train-04174` | B | A | pass |
| 9 | trigram | 16 | `train-04311` | B | A | pass |
| 10 | KL | 8 | `train-03461` | B | A | pass |
| 11 | KL | 9 | `train-04174` | B | A | pass |
| 12 | KL | 11 | `train-02842` | B | A | pass |
| 13 | KL | 12 | `train-04764` | B | A | pass |
| 14 | KL | 13 | `train-04918` | B | A | pass |
| 15 | KL | 17 | `train-03203` | A | A | disagree |
| 16 | 1616 | 8 | `train-03461` | B | A | pass |
| 17 | 1616 | 10 | `train-03311` | B | B | disagree |
| 18 | 1616 | 15 | `train-04078` | B | A | pass |
| 19 | 1616 | 17 | `train-03203` | B | A | pass |
| 20 | 1616 | 18 | `train-04745` | A | A | disagree |
