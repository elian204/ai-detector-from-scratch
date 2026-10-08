&nbsp;
# Position bias of the method 2 judge

No GRPO and no training. No downloads and no deletions. No API calls. The judge is the cached `Qwen/Qwen2.5-14B-Instruct-AWQ`. A 32B model was not loaded. Method 2 only: the next-token log probability of token id 32 (`A`) and token id 33 (`B`) after the chat template. No letter was generated. The prompt is `Which of these two answers to the same question is better? Reply with A or B.`

The both-order margins for the old 30, the held-out 30, and the 527-word support-vector essay were already saved, so those forwards were not run again. The identical-answer forwards and the prose-versus-junk forwards below are new. They ran on GPU 2.

For a pair with a preferred side, m1 is the margin when that side is in slot B, and m2 is the margin when that side is in slot A. The margin in an order is the preferred side's log probability minus the other side's. Then

signal = (m1 + m2) / 2

position bias = (m2 - m1) / 2

A positive signal means the preferred side wins on average. A positive bias means slot A is preferred. On the gold essays the preferred side is the corrected essay, so m1 is the order with the false essay in slot A, the same corrected-minus-false margin as in `results/grpo-judge-methods.md`.
&nbsp;
## Old 30

These are the 30 pairs in `results/grpo-judge-length-matched-gold.json`. Signal here is the average margin already used to score method 2.

| pair | m1 | m2 | signal | bias | |bias| > |signal| |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8.250 | 19.812 | 14.031 | 5.781 |  |
| 2 | -20.344 | 21.453 | 0.555 | 20.898 | yes |
| 3 | -20.891 | 20.031 | -0.430 | 20.461 | yes |
| 4 | -4.312 | 20.188 | 7.938 | 12.250 | yes |
| 5 | 5.500 | 23.234 | 14.367 | 8.867 |  |
| 6 | 15.094 | 20.250 | 17.672 | 2.578 |  |
| 7 | 17.766 | 24.359 | 21.062 | 3.297 |  |
| 8 | 12.625 | 21.969 | 17.297 | 4.672 |  |
| 9 | 21.531 | 22.141 | 21.836 | 0.305 |  |
| 10 | 6.969 | 22.609 | 14.789 | 7.820 |  |
| 11 | 19.484 | 21.953 | 20.719 | 1.234 |  |
| 12 | 18.281 | 22.297 | 20.289 | 2.008 |  |
| 13 | 7.156 | 17.016 | 12.086 | 4.930 |  |
| 14 | 18.500 | 21.453 | 19.977 | 1.477 |  |
| 15 | -5.969 | 14.125 | 4.078 | 10.047 | yes |
| 16 | 19.719 | 20.734 | 20.227 | 0.508 |  |
| 17 | 16.281 | 21.672 | 18.977 | 2.695 |  |
| 18 | 4.844 | 18.188 | 11.516 | 6.672 |  |
| 19 | 18.672 | 22.984 | 20.828 | 2.156 |  |
| 20 | 14.938 | 17.172 | 16.055 | 1.117 |  |
| 21 | 18.266 | 20.156 | 19.211 | 0.945 |  |
| 22 | 16.750 | 16.547 | 16.648 | -0.102 |  |
| 23 | 18.687 | 19.953 | 19.320 | 0.633 |  |
| 24 | 17.047 | 23.672 | 20.359 | 3.313 |  |
| 25 | 16.859 | 24.828 | 20.844 | 3.984 |  |
| 26 | 21.766 | 22.953 | 22.359 | 0.594 |  |
| 27 | 17.984 | 22.203 | 20.094 | 2.109 |  |
| 28 | 0.406 | 26.156 | 13.281 | 12.875 |  |
| 29 | 17.359 | 22.313 | 19.836 | 2.477 |  |
| 30 | 16.688 | 21.094 | 18.891 | 2.203 |  |

Absolute bias is greater than absolute signal on 4 of the 30 pairs: 2, 3, 4, 15.

Pair 3 is the gradient-descent pair. The false sentence is "Gradient descent steps in the same direction as the gradient of the loss." The corrected sentence steps in the opposite direction. m1 is -20.891 and m2 is 20.031, so the signal is -0.430 and the bias is 20.461. Absolute bias is greater than absolute signal. In both orders the next-token mass sits on A: the log probability of A is about -0.001 and the log probability of B is about -20. The false-in-A margin is slightly larger than the corrected-in-A margin, so the signal is negative. This is the selection pair method 2 missed.
&nbsp;
## Held-out 30

These are the 30 pairs in `results/grpo-judge-methods-heldout-gold.json`. Every signal is positive.

| pair | m1 | m2 | signal | bias | |bias| > |signal| |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -12.562 | 17.094 | 2.266 | 14.828 | yes |
| 2 | 11.000 | 20.344 | 15.672 | 4.672 |  |
| 3 | 10.938 | 21.438 | 16.188 | 5.250 |  |
| 4 | 19.203 | 17.516 | 18.359 | -0.844 |  |
| 5 | -20.328 | 23.281 | 1.477 | 21.805 | yes |
| 6 | 19.875 | 22.734 | 21.305 | 1.430 |  |
| 7 | 15.578 | 16.328 | 15.953 | 0.375 |  |
| 8 | -5.188 | 21.922 | 8.367 | 13.555 | yes |
| 9 | 21.531 | 18.781 | 20.156 | -1.375 |  |
| 10 | 19.844 | 25.187 | 22.516 | 2.672 |  |
| 11 | 16.719 | 21.781 | 19.250 | 2.531 |  |
| 12 | 20.578 | 18.812 | 19.695 | -0.883 |  |
| 13 | 13.281 | 18.578 | 15.930 | 2.648 |  |
| 14 | 18.125 | 17.328 | 17.727 | -0.398 |  |
| 15 | 13.797 | 20.141 | 16.969 | 3.172 |  |
| 16 | 21.781 | 24.734 | 23.258 | 1.477 |  |
| 17 | 18.672 | 25.297 | 21.984 | 3.312 |  |
| 18 | 17.109 | 25.438 | 21.273 | 4.164 |  |
| 19 | 18.688 | 23.609 | 21.148 | 2.461 |  |
| 20 | 14.344 | 22.891 | 18.617 | 4.273 |  |
| 21 | 15.078 | 22.016 | 18.547 | 3.469 |  |
| 22 | 20.719 | 19.547 | 20.133 | -0.586 |  |
| 23 | 19.969 | 19.891 | 19.930 | -0.039 |  |
| 24 | 16.547 | 23.391 | 19.969 | 3.422 |  |
| 25 | 20.469 | 23.078 | 21.773 | 1.305 |  |
| 26 | 17.047 | 23.172 | 20.109 | 3.063 |  |
| 27 | -1.531 | 19.141 | 8.805 | 10.336 | yes |
| 28 | 15.203 | 24.781 | 19.992 | 4.789 |  |
| 29 | 19.813 | 23.109 | 21.461 | 1.648 |  |
| 30 | -6.063 | 10.500 | 2.219 | 8.281 | yes |

Absolute bias is greater than absolute signal on 5 of the 30 pairs: 1, 5, 8, 27, 30.
&nbsp;
## Support-vector essay

The 527-word false essay against the 524-word correction has signal 2.453 and bias 6.172, and absolute bias is greater than absolute signal.
&nbsp;
## Identical answers

The same answer was placed in both slots. Each row is one forward pass. The margin is log P(A) minus log P(B). The ten answers are base completions in `results/grpo-base-detector-scores/base-answers.jsonl`, from 68 words through the 527-word answer at `validation-00016` index 2.

| words | prompt | index | margin |
| ---: | --- | ---: | ---: |
| 68 | `validation-00015` | 0 | 11.828 |
| 105 | `validation-00006` | 3 | 12.781 |
| 125 | `validation-00011` | 0 | 13.406 |
| 149 | `validation-00004` | 1 | 11.734 |
| 172 | `validation-00002` | 3 | 15.656 |
| 197 | `validation-00008` | 1 | 9.891 |
| 235 | `validation-00005` | 2 | 12.641 |
| 289 | `validation-00012` | 3 | 13.938 |
| 354 | `validation-00003` | 1 | 10.781 |
| 527 | `validation-00016` | 2 | 9.688 |

Every margin is positive, so slot A is preferred when the two texts match. The ten margins run from 9.688 to 15.656. The mean and the median are both 12.234. The margin does not grow with length. The 68-word answer is 11.828 and the 527-word answer is 9.688, and the largest margin is 15.656 at 172 words.
&nbsp;
## Prose against junk

Twenty pairs. Each pair is two different rollouts from the same prompt group in an existing log. The prose and junk labels follow `results/grpo-early-answer-labels.md`. Junk is a loop, padding, a stub, or a template that is not an answer. Prose is an attempt to answer, even where a fact is wrong. Steps 1–6 of the 1616, trigram, and KL logs share a start, and no shared text is used twice. Gold is that the prose side is preferred. m1 puts the junk in A and the prose in B. m2 puts the prose in A.

| pair | run | step | prompt | prose | junk | signal | bias | signal > 0 | |bias| > |signal| |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 416 | 5 | `train-01696` | r2 | r0 | -15.125 | -0.844 | no |  |
| 2 | 416 | 8 | `train-03461` | r2 | r1 | 19.625 | -0.234 | yes |  |
| 3 | 416 | 10 | `train-03311` | r0 | r1 | 18.734 | 1.391 | yes |  |
| 4 | 416 | 15 | `train-04078` | r3 | r0 | 2.891 | -8.109 | yes | yes |
| 5 | 416 | 16 | `train-04311` | r2 | r1 | 12.445 | -2.852 | yes |  |
| 6 | 416 | 17 | `train-03203` | r2 | r0 | 11.781 | -2.406 | yes |  |
| 7 | 416 | 20 | `train-03040` | r1 | r0 | -1.000 | -9.188 | no | yes |
| 8 | trigram | 9 | `train-04174` | r1 | r0 | 16.195 | -0.445 | yes |  |
| 9 | trigram | 16 | `train-04311` | r0 | r3 | 10.992 | -5.930 | yes |  |
| 10 | KL | 8 | `train-03461` | r0 | r3 | 21.133 | 1.492 | yes |  |
| 11 | KL | 9 | `train-04174` | r0 | r2 | 20.414 | 2.617 | yes |  |
| 12 | KL | 11 | `train-02842` | r0 | r1 | 20.055 | -0.664 | yes |  |
| 13 | KL | 12 | `train-04764` | r0 | r3 | 19.437 | 0.672 | yes |  |
| 14 | KL | 13 | `train-04918` | r3 | r2 | 17.531 | -0.828 | yes |  |
| 15 | KL | 17 | `train-03203` | r3 | r2 | 7.406 | 3.094 | yes |  |
| 16 | 1616 | 8 | `train-03461` | r1 | r2 | 18.961 | 3.305 | yes |  |
| 17 | 1616 | 10 | `train-03311` | r1 | r0 | -1.797 | -13.828 | no | yes |
| 18 | 1616 | 15 | `train-04078` | r3 | r0 | 13.875 | -2.563 | yes |  |
| 19 | 1616 | 17 | `train-03203` | r1 | r0 | 2.984 | -2.203 | yes |  |
| 20 | 1616 | 18 | `train-04745` | r2 | r3 | 5.422 | 4.453 | yes |  |

The prose side wins on 17 of the 20 pairs, where a win is signal > 0. Absolute bias is greater than absolute signal on 3 of the 20: pairs 4, 7, and 17. Pairs 7 and 17 are also two of the three losses. Pair 1 is the other loss: signal -15.125 and bias -0.844, so the junk side wins by a wide margin and the position term is small. Pair 7, `train-03040` in the 416-token log, has signal -1.000 and bias -9.188. Pair 17, `train-03311` in the 1616 log, has signal -1.797 and bias -13.828.

Twenty prose-versus-prose pairs from the same logs are written in full in `results/grpo-prose-prose-to-label.md`. They were not scored, and they have no gold label.
&nbsp;
## Training plan, not run

The judge in this plan sits on GPU 1. Four rollouts make 6 pairs, and each pair is scored in both orders, so a step is 12 one-token forward passes.

signal(i over j) = (m1 + m2) / 2

m1 is the order where i is in slot B, and m2 is the order where i is in slot A. The margin is the log probability of i's slot minus the log probability of j's slot.

wins_i is the count of other rollouts j with signal(i over j) > 0.

reward_i = (1 - P_AI_i) * length_score_i * wins_i / 3

There is no KL term and no trigram term. A group of four junk rollouts can still leak detector reward to whichever junk wins the comparisons.

An OpenAI qualification on 8 October 2026 did not pass the gate of every held-out pair and every prose-versus-junk pair. No snapshot is pinned. The four runs, the token counts, and the price table are in `results/grpo-openai-judge.md`.
