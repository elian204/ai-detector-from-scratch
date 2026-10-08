&nbsp;
# Junk gate

No GRPO and no training. The gate is rules. No model scores a completion, and no model scores one prose answer against another. The reward plan below has no judge term.
&nbsp;
## Rules

A completion gets reward 0 when any rule fires. These rules were fixed before the validation run.

Words are whitespace tokens, `text.split()`. A 6-gram is six consecutive words, casefolded. The 6-gram count is the number of times the most common 6-gram occurs. The trigram ratio is the number of distinct casefolded word trigrams divided by the number of trigram positions. A text with fewer than three words has ratio 1. A sentence is the casefolded words between `.`, `!`, and `?`. The phrase "the answer is" is counted on casefolded words after stripping leading and trailing `.,;:!?"'`()[]` from each word. A line count uses stripped nonempty lines.

The gate fires when any of these is true:

- Loop: the 6-gram count is at least 4, or the trigram ratio is below 0.70, or a sentence of at least 8 words occurs at least 3 times.
- Padding: "the answer is" occurs at least 4 times, or one nonempty line occurs at least 4 times.
- Stub: fewer than 30 words.
- Cap hit: the rollout field `hit_token_cap` is true. That field is true when the completion used the run's response token limit, 416 or 1616. This rule is not applied to base answers or human texts.

The thresholds sit past the base answers and the human texts. On the 80 base answers the 6-gram count runs from 1 to 3, the trigram ratio from 0.7878 to 1.0000, the sentence repeat is 1, "the answer is" is 0, the line repeat is 1, and the length runs from 68 to 527 words. On the 40 cut human texts the 6-gram count runs from 1 to 2, the trigram ratio from 0.7560 to 1.0000, the sentence repeat is 1, "the answer is" is 0, the line repeat is 1, and the length runs from 63 to 211 words. A 6-gram count of 4 is the first integer above both maxima. A trigram ratio of 0.70 is below both minima. A sentence repeat of 3, a phrase count of 4, and a line repeat of 4 are above both maxima. A stub below 30 words is below both minima, 63 and 68.
&nbsp;
## Validation

One run. The 20 junk sides are the junk rollouts in the prose-versus-junk table in `results/grpo-judge-bias.md`. The 80 base answers are `results/grpo-base-detector-scores/base-answers.jsonl`. The 40 human texts are the ones named in `results/grpo-length-matched-human.md`, cut the same way: the first 211 whitespace words, with the six shorter texts kept whole. The cap rule was not applied to the base answers or the human texts.

| set | flagged |
| --- | ---: |
| 20 junk sides | 20 |
| 80 base answers | 0 |
| 40 human texts | 0 |
| 20 prose sides | 9 |

No miss. Every junk side fired, and no base answer or human text fired. There is no revised gate and no second count.

Every junk side fired the 6-gram rule. Other rules fired on some of them. The quote is the first twelve words.

| pair | rules | quote |
| ---: | --- | --- |
| 1 | 6-gram count 8; trigram ratio 0.637; cap hit | What should a thorough explanation of presidential authority cover? Answer: A thorough |
| 2 | 6-gram count 4; trigram ratio 0.467 | disease outbreak prediction requires background, current approaches, and unresolved questions. The background |
| 3 | 6-gram count 7; trigram ratio 0.193; sentence repeat 6; cap hit | the idea that software is something that people can create and control |
| 4 | 6-gram count 6; trigram ratio 0.534; sentence repeat 3; cap hit | The development of reproductive organs is a complex process that occurs during |
| 5 | 6-gram count 7; trigram ratio 0.614; cap hit | Surveillance is one of the most fundamental elements of political power and |
| 6 | 6-gram count 6; trigram ratio 0.591; sentence repeat 4; cap hit | To provide an answer, we should look back at how the field |
| 7 | 6-gram count 10; trigram ratio 0.283; sentence repeat 8; cap hit | The goal is to find a way to determine if a solution |
| 8 | 6-gram count 5; "the answer is" 5 | C Justification: The answer is A because in the question it is |
| 9 | 6-gram count 227; trigram ratio 0.017; cap hit | This is a multiple-choice question based on the text material. Please select |
| 10 | 6-gram count 7; trigram ratio 0.494 | Background: The World Health Organization and the Centers for Disease Control and |
| 11 | 6-gram count 6 | A thorough explanation of environmental regulations should cover the following: 1. The |
| 12 | 6-gram count 12; trigram ratio 0.245; sentence repeat 6 | The principles, challenges, and practical implications are as follows: Principles: 1. Understanding |
| 13 | 6-gram count 4; trigram ratio 0.537 | One of the most common misconceptions is that the matching algorithm is |
| 14 | 6-gram count 6; trigram ratio 0.549; sentence repeat 5 | Background: Human settlement locations are influenced by a variety of factors such |
| 15 | 6-gram count 4; trigram ratio 0.562 | A thorough explanation of health disease studies should cover the following: 1. |
| 16 | 6-gram count 360; trigram ratio 0.112; "the answer is" 360; cap hit | This is a follow up question to the original question. I think |
| 17 | 6-gram count 39; trigram ratio 0.055; sentence repeat 39; repeated line 39; cap hit | See "What's wrong with our software development process?" What else could be |
| 18 | 6-gram count 50; trigram ratio 0.067; sentence repeat 12; "the answer is" 11; repeated line 11; cap hit | The genetic code, which dictates the structure and function of proteins, is |
| 19 | 6-gram count 200; trigram ratio 0.013; repeated line 200; cap hit | A comprehensive explanation should include the following: - The background of the |
| 20 | 6-gram count 8; trigram ratio 0.357 | The following examples: 1. Linear Time-Varying (LTV) Systems - Example 1: Linear |

The gate also flags 9 of the 20 prose sides. Those flags are not misses. A prose side that hit the token cap, or that repeats enough to meet a loop rule, gets reward 0.

| pair | rules | quote |
| ---: | --- | --- |
| 2 | cap hit | Disease outbreak prediction involves several key areas of expertise and ongoing research. |
| 3 | trigram ratio 0.678 | In software management, one common misconception is that the software development process |
| 4 | 6-gram count 4; cap hit | Reproductive development is shaped by a variety of factors, including the species |
| 5 | cap hit | In the 1930s and 1940s, the United States experienced an era of |
| 6 | cap hit | In the 1960s, the health care field saw a significant amount of |
| 7 | 6-gram count 7; cap hit | Demographic data is often used in conjunction with population data. When it |
| 11 | 6-gram count 4; trigram ratio 0.676 | A thorough explanation of environmental regulations should cover the following: 1. The |
| 16 | 6-gram count 4; trigram ratio 0.697 | This is a follow-up question to "What background, current approaches, and unresolved |
| 18 | 6-gram count 4; trigram ratio 0.391; sentence repeat 4 | These factors include: 1) genetics, 2) the environment, 3) nutrition, 4) lifestyle, |
&nbsp;
## Identical answers

Ten forwards on `gpt-4.1-mini-2025-04-14` only. The texts are the ten base answers in the identical-answer table in `results/grpo-judge-bias.md`, and the same text is in both slots. One forward each. The prompt is `Which of these two answers to the same question is better? Reply with A or B.` Temperature is 0. The margin is log P(A) minus log P(B), from the first output token, with `top_logprobs` 5, the same A/B probability method as `results/grpo-openai-judge.md`, including the chosen-only fallback. All ten calls returned both letters in the top list, so the margin uses those log probabilities and the fallback was not used. The old 30, the held-out 30, and the junk pairs were not rescored.

The response `model` field was `gpt-4.1-mini-2025-04-14` on every call.

| words | prompt | index | margin |
| ---: | --- | ---: | ---: |
| 68 | `validation-00015` | 0 | 8.000 |
| 105 | `validation-00006` | 3 | 11.750 |
| 125 | `validation-00011` | 0 | 8.000 |
| 149 | `validation-00004` | 1 | 5.000 |
| 172 | `validation-00002` | 3 | 8.000 |
| 197 | `validation-00008` | 1 | 2.500 |
| 235 | `validation-00005` | 2 | 5.750 |
| 289 | `validation-00012` | 3 | 6.500 |
| 354 | `validation-00003` | 1 | 14.000 |
| 527 | `validation-00016` | 2 | 1.500 |

The tie threshold is 14, the maximum absolute margin in that list. A later pair is a tie unless its absolute signal is greater than 14, and so greater than every identical-answer margin. The ten margins run from 1.500 to 14.000. All ten are positive, so slot A is preferred when the texts match.
&nbsp;
## Rank plan, not run

Reward is 0 when the gate fires. Otherwise rank only the rollouts the gate let through. There is no judge term in the reward.

The provisional ranker, not yet accepted, is `gpt-4.1-mini-2025-04-14`. Its bar is the held-out set, which it already passed, plus later agreement with Eli's prose labels. It is not pinned.
&nbsp;
## Hand-label PDF

`results/grpo-prose-pairs.pdf` has 25 pairs. Each page set is the question and both full answers. A and B were assigned with a fixed seed. The PDF has no judge scores, no prose or junk labels, and no signals. Pairs 1–20 are the pairs in `results/grpo-prose-prose-to-label.md`. Pairs 21–25 are template essays from the trigram run, the loose restatement style rather than a pure loop, each against a normal prose rollout from the same prompt. The PDF does not mark which side is the template. The slot map is not in this repository.
