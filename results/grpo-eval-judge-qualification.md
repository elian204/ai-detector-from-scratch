&nbsp;
# Evaluation judge qualification

No training. `claude-opus-5-5` and `gpt-5.5` are the two evaluation judges of every judged run. Neither had been qualified. Both were run on the held-out 30 and on the 10 identical answers, the same sets and the same both-order rule as the Sonnet JSON qualification (`grpo-sonnet-json.md`). Each judge used its own evaluation protocol. Both returned the corrected side on all 30 held-out pairs. Neither returned a tie on the identical answers. Both put the letter A on every letter they gave, so a pair of identical answers ends as a tie only because the two orders name different answers, not because the judge said tie.
&nbsp;
## Protocol

The prompt was exactly "Which of these two answers to the same question is better? Reply with A or B." The layout was the evaluation layout: the prompt, then the question, then `A:` and its text, then `B:` and its text. It is the same prompt and layout the evaluation checks use.

`claude-opus-5-5` was called on the Messages API with `max_tokens` 1024 and no temperature. `gpt-5.5` was called on the Responses API with `max_output_tokens` 1024, reasoning effort low, and `store` false. A letter is read from the first nonempty line after markdown bold is stripped. The line has to open with A or B. A call with no letter was retried once. A second call with no letter stays a parse failure.

A pair counts as a win for a side only when both orders parse and name the same answer. If the orders name different answers, or either call still has no letter, the pair is a tie. That is the rule of the evaluation checks and of the Sonnet JSON qualification.

The held-out set is the 30 pairs in `results/grpo-judge-methods-heldout-gold.json`. The corrected text is the right side. Each pair was sent twice, with the corrected text in slot A and again in slot B. The identical set is the 10 base answers of `grpo-sonnet-json.md`, with the same text in both slots and sent twice. The reference pairs were not sent to Opus, because Claude Opus assigned their labels. `gpt-5.5` was run on them separately, and that result is in its own section below.

The returned model string was `claude-opus-5-5` on all 80 Opus calls and `gpt-5.5-2026-04-23` on all 120 `gpt-5.5` calls. Every HTTP status was 200.
&nbsp;
## Result

| | `claude-opus-5-5` | `gpt-5.5`, reasoning low |
| --- | --- | --- |
| held-out 30, both orders on the corrected side | 30 of 30 | 30 of 30 |
| held-out misses, disagreements, parse failures | 0, 0, 0 | 0, 0, 0 |
| held-out letters | corrected in A returned A, corrected in B returned B, on every pair | the same |
| identical answers, pairs that end as a tie under the evaluation rule | 10 of 10 | 10 of 10 |
| identical answers, calls that returned the letter A, B, or no letter | 13, 0, 7 | 20, 0, 0 |
| identical answers, pairs with A in both orders | 5 of 10 | 10 of 10 |
| identical answers, pairs with a missing letter in at least one order | 5 of 10 | 0 of 10 |
| identical answers, pairs with B in either order | 0 of 10 | 0 of 10 |
| calls that needed the retry | 13 of 80 | 0 of 80 |
| calls with no letter after the retry | 7 of 80, all on identical answers | 0 of 80 |
| cost, 80 calls | $0.461164 | $0.225570 |
| tokens, input and output | 67,031 and 9,652 | 33,414 and 1,950, of which 1,422 reasoning |

Agreement between the two orders on the held-out set is 30 of 30 for both judges. On the identical answers the two orders cannot agree on an answer, because both slots hold the same text, so the figure to read is the letter. Both judges named A and never B. That is a position preference for slot A. With real pairs, a judge that answers A when the sides are close would put the pair in a tie, since the two orders then name different answers.

The held-out result carries no position information, because the corrected side moved with the slot on every pair.

The identical-answer check therefore does not give the "no preference" result the Sonnet JSON judge gave (20 of 20 `tie`). It gives the result that matters to the evaluation rule, which is that no identical pair would count as a win for either side. It also shows that both judges lean to slot A when they cannot separate two answers, and the both-order rule is what removes that lean from the win rate.
&nbsp;
## Opus replies with no letter

Seven Opus calls had no letter after the retry, all on identical answers. The text of those calls was not saved in the main run. To see what they were, the seven prompts were sent once more, and the new replies were read. This is a fresh sample of the same prompts, not the original replies, and it cost about $0.05 by estimate from the token counts, which is not in the table above.

All seven new replies opened with a sentence that the two answers are identical, such as "The two answers are identical, so neither is better. If I must pick one: **A**", and then named A at the end. Opus recognized the tie in words and the letter parser did not read it, because the first line does not open with A or B. Four of the seven replies went on to name the faults the answers share. On identical answers Opus is therefore a prose tie judge that is pushed to pick A, and the parser turns the prose tie into a missing letter. On two close non-identical answers such a reply would drop the pair from the agreed pairs, which is the same effect as a tie.

Retries recovered 6 of the 13 calls that needed them. The replies of the other calls were not saved, so whether the pairs with A in both orders also carried a tie sentence after the letter is not known.
&nbsp;
## Identical answers

| prompt | words | `claude-opus-5-5` order 1 | order 2 | pair | `gpt-5.5` order 1 | order 2 | pair |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| validation-00015 | 68 | A | A | tie | A | A | tie |
| validation-00006 | 105 | A | none | tie | A | A | tie |
| validation-00011 | 125 | A | A | tie | A | A | tie |
| validation-00004 | 149 | A | none | tie | A | A | tie |
| validation-00002 | 172 | A | A | tie | A | A | tie |
| validation-00008 | 197 | none | none | tie | A | A | tie |
| validation-00005 | 235 | A | A | tie | A | A | tie |
| validation-00012 | 289 | A | A | tie | A | A | tie |
| validation-00003 | 354 | none | A | tie | A | A | tie |
| validation-00016 | 527 | none | none | tie | A | A | tie |

Opus had a missing letter on 3 of the 5 pairs of 197 words or more and on 2 of the 5 pairs below that. The sample is too small to say that length matters.
&nbsp;
## `gpt-5.5` on the reference pairs

This part is `gpt-5.5` only and is reported apart from the qualification above. The labels are Claude Opus labels (Claude app, version not recorded) that Eli Bogdanov reviewed as matching his judgment, not human labels (`grpo-reference-labels.md`, source text in `grpo-reference-labels-source.md`). The same 20 pairs in the three bands were sent as for Sonnet. Pairs 13, 14, 16, 18, and 21 were not sent. They are the labeller's own both-unacceptable set: "Both-unacceptable pairs: 14, 16 and 18, and arguably 13 and 21." (`grpo-reference-labels-source.md`). The letters are the PDF slots. A miss is both orders agreeing on the wrong side. A disagreement or a missing letter is not a miss. The cost was $0.199460 for 40 calls, and there were no retries and no parse failures.

High and medium-high pairs 3, 6, 7, 8, 10, 11, 15, 17, 19, 20, 22, and 25 are 12 of 12 correct. Medium pairs 1, 5, 9, 12, 23, and 24 have 5 correct and no miss. Pair 9 returned `B` then `B`, a disagreement. The bar of no misses on High and Medium-High and at most one on Medium is met. Low pairs 2 and 4 are both misses. Low misses are reported and do not enter the bar. Sonnet JSON missed the same two low pairs.

| pair | band | reference | display | swapped | result |
| ---: | --- | --- | --- | --- | --- |
| 1 | medium | B | B | A | correct |
| 2 | low | B | A | B | miss |
| 3 | high | A | A | B | correct |
| 4 | low | A | B | A | miss |
| 5 | medium | A | A | B | correct |
| 6 | high | A | A | B | correct |
| 7 | high | B | B | A | correct |
| 8 | high | B | B | A | correct |
| 9 | medium | B | B | B | disagree |
| 10 | high | A | A | B | correct |
| 11 | high | A | A | B | correct |
| 12 | medium | B | B | A | correct |
| 15 | high | A | A | B | correct |
| 17 | high | B | B | A | correct |
| 19 | high | B | B | A | correct |
| 20 | high | B | B | A | correct |
| 22 | high | B | B | A | correct |
| 23 | medium | A | A | B | correct |
| 24 | medium | A | A | B | correct |
| 25 | high | B | B | A | correct |

Of the 20 pairs, 19 had the two orders agree on an answer, and 17 of those agreed on the reference side. The 17 correct pairs are 12 High, 5 Medium, and 0 Low.
&nbsp;
## Cost and spend

| item | cost |
| --- | ---: |
| Opus, 80 calls on held-out and identical | $0.461164 |
| `gpt-5.5`, 80 calls on held-out and identical | $0.225570 |
| `gpt-5.5`, 40 calls on the reference pairs | $0.199460 |
| Opus, 7 re-sampled calls to read the replies with no letter | about $0.05, estimated |

Opus is priced at $4 per million input tokens and $20 per million output tokens. `gpt-5.5` is priced at $5 per million input tokens and $30 per million output tokens, with reasoning tokens counted in the output. The two smoke calls on a trivial pair are not counted. The total spend for this note is about $0.94.
&nbsp;
## What this means for the evaluation checks

Both judges pass the held-out set with no miss, so their both-order agreement is a usable signal for a clear factual difference. Neither shows a tie on identical answers in the sense of a tie verdict. Both lean to slot A when the answers are the same, and Opus often says in words that they are the same. The agreed-pairs rule is what handles this, since a lean to one slot makes the two orders name different answers and the pair drops out. The held-out set has a one-sentence factual difference, so it says nothing about how either judge reads a difference of style or length. Opus has no reference-pair result, because the labels came from Opus.
&nbsp;
## Files

The call-level state and the summary JSON stay outside the repository with the other judge caches. They hold only letters, token counts, and costs, and the API keys were read at run time from a local file and are not written anywhere.
