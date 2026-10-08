&nbsp;
# Preflight 2

No GRPO, no training, and no pilot. Claude was not called.
&nbsp;
## Tie threshold

The accepted tie threshold is 7. A win is signal greater than 7. Absolute signal at most 7 gives the win to neither side. `results/grpo-training-config.md` uses that threshold.
&nbsp;
## Check 2, repetition ranges

These ranges are before any new threshold. The 6-gram count is the highest number of times one casefolded whitespace-word 6-gram occurs. The trigram ratio is the number of distinct casefolded word trigrams divided by the number of trigram positions. For an even count, the median is the average of the two central sorted values.

The junk sides are the 20 junk rollouts in the prose-versus-junk table in `results/grpo-judge-bias.md`. The preferred sides are the PDF slots named by the reference labels, excluding pairs 13, 14, 16, 18, and 21. The base answers are the 80 rows in `results/grpo-base-detector-scores/base-answers.jsonl`.

| group | n | 6-gram min | median | max | trigram min | median | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| junk sides | 20 | 4 | 7 | 360 | 0.0132 | 0.4804 | 0.7418 |
| preferred sides | 20 | 1 | 3 | 7 | 0.5365 | 0.7888 | 0.9970 |
| base answers | 80 | 1 | 1 | 3 | 0.7878 | 0.9732 | 1.0000 |

A gap would mean the junk range and the preferred range do not overlap. On the 6-gram count the junk range is 4 to 360 and the preferred range is 1 to 7, so they overlap. On the trigram ratio the junk range is 0.0132 to 0.7418 and the preferred range is 0.5365 to 0.9970, so they overlap. There is no gap. No hard cut is applied. A cut placed above the base answers would pass through preferred sides: the base 6-gram count stops at 3, and seven preferred sides are at 4 or above.

Mild repetition is left to the judge. The multiplier is

repetition_i = trigram_ratio_i

On the junk sides that multiplier has minimum 0.0132, median 0.4804, and maximum 0.7418. On the preferred sides it has minimum 0.5365, median 0.7888, and maximum 0.9970. On the base answers it has minimum 0.7878, median 0.9732, and maximum 1.0000.

Stub under 30 words and a completion that reached 1616 tokens stay hard zeros. A 416 `hit_token_cap` is not a cap hit. None of the three groups has a stub. Five junk sides reached 1616 tokens, pairs 9, 16, 17, 18, 19. None of the preferred sides reached 1616 tokens. None of the base answers reached 1616 tokens. On the 80 base answers the hard indicator is 1.
&nbsp;
## Check 3, softened human score

s_i = sigmoid(z_i / T). z_i is the frozen `qwen3-variable` logit for the human class, index 0, before the saved temperature scaling. That saved temperature is 1.4665638128271772. The usual score divides the logits by that temperature and then takes a softmax. These z_i values do not.

On the 80 base answers, z_i has minimum −7.3438, median −6.0469, and maximum 4.8125.

| T | s min | s median | s max |
| ---: | ---: | ---: | ---: |
| 1 | 0.000646 | 0.002362 | 0.991938 |
| 2 | 0.024798 | 0.046389 | 0.917303 |
| 4 | 0.137532 | 0.180691 | 0.769080 |
| 8 | 0.285372 | 0.319547 | 0.646014 |

length_score_i stays min(word count, 250) / max(word count, 250). Because check 2 found no gap, repetition_i is the trigram ratio, not 1. The reward is

reward_i = hard_i * repetition_i * s_i * length_score_i * wins_i / N

hard_i is 0 for a stub under 30 words or a real 1616-token cap, and 1 otherwise. Index 0 is the unpaid base opponent. Indexes 1, 2, and 3 are rewarded. The base opponent is included in wins and in N when it is not a hard zero. On these 80 answers it is not a hard zero, so N is 3. A win is signal greater than 7.

The saved dry run has both-order signals for 41 unordered pairs. The other 79 pairs were not saved. Every base answer has s_i above 0 at each of these four temperatures, so those missing signals are needed for the wins. The key file was not in the store, and those pairs were not judged. A missing signal is not treated as a tie. The spread counts are not computed.
