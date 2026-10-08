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

The 41 saved dry-run signals were reused. The other 79 pairs were judged with `gpt-4.1-mini-2025-04-14`, both orders, temperature 0, the same A/B probability method. That was 158 forwards. The response `model` field was `gpt-4.1-mini-2025-04-14` on every new call. 105 of those calls had both letters in the top list, 53 used the chosen-only fallback, and 39 of those fallback calls were clamped. A missing signal was not used.

A group has a real spread when the largest of its three rewards exceeds the smallest by more than 1e-4 and the largest is above 0.

| T | groups with a real spread |
| ---: | ---: |
| 1 | 20 of 20 |
| 2 | 20 of 20 |
| 4 | 20 of 20 |
| 8 | 20 of 20 |

The smallest T that does this in most of the 20 groups is 1. It is not accepted for a pilot until the training config is reviewed.
&nbsp;
## Squared trigram at T = 4

No new judge calls. The signals and the raw human logits are the ones already saved. Check 2 is unchanged: there is no hard repetition cut, and the trigram ratio stays in the reward. This rerun changes the multiplier to the square of that ratio and sets T to 4.

reward_i = hard_i * (trigram_ratio_i ** 2) * sigmoid(z_i / 4) * length_score_i * wins_i / N

hard_i is 1 on all 80 answers. None is a stub under 30 words, and none reached 1616 tokens. Index 0 is the unpaid base opponent and is included in wins and in N. N is 3. A win is signal greater than 7. The spread is the largest reward minus the smallest.

| prompt | index 1 | index 2 | index 3 | spread |
| --- | ---: | ---: | ---: | ---: |
| validation-00001 | 0.133024 | 0.000000 | 0.000000 | 0.133024 |
| validation-00002 | 0.085306 | 0.088544 | 0.037088 | 0.051456 |
| validation-00003 | 0.101780 | 0.072368 | 0.000000 | 0.101780 |
| validation-00004 | 0.105513 | 0.102547 | 0.000000 | 0.105513 |
| validation-00005 | 0.040988 | 0.046864 | 0.000000 | 0.046864 |
| validation-00006 | 0.000000 | 0.113366 | 0.082150 | 0.113366 |
| validation-00007 | 0.000000 | 0.049903 | 0.091548 | 0.091548 |
| validation-00008 | 0.000000 | 0.067660 | 0.165232 | 0.165232 |
| validation-00009 | 0.089824 | 0.077947 | 0.000000 | 0.089824 |
| validation-00010 | 0.000000 | 0.034449 | 0.052662 | 0.052662 |
| validation-00011 | 0.000000 | 0.039016 | 0.000000 | 0.039016 |
| validation-00012 | 0.000000 | 0.037944 | 0.080910 | 0.080910 |
| validation-00013 | 0.047913 | 0.068515 | 0.000000 | 0.068515 |
| validation-00014 | 0.112294 | 0.272567 | 0.029681 | 0.242886 |
| validation-00015 | 0.038717 | 0.044303 | 0.112884 | 0.074167 |
| validation-00016 | 0.046096 | 0.039815 | 0.084776 | 0.044961 |
| validation-00017 | 0.106094 | 0.000000 | 0.020945 | 0.106094 |
| validation-00018 | 0.000000 | 0.104799 | 0.100232 | 0.104799 |
| validation-00019 | 0.077391 | 0.044260 | 0.063403 | 0.033132 |
| validation-00020 | 0.067290 | 0.104230 | 0.000000 | 0.104230 |

20 of the 20 groups have a spread above 1e-4 with the largest reward above 0. The median spread is 0.090686.
