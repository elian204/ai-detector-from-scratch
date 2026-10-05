&nbsp;
# Length-matched human scores

No training. Scoring only, on GPU 0, one detector at a time. No Binoculars call and no quality judge.

The 40 texts are the same test-split label-0 rows as the step 0 table, ids 14, 15, 16, 17, 28, 29, 30, 31, 42, 43, 44, 45, 46, 47, 55, 56, 57, 58, 59, 94, 95, 96, 201, 283, 284, 300, 301, 302, 303, 304, 398, 399, 418, 419, 423, 424, 425, 426, 427, and 428. Each text was cut to its first 211 whitespace words, the step 0 base-answer mean rounded to a whole word. Thirty-four texts were longer than that and were cut. Six were already shorter, so they were kept whole:

| id | words |
| --- | ---: |
| 43 | 100 |
| 301 | 174 |
| 303 | 103 |
| 399 | 120 |
| 426 | 91 |
| 427 | 63 |

P(human) is 1 minus temperature-scaled P(AI) from `score_many`, the same call as step 0. Frozen `qwen3-variable` uses temperature 1.4665638128271772 and a 1023-token text limit. Frozen DistilBERT uses temperature 1.2900265218781632 and a 512-wordpiece limit. The six uncut texts were scored again with the rest. Their scores match the saved full-text scores. The largest difference is about 1e-9, on one DistilBERT value.
&nbsp;
## Scores

Full-text numbers are the step 0 human scores. Cut-text numbers are this rescore.

| text | qwen mean | qwen median | qwen > 0.5 | qwen > 0.9 | DistilBERT mean | DistilBERT median | DistilBERT > 0.5 | DistilBERT > 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.9997 | 0.9997 | 40 / 40 | 40 / 40 | 0.9977 | 0.9984 | 40 / 40 | 40 / 40 |
| first 211 words | 0.9997 | 0.9998 | 40 / 40 | 40 / 40 | 0.9986 | 0.9990 | 40 / 40 | 40 / 40 |

The human scores stay near 1. They do not drop. At more digits, the qwen means are 0.999677 full and 0.999688 cut. The DistilBERT means are 0.997662 full and 0.998556 cut. The lowest cut score is 0.9989 on qwen3-variable and 0.9874 on DistilBERT, and both of those lows are texts that were already shorter than 211 words.

On the full texts, qwen3-variable truncated 13 of 40 and DistilBERT truncated 24 of 40. On the cut texts, neither detector truncates any row. The longest cut text is 295 qwen tokens and 311 DistilBERT wordpieces.
