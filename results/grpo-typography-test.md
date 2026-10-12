&nbsp;
# Typography test

The qualitative read in `results/grpo-t-sweep-qualitative.md` noticed that the trained answers at T = 1 drop curly quotes and markdown bold, and that texts with those marks score low on the detectors. It suggested the human-score gain might come partly from typography. This note tests that. No training and no API call was used.
&nbsp;
## Method

The texts are the 50 T = 1 step-40 evaluation answers in `results/grpo-length-200-t1/eval-step-0040.jsonl` and their 50 matched base answers. Each answer was rescored with the frozen `qwen3-variable` detector as sigmoid(z / 1), where z is the raw human-class logit, and with DistilBERT P(human). Both ran on CPU. Rescoring the unchanged texts reproduced the stored z and DistilBERT values exactly, with a maximum difference of 0.0.

Three normalizations were applied in a row. A is the one the test asked for. Curly single and double quotes and apostrophes become straight ones. Markdown bold and italics, headers, list bullets, block quotes, rules, code marks, and links are stripped to plain text. B adds ASCII replacements for the em dash, en dash, ellipsis, non-breaking space, arrow, and multiplication sign, and collapses runs of spaces. C adds the flattening of all whitespace, including newlines, to single spaces. C is a diagnostic and not part of the test.

The gap is trained minus base. The interval is a paired bootstrap over the 50 prompts, 4000 resamples.
&nbsp;
## Result

| Text | Trained sigmoid(z) | Base sigmoid(z) | Gap, sigmoid | Trained DistilBERT | Base DistilBERT | Gap, DistilBERT |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 0.363 | 0.173 | 0.189 (0.053 to 0.328) | 0.627 | 0.233 | 0.394 (0.218 to 0.563) |
| A, quotes and markup | 0.374 | 0.176 | 0.198 (0.066 to 0.332) | 0.650 | 0.291 | 0.360 (0.196 to 0.517) |
| B, A plus ASCII | 0.374 | 0.176 | 0.198 (0.066 to 0.332) | 0.650 | 0.291 | 0.359 (0.196 to 0.517) |
| C, B plus flat whitespace | 0.741 | 0.311 | 0.430 (0.280 to 0.583) | 0.650 | 0.291 | 0.359 (0.196 to 0.517) |

The mean raw logit z is -0.951 for trained and -3.695 for base on the original texts, a gap of 2.74. After A it is -0.834 and -3.511, a gap of 2.68.

The human-score gain does not vanish. After the requested normalization the sigmoid gap is 0.198, slightly above the original 0.189, and the DistilBERT gap is 0.360, below the original 0.394 by 0.034. Both intervals still exclude 0. The share of answers above 0.5 goes from 32% to 34% for trained and stays at 16% for base on the Qwen3 score, and from 62% to 64% against 22% to 28% on DistilBERT.

The DistilBERT gap shrinks because the base answers gain more than the trained ones. Base DistilBERT rises from 0.233 to 0.291, and trained rises from 0.627 to 0.650. Typography accounts for about one tenth of the DistilBERT gap. It does not account for the rest.
&nbsp;
## How many answers the normalization touched

Few trained answers changed. Under A, 2 of 50 trained answers changed and 48 are byte-identical. 11 of 50 base answers changed and 39 are identical. B changes 2 trained and 13 base answers. So the test barely touches the trained side, and the earlier reading, that dropped bold and curly punctuation explains the gain, is not supported. The low pooled scores of texts with bold or curly marks reflect that those marks mostly occur in base answers, which score low overall. Removing the marks lifts those texts only a little.

The corrected statement for `results/grpo-t-sweep-qualitative.md` is that typography is a small part of the DistilBERT gap and none of the Qwen3 gap.
&nbsp;
## Layout, the Qwen3 detector, and the diagnostic C

Flattening whitespace to single spaces raises the Qwen3 score a lot for both sides. The sigmoid mean goes from 0.374 to 0.741 for trained and from 0.176 to 0.311 for base, and the gap widens from 0.198 to 0.430. 33 trained and 25 base answers change under C, almost all by losing newlines. DistilBERT does not change under C, because its tokenizer ignores whitespace. So the Qwen3 detector reads layout, such as paragraph breaks and numbered lists on separate lines, as a signal. Its gain is not typography in the sense tested here, but it depends on layout, and a model could raise it by changing line breaks alone. The reward uses z without any flattening, so the training signal carries that sensitivity.
&nbsp;
## Files

The scores and the per-answer values are in `results/grpo-length-200-t1/typography-scores.json`. It is not committed. The script was `/tmp/grpo_typo.py`.
