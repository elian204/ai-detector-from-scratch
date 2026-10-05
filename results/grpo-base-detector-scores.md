&nbsp;
# Frozen-base answers under the two homemade detectors

No training. Generation and scoring only, on GPU 0. The policy is frozen `Qwen/Qwen3-0.6B-Base` in float32, `eval()`, no gradient. Decoding matches the GRPO trainer: temperature 0.8, top-p 0.9, `max_new_tokens` 1616, `render_prompt` with target 250 words. Seed 42, set after the model was on the GPU and before the first sample.

The prompts are the GRPO validation split of `rasbt/human-writing-prompts-6k`, which the trainer does not update on. The first 20 rows in that split, `validation-00001` through `validation-00020`, four answers each, 80 answers. The dataset row for the first prompt has `target_words` 1000; the renderer was still given 250.

`score_many` returns temperature-scaled P(AI). P(human) below is 1 minus that. The training verifier is frozen `qwen3-variable`, temperature 1.4665638128271772, text truncated at 1023 tokens. None of the 80 reached that limit. Held-out DistilBERT uses temperature 1.2900265218781632 and truncates at 512 wordpieces. Eight answers were longer than that.
&nbsp;
## Scores

Mean word count 211. None of the 80 hit 1616. Word counts run from 68 to 527. Generated tokens run from 82 to 685.

| detector | mean P(human) | median | above 0.5 | above 0.9 |
| --- | ---: | ---: | ---: | ---: |
| qwen3-variable | 0.133 | 0.000 | 11 / 80 | 8 / 80 |
| DistilBERT | 0.223 | 0.0002 | 18 / 80 | 16 / 80 |

The scores are split. Under the training verifier, 56 of the 80 answers are exactly 0, and 8 are above 0.9. DistilBERT is the same shape: 56 are below 0.01, and 16 are above 0.9. Ten answers are above 0.5 on both detectors. Seven are above 0.9 on both.
&nbsp;
## Highest and lowest training score

The highest training P(human) is 0.998, on `validation-00004` sample 3 (124 words). DistilBERT also scores it 0.997.

```text
One of the common challenges in mathematical theory concepts is the need to establish a solid foundation in the underlying principles. The complexity of mathematical theories can make it difficult to understand and apply the concepts to real-world problems. Another challenge is the need to prove theorems and establish relationships between different mathematical concepts.
```

The lowest training score is 0, and 56 answers tie there. One of them is `validation-00001` sample 1 (226 words). DistilBERT scores that same answer 0.989, so the two detectors disagree on it.

```text
One of the most common misconceptions about safe learning systems is that they are inherently safe or effective. In reality, while safe learning environments can enhance safety and learning outcomes, they are not foolproof. Misconceptions arise from a misunderstanding of what makes a learning system safe.
```
