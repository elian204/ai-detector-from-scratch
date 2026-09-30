# Stage 20 — GRPO diagnostics

Tooling we added to measure the stage-18 GRPO run. The published trainer,
`scripts/18_reinforcement-learning/05_train_grpo_human_writing.py`, is **never
edited**; everything here imports it and observes it.

| file | purpose |
| --- | --- |
| `run_grpo_baseline.py` | runs the published trainer under instrumentation (per-rollout JSONL, gradient norms, parameter-movement probe, manifest) |
| `analyze_run.py` | builds the reward-hacking table from a run's `rollouts.jsonl` |
| `numerics_probe.py` | measures bfloat16 update and reward resolution before training |

---

# How the trainer actually works

This section is the "Phase 0" of the project: the five mechanisms you have to
hold in your head to read a GRPO run. Each one is paired with its footprint in a
single real group from our float32 run — **step 39, target 100 words, token cap
176** — so the mechanism and its consequence can be seen together.

```
 i words  tok  cap?      P_AI   human  len_sc  reward     adv
 0   116  176  True  0.000286  0.9997  0.8621  0.8618  +0.9519
 1   115  176  True  0.000286  0.9997  0.8696  0.8693  +1.0284
 2   145  176  True  0.000393  0.9996  0.6897  0.6894  -0.8067
 3   153  176  True  0.000286  0.9997  0.6536  0.6534  -1.1736
```

Read that table once more before continuing. **All four rollouts have the same
human probability to four decimals.** Every bit of the advantage spread comes
from the length term. That single observation is the whole finding of Phase 1.

## 1. How rollouts become advantages

`compute_grpo_loss` takes **one prompt per step** and samples `num_rollouts`
completions from it (4 in our runs, 8 by default), in batches of
`rollout_batch_size`. Each completion is scored, giving rewards `r_0..r_{G-1}`.

`normalized_advantages` (trainer line 195) then computes

```
A_i = (r_i - mean(r)) / (std(r, unbiased=False) + 1e-4)
```

The baseline is **the group's own mean**. There is no value network and no
critic — that is what "group-relative" in GRPO means. The question asked of every
rollout is never "is this good?" but "did this beat its siblings on this exact
prompt?"

Verified on the step-39 group: mean reward 0.768483, population std 0.097952, and

```
i=1:  (0.869316 - 0.768483) / (0.097952 + 1e-4) = +1.0284   (logged +1.0284)
i=3:  (0.653408 - 0.768483) / (0.097952 + 1e-4) = -1.1736   (logged -1.1736)
```

Two consequences that matter more than they look:

* **Only the ranking inside the group carries signal.** The absolute reward level
  is discarded by the subtraction. A group where every rollout scores 0.9997 and
  a group where every rollout scores 0.0001 both produce zero learning.
* **The advantages are bounded.** With a biased (population) standard deviation,
  `max|A_i| = sqrt(G-1)`, so 1.732 for G=4. The `+1e-4` cannot blow up — when the
  std is tiny the numerator is tiny by the same amount. Advantages shrink toward
  zero rather than exploding.

## 2. Why empty text is scored as `"."`

`compute_human_writing_rewards` line 94:

```python
safe_texts = [text if text.strip() else "." for text in texts]
```

Because `validate_text` in `src/ai_detector/scoring.py` raises `ValueError` on
empty or whitespace-only text, so the verifier would crash on an empty rollout.

The subtlety is which string each quantity is computed from. The verifier sees
`safe_texts`, but line 108 computes `word_count = len(text.split())` from the
**original** `texts`. So an empty rollout gets `word_count = 0`, hence
`length_score = 0`, hence `reward = 0`. The substitution is therefore harmless
*to the reward*.

It is not harmless to the **logged metric**: `human_probability_mean` in
`metrics.csv` averages in `P_human(".")` for every empty rollout, so a run that
starts emitting empty strings can show a moving headline number that no gradient
ever saw. We measured `empty_frac = 0.00` in every window of both runs, so this
never bit us — but it is the reason `rollouts.jsonl` records an `is_empty` flag
per rollout rather than trusting the mean.

## 3. Why the length term still allows repetition

The reward is

```
r = (1 - P_AI) * min(w/t, t/w)
```

The length term constrains the **word count** and says nothing whatsoever about
content. 115 words of one sentence repeated twelve times scores exactly like 115
words of real prose.

Step 39, rollout 1, arithmetic closing exactly:

```
length_score = min(115, 100) / max(115, 100) = 0.8696
reward       = 0.9997 * 0.8696 = 0.8693      (logged 0.8693)
```

That rollout's text is `"1. A. Correcting the wrong code in the current
revision. 2. B. Correcting the wrong code in the current revision. 3. C. ..."`
repeated to the token cap. It is the **highest-reward** rollout in its group.

The deeper point is that the two factors are **multiplicative**. Once the
detector is fooled, `(1 - P_AI) -> 1.0` becomes a constant, and the only term
with a gradient left is the length heuristic. Our tables show this directly:
`length_score` tracks `reward` to four decimals from step ~50 onward. GRPO stops
optimizing "sound human" — it has already won that — and spends the remaining
450 steps learning to hit a word count.

## 4. `--skip-zero-advantage-updates`

If every reward in a group is identical, then `r_i - mean(r) = 0` exactly, so
every advantage is exactly zero, so `loss = -mean(0 * logprob) = 0`, so every
gradient is zero. The trainer detects this (line 252) with
`torch.allclose(advantages, zeros, atol=1e-8)`.

With the flag **on**, `loss_tensor` is returned as `None` and the guard at line
398 skips `zero_grad`, `backward`, clipping and `optimizer.step()` entirely.

With the flag **off**, `optimizer.step()` still runs on all-zero gradients — and
AdamW is not a no-op there. It decays existing momentum and, because
`weight_decay` defaults to 0.01, multiplies every weight by `(1 - lr*wd)`
regardless of gradient. You get parameter drift with no learning signal behind
it. That is why we passed the flag in both runs.

How often it fires was the prediction I got most wrong. I expected "rarely,"
reasoning that identical float rewards are unlikely. Measured in the float32 run:

| steps | zero-advantage | optimizer stepped |
| --- | ---: | ---: |
| 1-25 | 0 % | 100 % |
| 51-75 | 16 % | 84 % |
| 76-100 | **48 %** | **52 %** |

It fires constantly, and for the opposite reason to the one I had in mind: not
because the detector saturates at `P_AI = 1` (its confident-AI end), but because
it saturates at `P_AI ~ 0` — every rollout is equally "human", so there is
nothing left to rank. **The run switches itself off as it converges.**

## 5. The loss is REINFORCE — no ratio clip, no KL

```python
loss = -(advantages.detach() * logprobs).mean()      # trainer line 285
```

where `logprobs` is the **sum** of token log-probabilities over the completion
(`sequence_logprob`, line 192). Three absences, each load-bearing:

* **`advantages.detach()`** — the advantage is a constant. Gradient flows only
  through `logprob`. This is vanilla policy gradient: push up the probability of
  tokens in above-average rollouts, push down the rest.
* **No ratio and no clip.** PPO bounds how far one update can move the policy by
  clipping `pi_new/pi_old`. There is no old policy here, so nothing bounds the
  step. Note for Phase 3: a ratio clip is *inert* at the first update on fresh
  rollouts, because the ratio starts at exactly 1. Making clipping do anything
  requires multiple updates on retained rollouts, or some other source of policy
  lag — worth knowing before running that ablation and concluding "clip did
  nothing."
* **No KL to the base model.** Nothing anchors the policy to fluent English. The
  only reason the output is English at step 0 is that it started there. This is
  the single omission that makes degeneration the expected outcome, and it is
  deliberate: the upstream script is adapted from a chapter that adds clip and KL
  *later*, on purpose.

One measured detail about the magnitude. `train` calls
`clip_grad_norm_(model.parameters(), 1.0)`, and we log the norm *before*
clipping: it runs **280 to 2829**. So the gradient is scaled down by a factor of
roughly 300-2800 on every single step. Only the *direction* survives; the step
length is always essentially `lr`. Two things follow: the gradient norm is a
useful signal-strength readout but tells you nothing about step size, and the
learning rate is doing all the work of setting how far the policy moves — which
is exactly why the bfloat16 rounding threshold at `lr = 1e-5` mattered so much.

## Sampling is not quite on-policy

Worth knowing, and left unchanged for faithfulness: rollouts are sampled with
`temperature=0.8, top_p=0.9`, but `sequence_logprob` scores them under the
**unmodified** model distribution. The gradient is therefore not an exact
on-policy estimator for the distribution the samples came from.
