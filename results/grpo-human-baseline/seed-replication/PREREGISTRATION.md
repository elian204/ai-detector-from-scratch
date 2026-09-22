# Seed replication — pre-registered decision rule

Written **before** any replication run was launched: the file was created
at ~21:02 IDT on 2026-09-18 and the tmux session `grpo-seeds` was created at
21:03:02 (orchestrator log). This timestamp paragraph was then corrected at
21:05 because the first draft gave a wrong wall-clock time; nothing else in
the file was touched after launch.
Nothing below is changed after this point; if the rule turns out to be a poor
one, that is reported, not revised.

## Claim under test

"A bfloat16 policy avoids mode collapse and stays in the milder repetition
regime; a float32 policy mode-collapses and drifts to gibberish." Suggested by
the seed-42 pair (run A vs run B), where dtype and sampling trajectory are
confounded because dtype changes the logits and therefore the samples.

## Design

* Seeds **43, 44, 45**, each run with `--policy-dtype bfloat16` and
  `--policy-dtype float32`. Six new runs. Together with the existing seed-42
  pair: **n = 4 per dtype**.
* Everything else identical to runs A/B: 500 steps, 4 rollouts, targets
  {50, 100}, `--skip-zero-advantage-updates`, same trainer, same verifier.
* The seed controls both the prompt shuffle and the sampling RNG, so "seed"
  means "trajectory", which is what the claim is about.

## Metrics (fixed now)

Per run, computed by `analyze_run.py` over all 500 steps:

1. **`collapsed_steps`** — number of steps where all 4 rollouts are
   byte-identical. A run **mode-collapses** if `collapsed_steps >= 25` (5%).
   Reference: run A = 0, run B = 105.
2. **`reaches_gibberish`** — any 50-step window with mean base-model logprob
   below -2.0. Reference: run A = no, run B = yes (-3.55, -4.67, -2.30).

## Decision rule

The regulariser claim is **SUPPORTED** only if all four float32 runs
mode-collapse **and** none of the four bfloat16 runs do (4/4 vs 0/4). Under a
null where collapse is a coin flip independent of dtype, that pattern has
one-sided Fisher exact p = 0.014.

Any other outcome is **NOT SUPPORTED**, and the observed counts are reported
as-is. In particular, 3/4 vs 1/4 is reported as "dtype is not determinative",
not as "mostly supported".

`reaches_gibberish` is secondary and descriptive; it does not enter the rule.

## What this cannot show

Even 4/4 vs 0/4 does not identify *why*: it is consistent with bf16's frozen
parameters acting as a regulariser, and equally with bf16's slower effective
learning rate simply not reaching collapse within 500 steps. Distinguishing
those needs a float32 run at a matched effective step size, which is a
different experiment.
