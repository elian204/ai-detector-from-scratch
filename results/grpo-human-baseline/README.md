&nbsp;
# GRPO human-writing baseline — Phase 1 + 2

**Question.** Can no-KL GRPO against a frozen neural AI-text detector raise the
training verifier's "human" score, and is that real writing or reward hacking?

**Answer.** It raises the score to 0.9996 within ~50 steps and holds it there for
the remaining 450, while the text degrades monotonically from prose, to repeated
sentences, to word salad, to digit salad. It is reward hacking, and the training
verifier's score becomes uninformative almost immediately.

The single sentence worth taking away: a detector with **99.92% validation
accuracy** assigns **P(human) = 0.9995** to

```text
( Take 1 4 4 / 4 / 4 / ../ ../ /../ / 3  [ align ] Tarefas Group common Core 4l - 4l 4l 4/4 /4 4 / 4 /
```

&nbsp;
## Runs

Two runs, identical seed, prompts, reward, loss and hyperparameters. One
difference: the policy dtype. 500 steps, 4 rollouts/step, prompts filtered to
`target_words` in {50, 100} (1,781 available), `--skip-zero-advantage-updates`.
Both completed (`EXIT=0`), 6.7 GB and 12.2 GB peak respectively, ~1 h each in
parallel on two Quadro RTX 6000.

| | Run A | Run B |
| --- | --- | --- |
| policy dtype | bfloat16 (as published) | float32 |
| launcher | `run_bf16.sh` | `run_fp32.sh` |
| output | `run-A-bf16/` | `run-B-fp32/` |

`scripts/18_reinforcement-learning/05_train_grpo_human_writing.py` is **not
edited**. `scripts/20_grpo-diagnostics/run_grpo_baseline.py` imports it and
patches only its output paths and its logging sink. See
`scripts/20_grpo-diagnostics/README.md` for how the trainer works and what the
instrumentation does.

&nbsp;
## Before training: two bfloat16 defects in the published recipe

Both are properties of the number **format**, not of our GPUs. Native bf16
support buys speed, not precision, so an A100 behaves identically here.

**1. Most of the policy cannot move.** bf16 spacing near a typical Qwen3-0.6B
weight (median |w| = 0.0169) is 1.22e-4, and an AdamW step at the default
`lr = 1e-5` is far below half that. Measured over 4,000,416 sampled weights:
**86.9%** of parameters cannot move at `lr = 1e-5` (24.3% at 1e-4, 1.3% at 1e-3).
Confirmed with real AdamW: 20 steps moved 0/1000 bf16 parameters and 1000/1000
float32 ones. Confirmed again in the run itself: run A's movement probe sits at
**10.6%** for all 500 steps and never grows.

**2. The reward loses the ranking GRPO depends on.** The verifier softmaxes in
bf16 and casts to float32 afterwards, and the representable neighbours of 1.0 are
0.99609375 and 1.0. Three well-formed answers from the first smoke group:

| rollout | words | P_AI bf16 | P_AI fp32 | reward bf16 | reward fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 98 | 1.00000000 | 0.99976927 | 0.00000 | 0.00023 |
| B | 58 | 1.00000000 | 0.99927133 | 0.00000 | 0.00042 |
| C | 93 | 1.00000000 | 0.99861991 | 0.00000 | 0.00128 |

A 5.5× reward spread becomes an exact tie. Advantages are computed *within* a
group, so the tie deletes the signal from every confidently-detected rollout.

This defect matters only at initialization: once the policy is scored as *human*
the probabilities live at the other end of the range, where bf16 is fine. That is
why `rollouts.jsonl` logs both the training reward and a float32 reference score
— it is the only way to tell "the detector saturated" from "bf16 tied everything".

&nbsp;
## Result: the trajectory (float32, run B)

| steps | reward | human_train | human_heldout | dist-3g | base logprob | uniq/grp | collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 0.708 | 0.9714 | 0.8879 | 0.667 | -0.996 | 4.00 | 0.00 |
| 51-100 | 0.929 | 0.9997 | 0.9940 | 0.630 | -0.395 | 2.28 | 0.32 |
| 101-150 | 0.972 | 0.9997 | 0.9986 | 0.613 | -0.394 | 1.66 | 0.54 |
| 151-200 | 0.946 | 0.9996 | 0.9991 | 0.627 | -0.446 | 1.22 | **0.82** |
| 201-250 | 0.921 | 0.9996 | 0.9912 | 0.670 | -0.796 | 2.52 | 0.40 |
| 251-300 | 0.936 | 0.9997 | 0.9978 | 0.706 | -0.795 | 3.96 | 0.00 |
| 301-350 | 0.933 | 0.9997 | 0.9996 | 0.765 | -0.947 | 3.66 | 0.02 |
| 351-400 | 0.766 | 0.9996 | 0.8706 | 0.886 | **-3.551** | 3.82 | 0.00 |
| 401-450 | 0.844 | 0.9996 | 0.8926 | 0.913 | **-4.669** | 4.00 | 0.00 |
| 451-500 | 0.900 | 0.9996 | **0.7564** | 0.594 | -2.296 | 4.00 | 0.00 |

`human_train` is pinned at 0.9996 from step 51 onward and never moves again. Every
other column keeps changing. **The training verifier's score stops carrying
information after step ~50** — it reads the same for coherent repetition, for
word salad, and for digit salad.

&nbsp;
### Four regimes, not one

1. **Exploring** (1-50). Prose. Reward climbing.
2. **Repetition** (51-200). One sentence repeated to the token cap. Escalates
   into **mode collapse**: at step 192 all four rollouts are *byte-identical*
   despite temperature 0.8 / top-p 0.9, and 82% of groups in 151-200 are fully
   collapsed (105 of 500 steps overall). Identical rollouts give identical
   rewards, zero variance, zero advantage — so the step is skipped. The run
   stalls not for lack of signal but because it stopped exploring.
3. **Escape** (201-350). Diversity returns (uniq/group 1.22 -> 3.96). Because
   `--skip-zero-advantage-updates` freezes the policy on collapsed steps, only
   the rare high-variance prompts land updates, and those perturb it out of the
   basin. The flag is acting as an escape hatch from mode collapse.
4. **Gibberish** (351-500). Word salad, then digit salad. Reward *falls* from
   0.972 to 0.766: the policy abandons a better-rewarded exploit. With no KL and
   no clip, nothing holds it anywhere.

&nbsp;
### No single automatic metric catches both failure modes

| | repetition (regime 2) | gibberish (regime 4) |
| --- | --- | --- |
| distinct-3-gram | **falls** 0.667 -> 0.613 | **rises** 0.765 -> 0.913 |
| base-model logprob | **rises** -1.00 -> -0.39 | **falls** -0.95 -> -4.67 |
| held-out detector | agrees, 0.9986 | diverges, 0.8926 |

Repetition is maximally *predictable*, so a language model likes it and mean
token logprob goes up. Word salad never repeats a trigram, so distinct-n-gram
goes up. Each metric is blind to the failure the other catches, and a study
logging only one would confidently report the opposite of the truth. Both were
needed, plus reading the samples.

The project's pre-registered Phase 2 signature predicted held-out score flat-or-
falling and base logprob falling. Of five predicted signals, **two were inverted**
and they were inverted in *different* regimes. That is recorded here rather than
quietly corrected, because the inversion is the finding.

&nbsp;
## bfloat16 vs float32 — replicated across seeds

The seed-42 pair suggested that the bfloat16 policy avoids mode collapse and the
float32 policy does not. Because dtype changes the logits and therefore the
samples, that pair alone confounds dtype with trajectory, so it was replicated
under a decision rule fixed **before** launch
(`seed-replication/PREREGISTRATION.md`), on seeds 43, 44, 45.

| | float32 | bfloat16 |
| --- | ---: | ---: |
| runs that mode-collapse (>= 25 of 500 steps with 4 identical rollouts) | **4/4** | **0/4** |
| collapsed steps per run | 105, 462, 259, 456 | 0, 0, 0, 14 |
| collapse permanent once entered | 3/4 | — |
| reaches gibberish (base logprob < -2) | 1/4 | 0/4 |
| final training-verifier P(human) | 0.999, 0.999, 0.999, 0.999 | 0.998, 0.999, 0.884, 0.999 |

**Pre-registered rule: SUPPORTED** (one-sided Fisher exact p = 0.014).
Full result and samples: `seed-replication/RESULT.md`.

Three things the replication overturned from the single-pair reading:

* Run B's escape from collapse was **luck**: three of four float32 runs never
  escaped. `--skip-zero-advantage-updates` permits escape, it does not cause it.
* Gibberish was a one-trajectory outcome (run B only). The general float32 end
  state is a single repeated string — `Answer: Answer: Answer: ...` — rated
  0.999 human.
* "bf16 stays in the repetition regime" was wrong. Each bfloat16 seed settled
  in a different basin; seed 44 ended in **coherent on-topic prose** that the
  training verifier rates ~0.9 human and the held-out detector rates ~0.3 —
  verifier overfitting on real text, in the least-optimised run.

What remains **not** established is *why* bfloat16 avoids collapse. The frozen
87% of parameters (a structural regulariser) and the ~14× smaller effective step
(a rate effect that simply has not reached collapse by step 500) predict the same
table. Separating them needs a float32 run at matched step size or a bfloat16
run continued well past 500 steps.

&nbsp;
## Reproducing

```bash
uv sync --group training
uv run python scripts/15_classifier-api/download-models.py --fetch --model qwen3-variable

bash results/grpo-human-baseline/run_bf16.sh     # GPU 0
bash results/grpo-human-baseline/run_fp32.sh     # GPU 1

uv run python scripts/20_grpo-diagnostics/analyze_run.py \
  --run-dir results/grpo-human-baseline/run-B-fp32 --window 50 \
  --fluency-model Qwen/Qwen3-0.6B-Base --fluency-device cuda

uv run --with matplotlib python scripts/20_grpo-diagnostics/plot_run.py \
  --table "bfloat16=results/grpo-human-baseline/run-A-bf16/phase2-table.csv" \
  --table "float32=results/grpo-human-baseline/run-B-fp32/phase2-table.csv" \
  --output results/grpo-human-baseline/panels.png
```

&nbsp;
## Contents

| path | what |
| --- | --- |
| `run-A-bf16/`, `run-B-fp32/` | `manifest.json`, `metrics.csv`, `samples.txt`, `rollouts.jsonl`, `phase2-table.{csv,md}`, `checkpoints/` |
| `panels.png` | four diagnostic panels, both runs |
| `blind-taxonomy/` | 20 shuffled unlabelled rollouts, two blank rater sheets, and `key.json` (do not open before labelling) |
| `seed-replication/` | pre-registration, six replication runs, `RESULT.md`, `summary.txt` |
| `smoke/` | our 1-step smoke test |
| `codex-contaminated/` | logs polluted by a Codex consult; cited nowhere |

&nbsp;
## Still open

* Why bfloat16 avoids collapse: matched-step float32 (`lr ~ 7e-7`) or a long
  bfloat16 run, to separate "regulariser" from "slower".
* The blind hand taxonomy: two raters label `blind-taxonomy/blind-sample.md`
  independently, then agreement is reported. Categories already include both
  `repetition` and `gibberish`, which the trajectory showed are distinct.
* Phase 3 levers. Note two things learned here: a ratio clip is **inert** on the
  first update of fresh rollouts (the ratio starts at exactly 1), and
  `--skip-zero-advantage-updates` is not a neutral efficiency flag — it is load-
  bearing for escaping mode collapse, which makes toggling it the cheapest and
  most informative ablation available.
