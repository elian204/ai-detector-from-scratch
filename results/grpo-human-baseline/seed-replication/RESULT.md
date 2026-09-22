# Seed replication — result

Rule fixed in `PREREGISTRATION.md` before launch. Applied mechanically below.
All six new runs completed (`EXIT=0`), wave 1 21:03–21:58, wave 2 21:58–22:55
IDT, 2026-09-18.

## Primary metric: mode collapse (`collapsed_steps >= 25`)

| run | collapsed steps | collapses? |
| --- | ---: | --- |
| seed42 bf16 (run A) | 0 | no |
| seed43 bf16 | 0 | no |
| seed44 bf16 | 0 | no |
| seed45 bf16 | 14 | no |
| seed42 fp32 (run B) | 105 | **yes** |
| seed43 fp32 | 462 | **yes** |
| seed44 fp32 | 259 | **yes** |
| seed45 fp32 | 456 | **yes** |

**float32: 4/4. bfloat16: 0/4. Pre-registered rule: SUPPORTED** (one-sided
Fisher exact p = 0.014 under a coin-flip null).

Disclosed because the threshold would otherwise hide it: seed45-bf16 had 14
collapsed steps. bfloat16 is far less prone to collapse, not immune to it.

Collapse in float32 is usually **permanent**. Three of four fp32 runs never
escaped once collapsed (462, 259 and 456 steps, each a single unbroken tail to
step 500); only run B got out, around step 220. The earlier description of
`--skip-zero-advantage-updates` as an "escape hatch" is therefore too strong:
it *permits* escape if a high-variance prompt happens to arrive, and usually
none does.

What a collapsed float32 policy emits, step 480, all four rollouts identical:

```text
seed43:  choices  Answer: choices  Answer: choices  Answer: choices ...
seed45:  Answer: Answer: Answer: Answer: Answer: Answer: Answer: ...
```

Training-verifier P(human): 0.9991 and 0.9986.

## Secondary metric: gibberish (`base logprob < -2.0` in any window)

Only run B (seed42 fp32) reached it. 1/4 fp32, 0/4 bf16. Gibberish was a
one-trajectory phenomenon, not the float32 end state; the general float32 end
state is collapse onto a single degenerate string.

## Descriptive: what the bfloat16 runs did instead

| seed | final dist-3g | final held-out | final regime |
| --- | ---: | ---: | --- |
| 42 | 0.573 | 0.980 | repeated sentences |
| 43 | 0.876 | 0.858 | mostly coherent answers with fragmentary, repeated-question openers |
| 44 | 0.963 | 0.396 | **coherent, on-topic prose** |
| 45 | 0.546 | 0.987 | templated repetition (`Question #N is "Literature". Answer:`) |

Each bfloat16 seed settled in a different basin. What they share is not a text
style but *never collapsing*. Seed 44 is the one to look at: its text is a
genuine 100-word answer, the training verifier rates it 0.88–0.93 human, and
the held-out detector rates it 0.24–0.40 — the verifier-overfitting signature,
on real text, in the run with the least effective optimisation. It also has the
lowest reward of the eight (0.767): the less the policy exploited, the more it
wrote.

## Every run fooled the verifier

All eight runs end with training-verifier P(human) >= 0.88, seven of eight
>= 0.998. Whatever else varies by seed and dtype, this does not.

## What is and is not established

**Established (pre-registered, p = 0.014):** with this trainer, a float32
policy mode-collapses within 500 steps and a bfloat16 policy does not.

**Not established:** *why*. The prereg named two candidates and this design
cannot separate them:

1. bf16's frozen 87% of parameters act as a regulariser (a structural effect);
2. bf16 simply has a ~14× smaller effective step (measured mean |dw| 1.0e-5 vs
   1.5e-4) and has not *reached* collapse by step 500 (a rate effect).

Distinguishing them needs a float32 run at matched effective step size (e.g.
`lr = 7e-7`), or a bfloat16 run continued well past 500 steps. Until one of
those is run, "the published recipe's numerics bug is an accidental
regulariser" remains a hypothesis, and the defensible sentence is the narrower
one above.

## Full table

See `summary.txt` (generated) and each run's `phase2-table.{csv,md}`.
