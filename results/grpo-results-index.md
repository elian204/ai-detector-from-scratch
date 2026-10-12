&nbsp;
# GRPO results index

This file indexes the GRPO work on `Qwen/Qwen3-0.6B-Base` against the frozen `qwen3-variable` detector, with and without a judge in the reward. It was built from the existing notes under `results/`, the committed `run.log` files, and the saved `summary.json` and `steps.jsonl` files of the 200-word runs, which stay on the lab machine. No training and no judge call was made for it. A cell that the material does not fill says `n/a` and gives the reason. The inconsistencies found while reading are listed at the end of the Confounds section.
&nbsp;
## Headline findings

Each point is a statement of a note, with the note named.

1. Training against the detector alone raises the training verifier's P(human) to about 0.999 within about 50 steps and the text degrades. In the short-target pilot the float32 runs mode-collapsed in 4 of 4 seeds and the bfloat16 runs in 0 of 4, which met the pre-registered rule (Fisher p = 0.014). The 250-word float32 runs ended in loops that both detectors call human. Sources: `grpo-human-baseline/README.md`, `grpo-human-baseline/seed-replication/RESULT.md`, `grpo-lab-note.md`.
2. Two reward judges were used. `gpt-4.1-mini` failed the OpenAI gate and was accepted for the 40-step pilot on the reference bands, and that pilot stopped at step 20. `claude-sonnet-5-5` with a JSON schema then passed (identical answers 10 of 10 ties, held-out 30 of 30) and is the reward judge of every later run. The local Qwen2.5 pairwise judges failed the support-vector pair, and a forced Sonnet tool call was rejected by the API. The evaluation judges, Opus and `gpt-5.5`, were never qualified. Sources: section 2.
3. With Sonnet in the reward and Opus as the evaluation judge, no reward so far held quality and raised both detector scores. T = 1 raised both detectors about 3 times and Opus preferred the trained answer in 18 of 46 pairs. T = 4 stopped at step 40 on 17 of 44 with both detectors slightly above the base, and T = 2 finished 80 steps at 49 of 90 with both below the base. The slope reward held quality (Opus 49 of 83, `gpt-5.5` 51 of 80 at step 80) with both detectors below the base at three of its four checks and level at the fourth. The rank reward at step 40 and the std-floor reward at w = 0.3 at step 20 raised both detectors and lost quality (13 of 48 in both). Sources: `grpo-length-200-t-sweep.md`, `grpo-slope-reward.md`, `grpo-rank-reward.md`, `grpo-std-floor-reward.md`.
4. The T = 1 detector gain is not typography. Straightening quotes and stripping markup left the Qwen3 sigmoid gap at 0.198 (from 0.189) and the DistilBERT gap at 0.360 (from 0.394). The Qwen3 detector does read layout: flattening whitespace moved its trained mean from 0.374 to 0.741. Source: `grpo-typography-test.md`.
5. Length is not pinned by the reward. Slope seed 2 stopped at step 41 and std-floor w = 0.1 at step 37, both on a median length above 350 words, while slope seed 42 never passed a 10-step median of 205. Sources: `grpo-slope-seed2.md`, `grpo-std-floor-reward.md`, `grpo-slope-reward.md`.
6. The training judge did not clearly track the evaluation judge. In the std-floor w = 0.3 run Opus fell to 27% at step 20 while Sonnet's training q stayed at 0.45 to 0.50. In the rank run both fell. The adaptive constraint was therefore not run. Source: `grpo-sonnet-vs-opus-q.md`.
7. Every setting has one seed except slope, and the earliest sweep and quality-gate runs did not skip zero-advantage steps. See the Confounds section. Source: `grpo-slope-reward.md`.
&nbsp;
## 1. Runs

Every training run is listed, including aborted and confounded ones. The first table gives identity, outcome, cost, and the commits that recorded the run. The second gives configuration. The third gives the quality check and the detector scores at the last check reached. The fourth gives the Opus check history. Run ids are shared across the tables. Dry runs and other no-training items follow the tables.

All runs use the policy `Qwen/Qwen3-0.6B-Base`, group size 4, and the frozen `qwen3-variable` detector as the training verifier. Rollouts are sampled at temperature 0.8 and top-p 0.9. The detector-only runs (E) use the stage-18 trainer with AdamW at 1e-5, no KL unless stated, and no ratio clip (`grpo-lab-note.md`). The judged runs through J2 use their own loop (`grpo-pilot-40.md`, `grpo-training-config.md`). The 200-word runs (L) use `max_new_tokens` 700, and the prompt asks for approximately 200 words with a length target of 203.

The commit hashes are the commits that added or changed the recording note. For the 200-word runs git holds a `run.log` per run for L4 to L9 and the shared `results/grpo-length-200/sweep.log` for L1 to L3. Their `steps.jsonl`, `eval-step-*.jsonl`, `summary.json`, and weights are not committed.

### Identity, outcome, and cost

| id | run | seed | steps reached | stop reason | cost | commits |
| --- | --- | ---: | --- | --- | --- | --- |
| E1 | short-target pilot, run A, bf16 (`grpo-human-baseline/run-A-bf16`) | 42 | 500 of 500 | none, finished | no API judge; about 1 h on one Quadro RTX 6000, 6.7 GB peak | `3429e0b`, `1fc6d34` |
| E2 | short-target pilot, run B, fp32 (`run-B-fp32`) | 42 | 500 of 500 | none, finished | no API judge; about 1 h, 12.2 GB peak | `3429e0b`, `1fc6d34` |
| E3 | seed replication, 6 runs: seeds 43, 44, 45 in bf16 and fp32 | 43, 44, 45 | 500 of 500 each | none, all `EXIT=0` | no API judge; two waves of about 55 min | `3429e0b` |
| E4 | `grpo-250w-fp32` | 42 | 100 of 100 | none, finished | n/a: GPU time not documented; 12.23 GB peak | `1cea54b`, `1fc6d34` |
| E5 | `grpo-250w-cap1616` | 42 | 60 of 100 | training process died saving the step-60 checkpoint: disk full | n/a: GPU time not documented | `ad5fa69`, `95b111f`, `1fc6d34` |
| E5b | `grpo-250w-cap1616/continuation`, steps 51 to 100 from the step-50 checkpoint | 42 | 100 of 100 | none, finished | n/a: GPU time not documented; 15.50 GB peak | `95b111f`, `1fc6d34` |
| E6 | `grpo-250w-trigram` | 42 | 60 of 60 | none, finished | n/a: GPU time not documented; 15.29 GB peak | `95b111f`, `1fc6d34` |
| E7 | `grpo-250w-kl`, KL beta 0.05 | 42 | 60 of 60 | none, finished | n/a: GPU time not documented; 17.50 GB peak | `4cb7923` |
| E8 | `grpo-250w-kl-beta0.5` | 42 | 60 of 60 | none, finished | n/a: GPU time not documented; 17.72 GB peak | `d5492aa` |
| J1 | 40-step pilot (`grpo-pilot-40`) | 42 in the note; `shuffle_seed` 20261008 in `summary.json` | 20 of 40 | Opus win rate under 50% at step 20 (7 of 18, 38.9%) | Opus checks $0.4936 + $0.5324; reward-judge cost n/a: not recorded, estimated $0.026 per step | `c287f3e`, `e06e8a0` |
| J2 | Sonnet JSON 40-step run (`grpo-sonnet-json`), the first run to finish 40 of 40 with no stop rule | 42 (`shuffle_seed` 20261008) | 40 of 40 | none, finished | reward judge $0.320332 over 112 calls (note); $5.34 with the Opus and Sonnet check calls, summed from `summary.json`, not stated in a note; `gpt-5.5` final call n/a: the note gives tokens only | `711d081`, `0e5bb0e`, `5c0540e` |
| L1 | length-matched sweep, T = 4 | 42 | 40 of 80 | Opus win rate under 40% (17 of 44, 38.6%) | $3.068520; 1 h 28 min | `b89057d` |
| L2 | length-matched sweep, T = 2 | 42 | 80 of 80 | none, finished | $8.565829; 3 h 32 min | `b89057d` |
| L3 | length-matched sweep, T = 1 | 42 | 40 of 80 | Opus win rate under 40% (18 of 46, 39.1%); step 20 was already under 45% | $2.992922; 1 h 37 min | `b89057d` |
| L4 | quality-gate reward, T = 1 | 42 | 40 of 80 | Opus under 45% at two checks in a row (19 of 47, 18 of 43) | $3.049356; 2 h 06 min | `7a81eca`, `c09e618` |
| L5 | slope reward | 42 | 80 of 80 | none, finished | $7.951919; about 3 h 15 min | `d9d5317`, `21c0276` |
| L6 | slope reward, seed 2 | 2 | 41 of 80 | median rollout length 362.5 words over the last 10 steps, limit 350 | $3.285286; about 2 h | `1c7e938` |
| L7 | rank reward | 42 | 40 of 80 | Opus win rate under 40% (13 of 48, 27.1%) | $2.691864; 1 h 35 min | `627fa38`, `1c7e938` |
| L8 | std floor, rank weight 0.1 | 42 | 37 of 80 | median rollout length 368.5 words over the last 10 steps, limit 350 | $1.760900; 1 h 05 min | `3aac04c`, `505230f` |
| L9 | std floor, rank weight 0.3 | 42 | 20 of 80 | Opus win rate under 40% (13 of 48, 27.1%) | $1.236216; 41 min | `3aac04c`, `505230f` |

Cost for L1 to L9 is the reward judge plus Opus plus `gpt-5.5`, from the saved `summary.json` of each run. Each figure equals the note's. Wall times come from the `START` and `END` lines of the run logs (`results/grpo-length-200/sweep.log` and each run's `run.log`) and agree with the notes where a note gives one. Cost excludes the 200-word pool build and the dry runs. L1 to L3 ran back to back on one log, and the pool was extended first (`EXTEND` lines, 1 h 53 min, cost n/a: not in the sweep totals).

The "first clean 40-step run" is not a phrase any note uses. J2 is the first run to finish 40 of 40 steps with no stop rule, and it is not length-matched. L1 is the first length-matched run. It reached step 40 only because a stop rule fired, and it did not skip zero-advantage steps. Both are listed so either reading can be found.

### Configuration

| id | reward | judge, reward and evaluation | length rule and gate | advantage normalization | skip zero-advantage steps |
| --- | --- | --- | --- | --- | --- |
| E1, E2, E3 | (1 - P_AI) * length_score, targets 50 and 100 words, `max_new_tokens` 256, no KL, no trigram term; E1 and E3 bf16 runs and E2 and E3 fp32 runs | none | length_score = min(w, target) / max(w, target) | (r - mean) / (std + 1e-4), population std, per group | yes, `--skip-zero-advantage-updates` in `run_bf16.sh` and `run_fp32.sh` |
| E4 | (1 - P_AI) * length_score, target 250, fp32, cap min(1616, ceil(250 * 1.6) + 16) = 416 tokens | none | same | same | yes (`run.sh`) |
| E5, E5b | same as E4 with the 1616-token budget; E5b has a fresh AdamW state | none | same | same | yes (`run.sh`) |
| E6 | P(human) * length_score * (unique trigrams / all trigrams), linear trigram, 1616 budget, fp32 | none | same | same | yes (`run.sh`) |
| E7, E8 | P(human) * length_score - beta * kl, beta 0.05 and 0.5, no trigram term, 1616 budget, fp32 | none | same | same | yes (`run.sh`) |
| J1 | hard * trigram_ratio^2 * sigmoid(z / 4) * length_score * wins / N, T = 4; a win is gpt-4.1-mini signal above 7; the frozen base answer is an opponent and counts in wins and N; bf16 (from `summary.json`, not in the note); grad clip 1.0; 1616 budget | reward `gpt-4.1-mini-2025-04-14`, both orders; evaluation `claude-opus-5-5` on `validation-00001` to `-00020`, both orders, agreed pairs | target 250; hard = 0 for under 30 words or a 1616-token hit | per-group z-score over the 4 rollouts only, as above | n/a: not documented; 0 of 20 steps had equal rewards, so the question did not arise |
| J2 | same product formula, T = 4, win = both Sonnet orders agree; 700-token budget; bf16; grad clip 1.0 | reward `claude-sonnet-5-5`, JSON schema; evaluation Opus on 50 validation prompts every 10 steps with Sonnet on the same texts, `gpt-5.5` at the end | target 250; hard = 0 under 150 words, over 375 words, under 30 words, or at 700 tokens | same | n/a: not documented; 24 of 40 steps had all-equal rewards (from `summary.json`), and whether those steps called the optimizer is not recorded |
| L1, L2, L3 | hard * trigram_ratio^2 * sigmoid(z / T) * length_score * wins / N, T = 4, 2, 1; wins count the other gated-in rollouts and the matched base | reward `claude-sonnet-5-5`, JSON schema, no tools, no temperature, both orders; evaluation Opus (steps 20, 40, 60 on 50 prompts; step 80 on 100 prompts with `gpt-5.5`), agreed pairs | target 203; hard = 0 under 100 words, over 300 words, under 30 words, or at 700 tokens | (r - mean) / (std + 1e-4) (stated for the quality-gate run and the repo loop; the sweep note does not restate it) | no; `optimizer.step()` ran every step |
| L4 | hard * q * trigram_ratio^2 * sigmoid(z / 1) * length_score; q = 1 if Sonnet does not prefer the matched base in either order, else 0 (strict reading) | same | same window 100 to 300 words | (r - mean) / (std + 1e-4) | no |
| L5, L6 | hard * length_score * trigram_ratio^2 * (q + 0.3 * sigmoid(z / 1)); q = 1, 0.5, 0 for a win in both orders, a split or tie or parse failure, a loss in both orders | same | target 203; hard = 0 only for under 30 words or a 700-token hit; no window | (r - mean) / (std + 1e-4), `unbiased=False` | yes, the repo criterion on the advantages (allclose, atol 1e-8) |
| L7 | hard * length_score * trigram_ratio^2 * (q + 0.3 * r); r = within-group rank of z scaled to [0, 1], ties share the average rank | same | same as L5 | same | yes, repo criterion |
| L8, L9 | hard * length_score^2 * trigram_ratio^2 * (q + w * r), w = 0.1 and 0.3 | same | length_score squared; hard as L5 | (r - mean) / max(std, 0.25), population std | yes, a reward range of at most 1e-4 skips backward, clip, and `optimizer.step()` |

For L5 to L9 the notes document AdamW at 1e-5, gradient clip 1.0, no KL, bf16, and gradient checkpointing (`grpo-slope-reward.md`, `grpo-std-floor-reward.md`). The sweep and quality-gate notes do not restate the optimizer, clip, or precision, so those are n/a for L1 to L4. The matched base is the gated-in pool sample, among 8 frozen base answers per prompt, closest in word count, with ties going to the lowest pool index. The ± 20% length band is 0.8 to 1.2 times the matched base's word count.

### Quality and detectors at the last check reached

Opus is `claude-opus-5-5`. Win rate is trained over trained plus base, agreed pairs only, both orders. The Qwen3 columns are sigmoid(z / T) with the run's own T for L1 and L2 (T = 4 and T = 2) and sigmoid(z / 1) for L3 to L9 and for J2 (J2 uses sigmoid(z / 4)). DistilBERT is P(human) from the held-out DistilBERT. Both detector columns are the trained answers against the length-matched base answers for the same prompts, except J1 and J2 (base answers of the saved pool) and E (see the notes).

| id | last check | Opus, all agreed pairs | Opus, within 20% length | `gpt-5.5`, all pairs and within 20% | Qwen3 sigmoid, trained vs base | DistilBERT P(human), trained vs base |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | step 500 | n/a: no judge | n/a | n/a | training verifier P(human) 0.9981 at the end; base n/a: not measured in the run | held-out 0.9800 (`summary.txt`); the held-out detector is not named in the notes |
| E2 | steps 451 to 500 | n/a | n/a | n/a | training verifier P(human) 0.9996, pinned from step 51; 105 collapsed steps | held-out 0.7564 |
| E3 | step 500 | n/a | n/a | n/a | final train P(human): s43 bf 0.9993, s43 fp 0.9991, s44 bf 0.8844, s44 fp 0.9994, s45 bf 0.9993, s45 fp 0.9985 | held-out: s43 bf 0.8576, s43 fp 0.6783, s44 bf 0.3964, s44 fp 0.3770, s45 bf 0.9870, s45 fp 0.9891 |
| E4 | step 100 | n/a | n/a | n/a | training verifier P(human) 0.9995 on the one "of the U.S." loop | 0.9958 on the loop |
| E5 | step 60 | n/a | n/a | n/a | 0.999047 | mean 0.997 on the four step-60 loops |
| E5b | step 100 | n/a | n/a | n/a | 0.9973 | 0.9993, 0.9995, 0.9989, 0.9991 |
| E6 | step 60 | n/a | n/a | n/a | 0.9983 | 0.9980, 0.9967, 0.9974, 0.9954 |
| E7 | step 60 | n/a | n/a | n/a | 0.9962 | stubs 0.9993, 0.9994, 0.9994 |
| E8 | step 60 | n/a | n/a | n/a | 0.99947 | n/a: the note says only that DistilBERT "still scores human", with no number |
| J1 | step 20 | 7/18 (38.9%) | n/a: not computed | n/a: not called | soft score sigmoid(z / 4) 0.4979 vs base 0.3447 | 0.7480 (the 4 training rollouts) vs base 0.3870 (20 saved base answers) |
| J2 | step 40 | 26/39 (66.7%) | 5/11 (45.5%) | 29/41 (70.7%); 7/14 (50.0%) | 0.1971 vs 0.2655 | 0.0924 vs 0.2160 |
| L1 | step 40 | 17/44 (38.6%) | 14/33 (42.4%) | n/a: not called | sigmoid(z / 4) 0.292563 vs 0.269687 | 0.206323 vs 0.173113 |
| L2 | step 80 | 49/90 (54.4%) | 37/72 (51.4%) | 50/83 (60.2%); 38/68 (55.9%) | sigmoid(z / 2) 0.199070 vs 0.236935; sigmoid(z / 1) 0.168 vs 0.210 | 0.173432 vs 0.285208 |
| L3 | step 40 | 18/46 (39.1%) | 10/31 (32.3%) | n/a: not called | 0.362741 vs 0.173465 | 0.626696 vs 0.232660 |
| L4 | step 40 | 18/43 (41.9%) | 2/3; n is too small to read | n/a: not called | 0.044962 vs 0.106397 | 0.083364 vs 0.174740 |
| L5 | step 80 | 49/83 (59.0%) | 36/63 (57.1%) | 51/80 (63.8%); 43/65 (66.2%) | 0.112886 vs 0.131564 | 0.150557 vs 0.235134 |
| L6 | step 40 | 27/45 (60.0%) | 22/39 (56.4%) | n/a: run stopped before step 80 | 0.061849 vs 0.147017 | 0.120923 vs 0.216925 |
| L7 | step 40 | 13/48 (27.1%) | 9/33 (27.3%) | n/a: run stopped before step 80 | 0.350149 vs 0.080392 | 0.400764 vs 0.104174 |
| L8 | step 20 | 21/41 (51.2%) | 18/33 (54.5%) | n/a: run stopped before step 80 | 0.121667 vs 0.188124 | 0.167181 vs 0.183812 |
| L9 | step 20 | 13/48 (27.1%) | 5/29 (17.2%) | n/a: run stopped before step 80 | 0.424078 vs 0.137102 | 0.560591 vs 0.147150 |

Reference values for the frozen base on the 250-word prompt, 80 answers: Qwen3 P(human) 0.133 and DistilBERT 0.223 (`grpo-base-detector-scores.md`). For L2 the note's T = 2 sigmoid is the first value, and the T = 1 sigmoid in `grpo-t-sweep-qualitative.md` is the second. The L1 sigmoid at T = 1 is n/a: no note gives it. The percentages are the notes' counts divided out. The notes give percentages for L5 to L9 and J1 and J2, and the other percentages are arithmetic on the stated counts. L4 had 47 of 50 and 48 of 50 trained answers over 300 words at its two checks, so its length band is nearly empty.

The step-80 Wilson 95% intervals the notes give are 0.48 to 0.69 for L5 Opus and 0.53 to 0.73 for L5 `gpt-5.5` (`grpo-slope-reward.md`), 0.36 to 0.66 for L8 at step 20 (`grpo-std-floor-reward.md`), and 0.17 to 0.41 for L7 at step 40 (`grpo-rank-reward.md`). No other check has an interval in a note.

### Opus check history

Agreed pairs, trained wins over agreed pairs, at each check the run reached. A dash means the run had stopped.

| id | step 10 | step 20 | step 30 | step 40 | step 60 | step 80 |
| --- | --- | --- | --- | --- | --- | --- |
| J1 | 11/20 | 7/18 | - | - | - | - |
| J2 | 20/42 | 23/47 | 23/46 | 26/39 | - | - |
| L1 | | 19/40 | | 17/44 | - | - |
| L2 | | 26/43 | | 23/42 | 22/41 | 49/90 |
| L3 | | 18/42 | | 18/46 | - | - |
| L4 | | 19/47 | | 18/43 | - | - |
| L5 | | 25/46 | | 22/39 | 28/45 | 49/83 |
| L6 | | 28/45 | | 27/45 | - | - |
| L7 | | 24/45 | | 13/48 | - | - |
| L8 | | 21/41 | | - | - | - |
| L9 | | 13/48 | - | - | - | - |

The J1 step-10 check is on 20 trained answers and the step-20 check on 20. J2 checked every 10 steps. The L runs checked at steps 20, 40, 60, and 80 only.

### Dry runs and other items with no training

| item | note | what it showed |
| --- | --- | --- |
| reward dry run, `gpt-4.1-mini`, win above 14 | `grpo-preflight.md` (`dffc2f5`) | 80 saved base answers; 15 of 20 groups had all-zero reward |
| tie threshold and temperature | `grpo-preflight-2.md` (`4d771c9`, `20c2fce`, `9b9bfaa`) | threshold set to 7; spread in 20 of 20 groups at T = 4 with trigram squared |
| Sonnet free-text dry run | `grpo-sonnet-qualify.md` (`8ef627a`) | spread in 7 of 20 groups; the 40-step run did not start |
| Sonnet forced-tool run | `grpo-sonnet-tool.md` (`d226745`) | HTTP 400; not started |
| 250-word length-matched dry run | `grpo-length-matched-dry-run.md` at `a149083`, superseded | spread 18 of 20, gate fires 63 of 160, $0.504396 |
| 200-word length-matched dry run | `grpo-length-matched-dry-run.md` (`031d9c4`) | spread 18 of 20, 0 parse failures, gate fires 47 of 160, $0.462480 |
| quality-gate dry run, T = 1 | `grpo-quality-gate-reward.md` (`7a81eca`) | spread 15 of 20 on the stored verdicts; the training run then had 33 of 40 zero-advantage steps |
| slope, rank, and std-floor dry runs | their notes | spread 20 of 20 on both sets in each |
| typography test | `grpo-typography-test.md` (`d9d5317`) | rescoring only |
| Lagrangian constraint | `grpo-sonnet-vs-opus-q.md` (`0c99c4a`) | not run, per the decision rule |

Not started, according to the notes: seed 2 of the T sweep (`grpo-length-200-t-sweep.md`), seed 2 of the quality-gate, rank, and std-floor runs, and the Sonnet free-text and forced-tool 40-step runs.
&nbsp;
## 2. Judge qualification

One row per judge or method. Gold or reference labels for each set are described below the table. "Both orders" means each pair is scored with the better side in slot A and again in slot B. Costs are API costs from the notes. A local model costs no API money.

| # | model | method | held-out set | reference bands or labels | position bias | cost | outcome | note |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `Qwen/Qwen2.5-7B-Instruct`, bf16 | generated letter, single order | 7B set, 10 pairs written so the better side was certain | gold fixed before the run; labeler not named | n/a: single order | local GPU | failed: missed the support-vector pair (pair 1); 3 of 3 padded or repetitive pairs and 7 of 9 others | `grpo-judge-qualification.md` |
| 2 | `Qwen/Qwen2.5-14B-Instruct-AWQ`; the 32B was not fetched, 17 GB free against a 25 GB bar | generated letter, single order | 14-pair set, pairs 1 to 14 | gold hash saved before the model loaded; labeler not named | n/a: single order | local GPU | failed: only miss was pair 1, the support-vector pair | `grpo-judge-32b.md` |
| 3 | 14B-AWQ | old prompt, generated letter, both orders | "old 30": 30 long fluent essays, one false sentence, matched word counts; plus the 527-word support-vector pair | gold is the corrected side; author not named | misses on pairs 2, 3, 15 were A in both orders | local GPU | 26 of 30; support-vector pair missed; reported as a comparison | `grpo-judge-length-matched.md` |
| 4 | 14B-AWQ | factual-correctness-first prompt, 192 tokens, both orders | old 30 | same | n/a: no bias figure | local GPU | failed: 17 of 30 against a bar of 30 of 30 | `grpo-judge-length-matched.md` |
| 5 | 14B-AWQ | method 1: single essay, "does this contain a false claim" | old 30 | same | n/a: one essay at a time | local GPU | 9 of 30; not chosen | `grpo-judge-methods.md` |
| 6 | 14B-AWQ | method 2: next-token log probability of `A` against `B`, both orders | old 30 for selection, then held-out 30 (122 to 144 words, one-sentence difference) | gold is the corrected side; hashes saved before scoring | the bias term exceeded the signal on 4 of the old 30, 5 of the held-out 30, 3 of 20 prose-vs-junk pairs; identical answers: slot A won all 10, margins 9.688 to 15.656 | local GPU | passed the held-out bar: 29 of 30 selection (miss pair 3), 30 of 30 held-out, smallest margin 1.477; not used as the reward judge | `grpo-judge-methods.md`, `grpo-judge-bias.md` |
| 7 | 14B-AWQ | method 3: per-side correct or incorrect labels | old 30 | same | n/a | local GPU | 22 of 30; not chosen | `grpo-judge-methods.md` |
| 8 | `gpt-5.4-nano-2026-03-17`, reasoning none | A/B log probabilities, both orders, pass needs signal of at least 3 | old 30, held-out 30, 20 prose-vs-junk | same, plus junk labels from `grpo-early-answer-labels.md` | per-pair bias in the note, not summarized | $0.017697 | failed: 7 of 30, 1 of 30, 6 of 20 | `grpo-openai-judge.md` |
| 9 | `gpt-5.4-mini-2026-03-17`, reasoning none | same | same | same | per-pair in the note | $0.066567 | failed: 30 of 30, 30 of 30, 15 of 20 | `grpo-openai-judge.md` |
| 10 | `gpt-4.1-mini-2025-04-14` | same | same | same | identical answers: slot A won all 10, margins 1.500 to 14.000 | $0.034114 | failed the OpenAI gate: 30 of 30, 30 of 30, 17 of 20; later accepted as the pilot reward judge on the reference bands (row 10b) | `grpo-openai-judge.md`, `grpo-junk-gate.md` |
| 10b | `gpt-4.1-mini-2025-04-14` | same, tie threshold 14, then 7 | 25 reference pairs; 22 counted (both-bad pairs 9, 13, 25 dropped) | Claude Opus labels reviewed by Eli, not human labels; bands: 12 High and Med-High, 6 Medium, 2 Low; 5 pairs outside the bands | as row 10 | n/a: per-call cost not stated; estimate $0.026 per step | accepted: 15 matches, 6 ties, 1 miss (pair 4); band bar met at both thresholds; threshold 7 used in J1 | `grpo-reference-labels.md`, `grpo-preflight.md`, `grpo-training-config.md` |
| 11 | `gpt-5.4-mini-2026-03-17`, reasoning low | written letter, Responses API, both orders | same 80 pairs | same | the 5 junk failures were the same letter in both orders | $0.089373 | failed: 30 of 30, 30 of 30, 15 of 20 | `grpo-openai-judge.md` |
| 12 | `claude-sonnet-5-5` | written letter, pairwise prompt, `max_tokens` 32, temperature omitted | held-out 30; 20 of the 25 reference pairs; 10 identical pairs | same | identical answers: no opening letter in 10 of 10, so no slot preference was readable | $0.216448 for 120 calls; dry run $0.181064 | qualification passed (held-out 30 of 30, no misses in High and Medium); dry run spread on 7 of 20 groups, so the run did not start | `grpo-sonnet-qualify.md` |
| 13 | `claude-sonnet-5-5` | forced tool call `select_winner` | none sent | n/a | n/a: no verdicts | n/a: one probe, cost not stated | not runnable: HTTP 400, `tool_choice` type `tool` unsupported for this model | `grpo-sonnet-tool.md` |
| 14 | `claude-sonnet-5-5` | JSON schema, `winner` in A, B, tie; no tools, no temperature; `max_tokens` 1024; both orders | held-out 30; 20 reference pairs; 10 identical pairs; dry run | same | identical answers: 20 of 20 calls returned `tie` | probe $0.000688; step 2 $0.246218; step 3 $0.209286; 200-word dry run $0.462480 | passed and adopted as the only reward judge: 10 of 10 ties, held-out 30 of 30, no misses in High and Medium (low pairs 2 and 4 missed, not counted), 0 parse failures, dry-run spread 15 of 20 | `grpo-sonnet-json.md`, `grpo-length-matched-dry-run.md` |
| 15 | `claude-opus-5-5`, evaluation judge, never in a reward | written A or B, both orders, agreed pairs | n/a: no held-out, reference-band, or identical-answer run for Opus appears in any note | n/a | n/a: not measured | rates $4 and $20 per million tokens; $5.367 for the 505 calls of L5 | never qualified; used unqualified as the evaluation judge | `grpo-pilot-40.md`, `grpo-training-config.md` |
| 16 | `gpt-5.5`, returned id `gpt-5.5-2026-04-23`, reasoning low | written letter, Responses API, both orders, agreed pairs | n/a: no qualification run in any note | n/a | n/a | $1.168 for 201 calls in L5; J2 final call 63,127 input and 9,688 output tokens | never qualified; used as a second evaluation judge only (J2 step 40, L2 and L5 step 80) | `grpo-sonnet-json.md`, `grpo-length-200-t-sweep.md`, `grpo-slope-reward.md` |

Sets. "Old 30" and "held-out 30" are constructed factual pairs with one false sentence, gold SHA hashed before scoring (`grpo-judge-length-matched.md`, `grpo-judge-methods.md`). The 20 prose-vs-junk pairs are two rollouts from one prompt group, with the prose side as gold, labeled per `grpo-early-answer-labels.md`; that note does not name the labeler. The 10 identical-answer pairs have the same base answer in both slots and expect a tie. The 25 reference pairs are in `grpo-prose-pairs.pdf` and `grpo-reference-labels.md`: pairs 1 to 20 are prose against prose and pairs 21 to 25 are trigram-run template essays against a normal answer. The labeling Opus snapshot, the assignment of bands, and the reason for the five out-of-band pairs are not stated in any note.

The gate for the reference pairs is no misses on High and Med-High, at most one miss on Medium, and Low misses reported only (`grpo-preflight.md`). A miss is both orders naming the wrong side. A disagreement or a missing letter is not a miss.
&nbsp;
## 3. Confounds and caveats

### a. Runs that did not skip zero-advantage steps

L1 to L4 called `optimizer.step()` on every step. On a step whose advantages were all zero the gradient was zero, but AdamW still applied its stored momentum and weight decay, so the weights moved (`grpo-slope-reward.md`, optimizer check: a maximum weight change of 3.1e-6 in the check, against none on the repo skip path). The count of zero-advantage steps by the repo criterion, advantages within 1e-8 of zero, is below. I recounted these from each run's `steps.jsonl`, and the counts match the notes.

| run | steps | zero-advantage steps | what the note says |
| --- | ---: | ---: | --- |
| L1, T = 4 | 40 | 2 | small, effect cannot be measured from the logs |
| L2, T = 2 | 80 | 4 | same |
| L3, T = 1 | 40 | 4 | same |
| L4, quality gate | 40 | 33 | large: 28 steps had no judged rollout and 5 had equal rewards, so most weight changes after the first steps came from momentum; the drift to answers over 300 words cannot be separated from this |
| L5 to L9 | | 0 for L5, L6, L8, L9; 1 for L7 | these runs skip, so the one L7 step made no update |

J1 had 0 of 20 steps with equal rewards. J2 had 24 of 40 steps with all-equal rewards in `summary.json`, and its note does not say whether those steps called the optimizer. The J2 case is not covered by the confound note, which names only the sweep and the quality-gate run.

### b. One seed per setting

Every setting ran once with seed 42, except slope, which also ran with seed 2. The seed-2 run stopped early on length, so there is no step-80 check for it. The T sweep, quality-gate, rank, and both std-floor runs have no second seed. The two slope seeds agreed on quality at steps 20 and 40 within the noise of 45 pairs (seed 42: 54.3% and 56.4%; seed 2: 62.2% and 60.0%) and disagreed on length (`grpo-slope-seed2.md`). The std-floor note says one seed per weight cannot rule out run noise in how fast w = 0.3 collapsed.

### c. Same model family, and whether the training judge tracked the evaluation judge

The reward judge is Sonnet and the evaluation judge is Opus. Both are Claude models. The reference labels used to qualify the Sonnet judge are Claude Opus labels. `gpt-5.5` is the only judge outside that family, and it was run only at step 80 of L2 (50 of 83 against Opus 49 of 90) and L5 (51 of 80 against Opus 49 of 83) and at step 40 of J2 (29 of 41 against Opus 26 of 39).

The log analysis in `grpo-sonnet-vs-opus-q.md` compares Sonnet's training q with the Opus checks. Across 10 checks the Spearman correlation is 0.92 and the Pearson 0.82, driven largely by the difference between runs that hold quality and runs that do not. Within runs it is mixed. In L9 Opus fell to 27% at step 20 and Sonnet's q stayed at 0.49, with a decided win rate of 0.48 (16 against 17, interval 0.33 to 0.65). In L7 Sonnet's decided win rate fell from 0.50 to 0.37 as Opus fell from 53% to 27%, with an interval at step 40 of 0.23 to 0.54. In L6 at step 40 Sonnet's decided win rate was 0.77 against Opus 60%. The note's decision was that the judge did not clearly track the fastest collapse, so the adaptive constraint was not run. The two judges see different texts (training rollouts against validation answers), so the comparison is in level and direction, not pair by pair.

### d. Other caveats already documented

- Length matching fails. A rollout longer than all 8 pool answers is matched to the longest gated-in pool answer, which is shorter, so the judge compares a longer answer with a shorter one. The share of such rollouts is 0.31 for L6 (51 of 164), 0.22 for L8 (32 of 148), and 0.11 for L5 (34 of 320), with 0.30 in the last block of L5. Their mean q against the others is 0.75 against 0.60 for L6, 0.39 against 0.53 for L8, 0.67 against 0.59 for L5, and 0.32 against 0.48 for L7 (14 rollouts). The effect on q has no consistent sign (`grpo-sonnet-vs-opus-q.md`).
- Detector layout sensitivity. The Qwen3 detector reads line breaks and list layout as a signal. Flattening whitespace changes its trained mean from 0.374 to 0.741 and the base from 0.176 to 0.311. DistilBERT ignores whitespace. The reward uses z with no flattening (`grpo-typography-test.md`). The two detectors disagree on single texts, with a correlation of 0.45 on the 50 T = 1 trained answers against 0.68 for the base (`grpo-t-sweep-qualitative.md`).
- Small n. A check has 39 to 48 agreed pairs at 50 prompts, so a 95% interval is about 15 points either side (0.36 to 0.66 at 41 pairs, 0.17 to 0.41 at 48). The within-20% band has 3 to 41 pairs.
- Stop-rule noise. The J1 pilot stopped at step 20 on 7 of 18 agreed pairs (38.9%) after 11 of 20 at step 10, with no interval in its note. L3 was recorded as stopped on under 40% at 18 of 46 (39.1%), where one more trained win would have given 41.3%.
- Reference labels are model labels. They are Claude Opus labels that Eli reviewed as matching his judgment, not human labels (`grpo-reference-labels.md`).
- Opus and `gpt-5.5` were never qualified (section 2, rows 15 and 16).
- J2 headline versus length band. J2's Opus 26 of 39 is on all pairs. On the 18 pairs within 1.2 times the base's length Opus is 5 of 11, and both detector means were below the base's (`grpo-step40-length.md`). Its trained answers at steps 10 to 30 averaged 117 to 136 words (`grpo-sonnet-json.md`), and 74.4% of its rollouts were hard-zeroed for length.
- Pool and dry-run costs are not in the run totals.

### Inconsistencies between notes

None changes a result. They are listed so the notes can be read together.

1. `grpo-pilot-40.md` says the training prompts were shuffled with seed 42. `grpo-pilot-40/summary.json` and the J2 summary record `shuffle_seed` 20261008.
2. `grpo-pilot-40.md` does not state the precision. `summary.json` says bfloat16, and all 250-word E runs are float32.
3. The "held-out about 0.3" figure for the short pilot is not tied to a detector name. `analyze_run.py` defaults to the logistic-regression detector, so the figure may be that detector and not DistilBERT. That is an inference. `grpo-lab-note.md` and the README put it next to DistilBERT text.
4. `grpo-lab-note.md` and the human-baseline README cite only bf16 seed 44 as the held-out drop (0.3964), but fp32 seed 44 also ends at 0.3770 (`seed-replication/summary.txt`).
5. `grpo-250w-kl/README.md` says the 1616-token float32 run peaked at 15.50 GB. That figure is the continuation's `run.log` value. The cap1616 parent run died without a peak line.
6. `grpo-training-config.md` reports J2's final Opus win rate as 26 of 39 and does not mention the 5 of 11 within the length band or that both detectors fell below the base. `grpo-step40-length.md` has both.
7. `grpo-step40-length.md` has a closing "Next experiment" section that proposes T = 4, 2, 1 with two seeds each, 6 runs, and a 100-word floor. The sweep that ran used one seed and a 100 to 300 word window, and seed 2 did not start (`grpo-length-200-t-sweep.md`). The section is stale.
8. `grpo-training-config.md` says 28 of 40 quality-gate steps "had no judged rollout and no gradient". `grpo-slope-reward.md` corrects the count: 33 of 40 steps had zero advantages and the optimizer still stepped on them.
9. The `sigmoid` column means different things across notes. In `grpo-length-200-t-sweep.md` it is sigmoid(z / T) for the run's T, so L1 (T = 4) and L2 (T = 2) are not comparable with L3 to L9, which are at T = 1. `grpo-t-sweep-qualitative.md` gives L2 at step 80 as 0.168 against 0.210 at T = 1, and 0.199 against 0.237 at T = 2.
10. `grpo-lab-note.md` covers the beta 0.05 KL run only. The beta 0.5 run is in `grpo-250w-kl-beta0.5/README.md`.
11. `grpo-length-matched-dry-run.md` now describes the 200-word dry run, while the directory `results/grpo-length-matched/` holds the 250-word artifacts and `results/grpo-length-200/` the 200-word ones. The 250-word figures in this index come from commit `a149083`.
12. `grpo-quality-gate-reward.md` says T = 2 and T = 4 "kept quality and did not become more human". L1 (T = 4) was stopped on the Opus under-40% rule at 17 of 44, and its two detector means at step 40 were slightly above its base (`grpo-length-200-t-sweep.md`). Only L2 kept quality through step 80.

The saved `summary.json` costs for L1 to L9 and the costs in the notes agree to the cent, and the zero-advantage counts match the notes.
&nbsp;
## 4. Notes under `results/`

Links are relative to `results/`. Where files sit in a folder the folder's own README is the entry.

### Detector-only GRPO runs (E1 to E8)

- [grpo-human-baseline/README.md](grpo-human-baseline/README.md): the short-target pilot, runs A and B, and what the text turns into.
- [grpo-human-baseline/seed-replication/PREREGISTRATION.md](grpo-human-baseline/seed-replication/PREREGISTRATION.md): the rule fixed before the six seed runs.
- [grpo-human-baseline/seed-replication/RESULT.md](grpo-human-baseline/seed-replication/RESULT.md): collapsed steps per run, float32 4 of 4 against bfloat16 0 of 4.
- [grpo-human-baseline/run-A-bf16/phase2-table.md](grpo-human-baseline/run-A-bf16/phase2-table.md) and [run-B-fp32](grpo-human-baseline/run-B-fp32/phase2-table.md): 50-step block tables for runs A and B.
- grpo-human-baseline/seed-replication phase-2 tables: [seed43-bfloat16](grpo-human-baseline/seed-replication/seed43-bfloat16/phase2-table.md), [seed43-float32](grpo-human-baseline/seed-replication/seed43-float32/phase2-table.md), [seed44-bfloat16](grpo-human-baseline/seed-replication/seed44-bfloat16/phase2-table.md), [seed44-float32](grpo-human-baseline/seed-replication/seed44-float32/phase2-table.md), [seed45-bfloat16](grpo-human-baseline/seed-replication/seed45-bfloat16/phase2-table.md), [seed45-float32](grpo-human-baseline/seed-replication/seed45-float32/phase2-table.md): block tables for each replicate.
- [grpo-human-baseline/blind-taxonomy/blind-sample.md](grpo-human-baseline/blind-taxonomy/blind-sample.md): a blind sample of rollouts to label with failure categories.
- [grpo-250w-fp32/README.md](grpo-250w-fp32/README.md): the 250-word float32 run, 416-token cap, ending in an "of the U.S." loop.
- [grpo-250w-cap1616/README.md](grpo-250w-cap1616/README.md) and [continuation](grpo-250w-cap1616/continuation/README.md): the 1616-token budget run through step 60, and steps 51 to 100.
- [grpo-250w-trigram/README.md](grpo-250w-trigram/README.md): the linear trigram repetition factor.
- [grpo-250w-kl/README.md](grpo-250w-kl/README.md) and [grpo-250w-kl-beta0.5/README.md](grpo-250w-kl-beta0.5/README.md): the KL runs at beta 0.05 and 0.5.

### Diagnostics on those runs (no training)

- [grpo-lab-note.md](grpo-lab-note.md): the reward, the update, and logged samples of each early run.
- [grpo-base-loops.md](grpo-base-loops.md): the frozen base does not loop, but scores policy loops as likely.
- [grpo-base-detector-scores.md](grpo-base-detector-scores.md): frozen-base answers under both detectors (mean P(human) 0.133 and 0.223).
- [grpo-step0-table.md](grpo-step0-table.md): step-0 scores for base answers and human texts.
- [grpo-length-matched-human.md](grpo-length-matched-human.md): length-matched scores for the step-0 human texts.
- [grpo-early-answer-labels.md](grpo-early-answer-labels.md): the highest-reward early answers labeled prose or junk.
- [grpo-kl-entry-spike.md](grpo-kl-entry-spike.md), [grpo-kl-winner-entry.md](grpo-kl-winner-entry.md), [grpo-kl-token-ratio.md](grpo-kl-token-ratio.md), [grpo-kl-token-ratio-policy.md](grpo-kl-token-ratio-policy.md): per-token log-ratio figures on the KL runs' loops and on policy text.

### Junk gate

- [grpo-junk-gate.md](grpo-junk-gate.md): the frozen rule gate and the identical-answer calibration for `gpt-4.1-mini`.
- [grpo-junk-gate-lists.md](grpo-junk-gate-lists.md): good parallel lists tested against the frozen gate.
- [grpo-junk-pairs-1-5-15.md](grpo-junk-pairs-1-5-15.md): full text of three prose-vs-junk pairs.

### Judge qualification

- [grpo-judge-qualification.md](grpo-judge-qualification.md): Qwen2.5-7B-Instruct, failed.
- [grpo-judge-32b.md](grpo-judge-32b.md): the 14-pair set on Qwen2.5-14B-AWQ, failed; 32B not fetched.
- [grpo-judge-length-matched.md](grpo-judge-length-matched.md): the length-matched factual test of the 14B model.
- [grpo-judge-methods.md](grpo-judge-methods.md): three factual-check methods, method 2 passes the held-out 30.
- [grpo-judge-bias.md](grpo-judge-bias.md): position bias of method 2, identical answers, prose against junk.
- [grpo-openai-judge.md](grpo-openai-judge.md): four OpenAI runs, none passed the gate.
- [grpo-reference-labels.md](grpo-reference-labels.md): Claude Opus reference labels for 25 pairs, reviewed by Eli.
- [grpo-prose-prose-to-label.md](grpo-prose-prose-to-label.md): the 20 unlabeled prose-vs-prose pairs.
- [grpo-prose-pairs.pdf](grpo-prose-pairs.pdf): the 25 pairs as built for labeling.
- [grpo-sonnet-qualify.md](grpo-sonnet-qualify.md): Sonnet free-text letter, qualified, dry run spread 7 of 20.
- [grpo-sonnet-tool.md](grpo-sonnet-tool.md): forced tool call rejected by the API.
- [grpo-sonnet-json.md](grpo-sonnet-json.md): the JSON-schema judge, its qualification, and the 40-step run J2.

### Pilots and preflight

- [grpo-preflight.md](grpo-preflight.md): reward dry run on 80 base answers, confidence bands, stop rules.
- [grpo-preflight-2.md](grpo-preflight-2.md): tie threshold 7 and the temperature that spreads the rewards.
- [grpo-training-config.md](grpo-training-config.md): the reward, gate, judge, and monitoring config, updated through the std-floor runs.
- [grpo-pilot-40.md](grpo-pilot-40.md): the 40-step pilot J1, stopped at step 20.
- [grpo-pilot-20-read.md](grpo-pilot-20-read.md): the step 0, 10, and 20 read.
- [grpo-pilot-pairs.md](grpo-pilot-pairs.md): pilot pairs for hand reading.
- [grpo-pilot-sonnet-pairs.md](grpo-pilot-sonnet-pairs.md): 20 shuffled unlabeled step-40 pairs from J2.
- [grpo-step40-length.md](grpo-step40-length.md): J2 at step 40 by length bin, with both detectors.

### Length-matched series (L1 to L9)

- [grpo-length-matched-dry-run.md](grpo-length-matched-dry-run.md): the 200-word dry run that cleared the bars.
- [grpo-length-200-t-sweep.md](grpo-length-200-t-sweep.md): T = 4, 2, 1.
- [grpo-t-sweep-qualitative.md](grpo-t-sweep-qualitative.md): a read of the answers behind T = 1 step 40 and T = 2 step 80.
- [grpo-typography-test.md](grpo-typography-test.md): the typography and layout test.
- [grpo-quality-gate-reward.md](grpo-quality-gate-reward.md): the quality-gate reward, dry run, and run.
- [grpo-slope-reward.md](grpo-slope-reward.md): the optimizer check, the slope reward, its dry run, and the seed-42 run.
- [grpo-slope-seed2.md](grpo-slope-seed2.md): slope with seed 2.
- [grpo-rank-reward.md](grpo-rank-reward.md): the rank reward and run.
- [grpo-std-floor-reward.md](grpo-std-floor-reward.md): the std floor, squared length, and the w = 0.1 and 0.3 runs.
- [grpo-sonnet-vs-opus-q.md](grpo-sonnet-vs-opus-q.md): Sonnet q against Opus checks, length-matching failures, and the Lagrangian decision.
- [grpo-results-index.md](grpo-results-index.md): this file.

Figures: [kl-entry-spike.png](kl-entry-spike.png), [kl-winner-entry.png](kl-winner-entry.png), [kl-token-ratio.png](kl-token-ratio.png), [kl-token-ratio-policy.png](kl-token-ratio-policy.png), and [grpo-human-baseline/panels.png](grpo-human-baseline/panels.png).

### Weights, logs, and keys

Weights are on the lab machine and are not committed. Each 200-word run has a sibling directory `results/grpo-length-200-<name>-weights`, the earlier judged runs have `grpo-pilot-40-weights` and `grpo-sonnet-json-weights`, and the E runs keep checkpoints under their own `checkpoints` directories. The repository ignores `*.safetensors`, `*.pt`, `checkpoints*/`, and `weights/`. Also not committed are `steps.jsonl`, `eval-step-*.jsonl`, `summary.json`, the judge call cache, and the pool and rollout `jsonl` files. Committed per run are `run.log` and the notes.

API keys are read at run time from a local environment file outside the repository. They are never written to a note, a log, or a commit, and `.env` is in `.gitignore`. A note that mentions a key (`grpo-openai-judge.md`) says only that it was read from a local file and is not written there.
