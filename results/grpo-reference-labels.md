&nbsp;
# Reference labels

No GRPO and no training. These are Claude Opus reference labels for the 25 pairs in `results/grpo-prose-pairs.pdf`, in pair order 1 to 25. Eli Bogdanov reviewed them as matching his judgment. They are not human labels.

The letter is the PDF slot. The A/B assignment used to build the PDF is not in this repository.

B B A A A A B B B A A B B B A A B A B B B B A A B
&nbsp;
## Scores

The judge is `gpt-4.1-mini-2025-04-14` only. The method is the A/B probability method in `results/grpo-openai-judge.md`. Temperature is 0. Each pair is scored in both orders. m1 is the margin with the reference side in slot B. m2 is the margin with the reference side in slot A. The margin is log P(reference slot) minus log P(other slot). signal = (m1 + m2) / 2.

The tie threshold is 14. A match is signal greater than 14. A tie is absolute signal at most 14. A miss is signal less than −14. Pairs 21–25 stay in this table as template pairs. Their reference letters were not changed.

The response `model` field was `gpt-4.1-mini-2025-04-14` on every call. Of the 50 calls, 31 had both letters in the top list and 19 used the chosen-only fallback. Fourteen of those fallback calls were clamped, the same floor as in `results/grpo-openai-judge.md`.

Three pairs are both bad, and they are left out of the counts. Pair 9 is two ten-item lists of the same regulations frame, "the role of the regulations" and "the importance of the regulations", with no distinct content. Pair 13 is a fabricated multiple-choice item against an answer that opens with an empty quote and then repeats one settlement frame. Pair 25 is two multiple-choice shells followed by the same "this will help in determining" frame. The letters for those three pairs are still in the table.

On the other 22 pairs the judge has 15 matches, 6 ties, and 1 miss. Pairs 1–20, after dropping 9 and 13, are 13 matches, 4 ties, and 1 miss. The template pairs that remain, 21–24, are 2 matches, 2 ties, and 0 misses. The miss is pair 4.

| pair | set | reference | m1 | m2 | signal | result | in the count |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | prose | B | 20.500 | 21.250 | 20.875 | match | yes |
| 2 | prose | B | 6.500 | 27.631 | 17.066 | match | yes |
| 3 | prose | A | 10.750 | 5.250 | 8.000 | tie | yes |
| 4 | prose | A | -20.250 | -27.631 | -23.941 | miss | yes |
| 5 | prose | A | -8.250 | 11.250 | 1.500 | tie | yes |
| 6 | prose | A | 24.750 | 8.746 | 16.748 | match | yes |
| 7 | prose | B | 27.631 | 19.750 | 23.691 | match | yes |
| 8 | prose | B | 27.631 | 27.631 | 27.631 | match | yes |
| 9 | prose | B | 26.250 | 4.500 | 15.375 | match | no |
| 10 | prose | A | 13.678 | 15.457 | 14.568 | match | yes |
| 11 | prose | A | 27.631 | 13.331 | 20.481 | match | yes |
| 12 | prose | B | 27.631 | 18.250 | 22.941 | match | yes |
| 13 | prose | B | 5.250 | 14.500 | 9.875 | tie | no |
| 14 | prose | B | 10.750 | -27.631 | -8.441 | tie | yes |
| 15 | prose | A | 27.631 | 20.250 | 23.941 | match | yes |
| 16 | prose | A | 8.750 | 18.250 | 13.500 | tie | yes |
| 17 | prose | B | 27.631 | 18.250 | 22.941 | match | yes |
| 18 | prose | A | 21.375 | 17.000 | 19.188 | match | yes |
| 19 | prose | B | 27.631 | 27.631 | 27.631 | match | yes |
| 20 | prose | B | 19.000 | 11.250 | 15.125 | match | yes |
| 21 | template | B | 10.250 | 14.500 | 12.375 | tie | yes |
| 22 | template | B | 25.375 | 27.631 | 26.503 | match | yes |
| 23 | template | A | 7.500 | 6.250 | 6.875 | tie | yes |
| 24 | template | A | 27.631 | 9.750 | 18.691 | match | yes |
| 25 | template | B | 9.000 | 6.739 | 7.869 | tie | no |
&nbsp;
## Gate on the 50 answers

The rules are the frozen rules in `results/grpo-junk-gate.md`. They were not retuned. A cap hit is the rollout field `hit_token_cap`, so a completion that stopped at the 416 limit of its own log counts. These answers are rollouts, and the cap rule is applied. The training budget of 1616 is a separate decision and does not change this run of the rules.

33 of the 50 answers fire. Both sides fire on pairs 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 25.

| pair | side | words | 6-gram count | trigram ratio | rules |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | A | 228 | 1 | 0.9823 | none |
| 1 | B | 339 | 1 | 0.9674 | cap |
| 2 | A | 204 | 1 | 0.9752 | none |
| 2 | B | 331 | 1 | 0.9970 | cap |
| 3 | A | 299 | 2 | 0.9697 | none |
| 3 | B | 313 | 2 | 0.9260 | none |
| 4 | A | 123 | 1 | 0.8678 | none |
| 4 | B | 301 | 3 | 0.7993 | none |
| 5 | A | 197 | 3 | 0.7897 | none |
| 5 | B | 157 | 2 | 0.9226 | none |
| 6 | A | 346 | 4 | 0.7878 | 6-gram 4; cap |
| 6 | B | 229 | 1 | 0.9515 | none |
| 7 | A | 167 | 2 | 0.8727 | none |
| 7 | B | 344 | 3 | 0.5614 | trigram 0.561; cap |
| 8 | A | 368 | 3 | 0.6339 | trigram 0.634; cap |
| 8 | B | 370 | 3 | 0.7717 | cap |
| 9 | A | 104 | 3 | 0.6961 | trigram 0.696 |
| 9 | B | 104 | 4 | 0.6765 | 6-gram 4; trigram 0.676 |
| 10 | A | 185 | 2 | 0.6776 | trigram 0.678 |
| 10 | B | 379 | 3 | 0.5411 | trigram 0.541; cap |
| 11 | A | 138 | 1 | 0.8456 | none |
| 11 | B | 289 | 9 | 0.4460 | 6-gram 9; trigram 0.446; sentence 5 |
| 12 | A | 268 | 4 | 0.6654 | 6-gram 4; trigram 0.665 |
| 12 | B | 166 | 3 | 0.7134 | sentence 3 |
| 13 | A | 201 | 6 | 0.5678 | 6-gram 6; trigram 0.568 |
| 13 | B | 263 | 5 | 0.6973 | 6-gram 5; trigram 0.697 |
| 14 | A | 272 | 5 | 0.5556 | 6-gram 5; trigram 0.556 |
| 14 | B | 145 | 2 | 0.8042 | none |
| 15 | A | 175 | 5 | 0.5838 | 6-gram 5; trigram 0.584 |
| 15 | B | 155 | 4 | 0.6013 | 6-gram 4; trigram 0.601 |
| 16 | A | 323 | 3 | 0.7632 | cap |
| 16 | B | 340 | 3 | 0.8107 | cap |
| 17 | A | 344 | 3 | 0.8567 | cap |
| 17 | B | 393 | 6 | 0.5985 | 6-gram 6; trigram 0.598; cap |
| 18 | A | 366 | 4 | 0.6456 | 6-gram 4; trigram 0.646; cap |
| 18 | B | 368 | 3 | 0.8005 | cap |
| 19 | A | 338 | 3 | 0.7232 | cap |
| 19 | B | 381 | 2 | 0.8047 | cap |
| 20 | A | 384 | 4 | 0.6283 | 6-gram 4; trigram 0.628; cap |
| 20 | B | 318 | 7 | 0.7215 | 6-gram 7; cap |
| 21 | A | 288 | 3 | 0.7273 | none |
| 21 | B | 135 | 2 | 0.8496 | none |
| 22 | A | 247 | 5 | 0.7347 | 6-gram 5; "the answer is" 5 |
| 22 | B | 172 | 3 | 0.8059 | none |
| 23 | A | 194 | 5 | 0.5365 | 6-gram 5; trigram 0.536 |
| 23 | B | 170 | 3 | 0.7262 | none |
| 24 | A | 138 | 1 | 0.8456 | none |
| 24 | B | 145 | 10 | 0.4895 | 6-gram 10; trigram 0.490 |
| 25 | A | 262 | 4 | 0.7500 | 6-gram 4 |
| 25 | B | 236 | 4 | 0.7906 | 6-gram 4 |
