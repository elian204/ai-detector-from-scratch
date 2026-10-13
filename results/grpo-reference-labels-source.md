&nbsp;
# Reference labels, source

Source: the labels and remarks written by Claude Opus in the Claude app, pasted by Eli Bogdanov on 2026-10-13. The model version is not recorded. The text below the line is unedited. The pair letters are the PDF slots of `results/grpo-prose-pairs.pdf`.

---

## Labels

| Pair | Pick | Confidence | Main reason |
|---|---|---|---|
| 1 | **B** | Medium | Structured, more substantive factor-by-factor treatment. A has dubious claims (slavery → human-rights emphasis) and a repeated phrase ("interact in practice"). B handles the "interaction" part less explicitly and is cut off at the end. |
| 2 | **B** | Low–Med | Evidence, rigor and replication are the right core criteria. B has a factual error: Hardy-Weinberg assumes *no* mutation. A is vaguer and uses odd criteria, such as "impact on future research" as a test of truth. |
| 3 | **A** | High | Complete and coherent. B has "uncertainties or uncertainties" and ends with a leaked follow-up question. |
| 4 | **A** | Low (close) | A is thin but error-free. B is richer but has errors: scatter plots for qualitative data, a false qualitative/quantitative dichotomy, and arbitrary method-to-context mappings. |
| 5 | **A** | Medium | A is confused (it says "the president should explain" and includes scandals and personal life) but lists the relevant elements. B invents a professor and a source, and says the president can declare war (that power belongs to Congress). |
| 6 | **A** | High | B is badly wrong: Founding Fathers leading 19th-century reform, Franklin's autobiography framed as anti-slavery literature, a fake source. A is on-topic, though it drifts into self-generated Q&A. |
| 7 | **B** | Med–High | B's history of scholarship and its open questions (preservation, misrepresentation, appropriation) are relevant, despite a verbatim self-repeat. A is vacuous ("current approaches are modern storytelling") and cites a fake source. |
| 8 | **B** | High | B is about *prediction* (ML, behavior, contact networks). A drifts to vaccine efficacy, mischaracterizes 19th-century epidemiology, and repeats itself. |
| 9 | **B** | Medium | B includes scope and application and has no exact duplicates. A's item 9 repeats item 3. Neither covers enforcement or compliance. |
| 10 | **A** | High | A names one real misconception (development as a linear process) with a sensible fix. B loops on circular filler. |
| 11 | **A** | High | B is a degenerate repetition loop. |
| 12 | **B** | Medium | B lists three actual (if shallow) misconceptions. A is incoherent ("for the business, not the people") with leaked Q&A. |
| 13 | **B** | Low–Med | B at least follows the background / approaches / open-questions structure, despite odd claims (UN, gig economy). A turns the question into a fake multiple-choice item. |
| 14 | **B** | Low (both bad) | Both miss the likely intent. A is tautological ("level of security required affects level of security required"). B is empty but not wrong. |
| 15 | **A** | Med–High | A names real mechanisms (epigenetics, neuroendocrine) with correct definitions. B is circular and treats "education" as a factor in an organism's reproductive success. |
| 16 | **A** | Low (both bad) | A is off-topic (New Deal) and wrong about the NSA. B is catastrophically false: the US "invaded the Soviet Union in 1941" and the Cold War "ended in 1945". |
| 17 | **B** | High | B actually answers the question (question–design fit, controls, placebo, blinding) before degenerating. A answers a different question and is full of false Framingham facts (wrong lead investigator, "led to aspirin"). |
| 18 | **A** | Low (both bad) | A is a fabricated, self-contradictory "Kosel method" but stays in the right domain. B has wrong-domain, absurd Fed history (Fed "created in the 1930s", FDR selling his portfolio). |
| 19 | **B** | High | B gives a coherent reading (factorial design, 2×2) that is mostly correct. A is incoherent numeric babble. |
| 20 | **B** | Med–High | B is on-topic (age structure, temporal comparison) though speculative. A misdefines "county" and loops. |
| 21 | **B** | Low–Med | B gives the relevant colonial and language-loss context. Its "held accountable for their actions" line is badly framed. A is circular filler. |
| 22 | **B** | High | After the "519" junk, B is coherent and relevant. A is a fabricated multiple-choice mess. |
| 23 | **A** | Medium | A is bland but consistent. B contradicts itself: the misconception is "designed based on user feedback", and the correction is to "encourage over-engineering". |
| 24 | **A** | Medium | A mentions techniques and challenges. B is one sentence template repeated 10×, with no principles or challenges. |
| 25 | **B** | High | Both carry leaked MCQ prompt text. B's criteria (accuracy, context, limitations, comparison with other studies) fit the question. A ends with "correct answer: A. Positive". |

**Copy-paste:** `B B A A A A B B B A A B B B A A B A B B B B A A B` (11 A / 14 B)

## Things worth knowing about the set

- **Duplicate questions:** 7/21, 9/22, 10/23, 11/24 and 20/25 repeat prompts. Pairs 11 and 24 also share an identical A response, and I picked A both times, so the labels are consistent there.
- **Both-unacceptable pairs:** 14, 16 and 18, and arguably 13 and 21. Here a forced choice is mostly "less wrong" and carries little signal. If your setup allows ties or a "both bad" flag, these are the candidates. If not, consider down-weighting them or excluding them from agreement metrics.
- **Rule I applied consistently:** relevance and absence of fabrication beat polish and length. Pairs 4 and 23 are where that rule decides against the longer or more detailed answer, so they are the ones most likely to disagree with a length-biased judge or reward model.
