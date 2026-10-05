&nbsp;
# Early highest-reward answers, prose or junk

No training. These rows are labels of completions already in the rollout logs. The 80 base answers were not rescored, and no judge model was called.

Each row is the single highest logged training reward at that step. The reward is the trainer's `reward` field: P(human) times the length score, times the trigram score on the trigram run, minus 0.05 times KL on the KL run. A star on the rollout index means all four rollouts have reward 0, so there is no unique winner. The row is the lowest rollout index.

Junk means a repetition loop, a tag loop, digit or punctuation salad, a spaceless token tail, or padding that is not an answer. Prose means the completion reads as an attempt to answer, even where a fact is wrong. None of these four logs stops before step 20. The 416-token log has 100 steps, and the other three have 60. The 1616 rows are from `results/grpo-250w-cap1616/rollouts.jsonl`, not the later continuation. The KL rows are `results/grpo-250w-kl`, β = 0.05, not the later β = 0.5 run.
&nbsp;
## Counts

| run | steps labeled | prose | junk | first junk step |
| --- | ---: | ---: | ---: | ---: |
| 416-token baseline | 1–20 | 11 | 9 | 5 |
| 1616 uncap | 1–20 | 14 | 6 | 9 |
| trigram repetition | 1–20 | 17 | 3 | 9 |
| KL, β = 0.05 | 1–20 | 9 | 11 | 8 |
&nbsp;
## 416-token baseline

`results/grpo-250w-fp32/rollouts.jsonl`. The token cap is 416.

| step | prompt | rollout | reward | label | reason |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `train-01184` | 0* | 0.000 | prose | Prose: it answers with "historical, social, economic, political, and cultural factors," and the cap cuts it off mid-list. |
| 2 | `train-02221` | 0* | 0.000 | prose | Prose: it tells the reader to "examine the validity and consistency" of each genetic explanation. |
| 3 | `train-03754` | 0 | 0.087 | prose | Prose: it gives steps, starting with "gather and analyze the relevant data." |
| 4 | `train-02519` | 2 | 0.827 | prose | Prose: it matches methods to the data, as in "if the data is qualitative." |
| 5 | `train-01696` | 0 | 0.764 | junk | Junk, a repetition loop: later items are only "the president's role in law" and the same frame again. |
| 6 | `train-02704` | 0 | 0.915 | prose | Prose: it attempts the question, calling the literature "a book that attempts to solve a problem," even with the Founding Fathers in the wrong century. |
| 7 | `train-02983` | 2 | 0.726 | prose | Prose: it attempts the question, from "Indigenous Studies" through a list of unresolved questions. |
| 8 | `train-03461` | 1 | 0.915 | junk | Junk, a repetition loop: the same block, including "best way to prevent the disease from spreading," is copied again after "Answer." |
| 9 | `train-04174` | 0 | 0.911 | junk | Junk, a repetition loop: it leaves the question and repeats "can neither be created nor be destroyed." |
| 10 | `train-03311` | 0 | 0.739 | prose | Prose: it names a misconception and says to correct it with "agile software development." |
| 11 | `train-02842` | 0 | 0.665 | junk | Junk, a repetition loop: "pass through the pores of a filter" is the sentence, over and over. |
| 12 | `train-04764` | 3 | 0.659 | junk | Junk, a repetition loop: the tail is "two sets of people and two sets of items." |
| 13 | `train-04918` | 3 | 0.694 | junk | Junk, a repetition loop: "the transcontinental railroad" and the same east-west sentence return again. |
| 14 | `train-00081` | 3 | 0.654 | junk | Junk, a repetition loop: "Does it make sense to you to solve" is asked again and again. |
| 15 | `train-04078` | 0 | 0.742 | junk | Junk, a repetition loop: "down the fallopian tube" and the blastocyst sentence repeat. |
| 16 | `train-04311` | 2 | 0.774 | prose | Prose: it attempts a historical account, including the "Works Progress Administration," even where the facts are wrong. |
| 17 | `train-03203` | 0 | 0.839 | junk | Junk, a repetition loop: "a great deal more detailed than the 1892 report" refers to that same report. |
| 18 | `train-04745` | 0 | 0.683 | prose | Prose: it attempts an account of a stabilization method, the "R. E. Kosel method," even though that engineer is invented. |
| 19 | `train-01721` | 1 | 0.739 | prose | Prose: it attempts the question by "allowing the production to vary" around a fixed setting. |
| 20 | `train-03040` | 1 | 0.786 | prose | Prose: it attempts the comparison by saying to "take into account the age of the population." |
&nbsp;
## 1616 uncap

`results/grpo-250w-cap1616/rollouts.jsonl`. The token cap is 1616. Steps 1–6 match the trigram and KL winners because those runs had not yet left the shared start.

| step | prompt | rollout | reward | label | reason |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `train-01184` | 0* | 0.000 | prose | Prose: it answers with a numbered list, "Here’s how these factors interact," and closes with a summary. |
| 2 | `train-02221` | 0* | 0.000 | prose | Prose: it gives a structured way to compare genetic explanations, starting at "Evidence Base." |
| 3 | `train-03754` | 0* | 0.000 | prose | Prose: it walks through the evaluation and names "cost, quality, efficiency, and sustainability." |
| 4 | `train-02519` | 0 | 0.027 | prose | Prose: it answers with factors, including "the rise of big data." |
| 5 | `train-01696` | 1 | 0.314 | prose | Prose: it says the explanation should cover the "limitations and boundaries of the president's authority." |
| 6 | `train-02704` | 0 | 0.395 | prose | Prose: it lists what the explanation should cover, beginning with "The context of the time period." |
| 7 | `train-02983` | 3 | 0.495 | prose | Prose: it separates background, current approaches, and unresolved questions, and says the traditions "evolved over time." |
| 8 | `train-03461` | 1 | 0.944 | prose | Prose: after restating the question, it lists unresolved points, including the "role of social and economic factors." |
| 9 | `train-04174` | 1 | 0.676 | junk | Junk, a repetition loop: "the role of the regulation" is copied out to "the next 1000 years." |
| 10 | `train-03311` | 1 | 0.899 | prose | Prose: it names misconceptions, including that software management is "not relevant to business success," and then correction steps. |
| 11 | `train-02842` | 0 | 0.928 | junk | Junk, a repetition loop: "still in its infancy" is the claim, repeated, and it ends at "Answer: some." |
| 12 | `train-04764` | 2 | 0.923 | prose | Prose: it makes claims about the algorithm, including "never developed by a single person," and then correction steps. |
| 13 | `train-04918` | 2 | 0.950 | prose | Prose: it attempts the question with "climate, geography, culture, technology, and politics" and a list of open questions. |
| 14 | `train-00081` | 3 | 0.908 | junk | Junk, a repetition loop: "true or false question" is analyzed again instead of the safety issue. |
| 15 | `train-04078` | 3 | 0.630 | prose | Prose: it starts from factors, "1) genetics" through lifestyle, and then discusses how they interact. |
| 16 | `train-04311` | 3 | 0.931 | junk | Junk, a repetition loop: "no definitive answer" and the media's role swap back and forth. |
| 17 | `train-03203` | 1 | 0.491 | prose | Prose: it picks an answer, "the significance of the study's location," even though that does not cover a study. |
| 18 | `train-04745` | 3 | 0.852 | junk | Junk, a repetition loop: "performance of the controller and the size of the system" is tradeoff 1 and tradeoff 7. |
| 19 | `train-01721` | 3 | 0.775 | prose | Prose: it attempts a tradeoff, "worst case scenario" against the best case, then drifts into other games. |
| 20 | `train-03040` | 2 | 0.883 | junk | Junk, padding that is not an answer: the tail is "The answer is:" stacked on itself. |
&nbsp;
## Trigram repetition

`results/grpo-250w-trigram/rollouts.jsonl`.

| step | prompt | rollout | reward | label | reason |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `train-01184` | 0* | 0.000 | prose | Prose: it answers with a numbered list, "Here’s how these factors interact," and closes with a summary. |
| 2 | `train-02221` | 0* | 0.000 | prose | Prose: it gives a structured way to compare genetic explanations, starting at "Evidence Base." |
| 3 | `train-03754` | 0* | 0.000 | prose | Prose: it walks through the evaluation and names "cost, quality, efficiency, and sustainability." |
| 4 | `train-02519` | 0 | 0.024 | prose | Prose: it answers with factors, including "the rise of big data." |
| 5 | `train-01696` | 1 | 0.251 | prose | Prose: it says the explanation should cover the "limitations and boundaries of the president's authority." |
| 6 | `train-02704` | 0 | 0.391 | prose | Prose: it lists what the explanation should cover, beginning with "The context of the time period." |
| 7 | `train-02983` | 1 | 0.634 | prose | Prose: it attempts the question, saying storytellers "invent new stories" from existing ones. |
| 8 | `train-03461` | 3 | 0.622 | prose | Prose: it separates background, current approaches, and open questions, including "machine learning and artificial intelligence." |
| 9 | `train-04174` | 0 | 0.725 | junk | Junk, a repetition loop: each item is only "because in the question it is stated that." |
| 10 | `train-03311` | 3 | 0.450 | prose | Prose: it names misconceptions, starting with "Over-optimization of software," and a correction for each. |
| 11 | `train-02842` | 2 | 0.478 | prose | Prose: it attempts principles, challenges, and implications, naming "mass spectrometry and X-ray diffraction." |
| 12 | `train-04764` | 1 | 0.624 | prose | Prose: it states a misconception, "only for the business and not for the people," and a way to correct it. |
| 13 | `train-04918` | 3 | 0.196 | junk | Junk, a repetition loop: later numbers repeat "creation of new cities and the development of new technologies." |
| 14 | `train-00081` | 2 | 0.593 | prose | Prose: it says to start from the "importance of safety measures" and the role of technology. |
| 15 | `train-04078` | 1 | 0.504 | prose | Prose: it lists factors, and "The environment plays a crucial role" is one of them. |
| 16 | `train-04311` | 0 | 0.630 | prose | Prose: it attempts a background, "censorship, repression, and surveillance," then lists open questions. |
| 17 | `train-03203` | 0 | 0.456 | junk | Junk, padding that is not an answer: "This is a multiple choice question based on the text provided" is copied in place of an explanation. |
| 18 | `train-04745` | 0 | 0.782 | prose | Prose: it offers examples, including "System Stabilization Metrics," even though the product is beside the point. |
| 19 | `train-01721` | 1 | 0.561 | prose | Prose: it offers examples, including an "increase in the number of variables," and picks one. |
| 20 | `train-03040` | 1 | 0.756 | prose | Prose: it says to judge competing explanations on "Accuracy" and the factors that follow. |
&nbsp;
## KL, beta 0.05

`results/grpo-250w-kl/rollouts.jsonl`.

| step | prompt | rollout | reward | label | reason |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `train-01184` | 0* | 0.000 | prose | Prose: it answers with a numbered list, "Here’s how these factors interact," and closes with a summary. |
| 2 | `train-02221` | 0* | 0.000 | prose | Prose: it gives a structured way to compare genetic explanations, starting at "Evidence Base." |
| 3 | `train-03754` | 0* | 0.000 | prose | Prose: it walks through the evaluation and names "cost, quality, efficiency, and sustainability." |
| 4 | `train-02519` | 0 | 0.027 | prose | Prose: it answers with factors, including "the rise of big data." |
| 5 | `train-01696` | 1 | 0.310 | prose | Prose: it says the explanation should cover the "limitations and boundaries of the president's authority." |
| 6 | `train-02704` | 0 | 0.385 | prose | Prose: it lists what the explanation should cover, beginning with "The context of the time period." |
| 7 | `train-02983` | 2 | 0.678 | prose | Prose: it answers in three parts, and the background says stories are "passed down from generation to generation." |
| 8 | `train-03461` | 3 | 0.644 | junk | Junk, a repetition loop: "methods and tools to predict the spread of diseases" is nearly the whole answer. |
| 9 | `train-04174` | 2 | 0.971 | junk | Junk, a repetition loop: the later items are "environmental policy, environmental policy, and environmental policy." |
| 10 | `train-03311` | 0 | 0.653 | junk | Junk, a repetition loop: "only about managing the software itself" is stated, then stated again as the next misconception. |
| 11 | `train-02842` | 1 | 0.756 | junk | Junk, a repetition loop: the numbers cycle "dark matter in the formation of galaxies" and stars. |
| 12 | `train-04764` | 3 | 0.701 | junk | Junk, a repetition loop: "find the best match for the given data" is the misconception and the correction. |
| 13 | `train-04918` | 2 | 0.892 | junk | Junk, a repetition loop: the open questions are "location of settlements in different parts of the world," repeated. |
| 14 | `train-00081` | 2 | 0.735 | prose | Prose: it attempts the question with concrete measures, "surveillance cameras, access controls, and physical barriers," then other attacks. |
| 15 | `train-04078` | 2 | 0.668 | prose | Prose: it names mechanisms, "epigenetics, neuroendocrinology, and social learning," and then summarizes them. |
| 16 | `train-04311` | 3 | 0.979 | junk | Junk, a repetition loop: cameras, agencies, and social media each "monitor the population in various ways." |
| 17 | `train-03203` | 2 | 0.955 | junk | Junk, a repetition loop: later items are all "the information that was given to the people who were included." |
| 18 | `train-04745` | 1 | 0.770 | junk | Junk, a repetition loop: every method is only there to "prevent the system from going unstable." |
| 19 | `train-01721` | 0 | 0.782 | junk | Junk, a repetition loop: "control the behavior of a system based on the input data" is every sentence. |
| 20 | `train-03040` | 0 | 0.641 | junk | Junk, a repetition loop: each item is the same frame, down to "The more scope the data, the more accurate." |
