&nbsp;
# Pilot read at steps 0, 10, and 20

The reward-judge win rate rose from step 10 to step 20 while the Claude win rate fell. At step 10, 2 of 4 rollouts beat the frozen base answer and 2 were ties. At step 20, 4 of 4 beat it. Claude went from 11/20 to 7/18.

No new reward-judge calls were made. The step 10 and step 20 signals were already in the pilot log. Step 0 has no saved generation, so its win rates are blank. Claude at step 0 was not measured.

The step 10 and step 20 rows are not the same prompt. Step 10 is `train-02178`. Step 20 is `train-04895`. Each row uses the four rollouts from that step. The base answer in the reward-judge column is the frozen initial-policy answer saved in that same step log, not the validation answers used for the Claude check. The validation generations from the Claude checks were not saved.

Step 0 is the 20 saved base answers at index 0, `validation-00001` through `validation-00020`. Those are the prompts paired with the step-20 sheet below. They are not `train-02178` or `train-04895`.

&nbsp;
## Scores

The soft score is the mean of sigmoid(z / 4). z is the raw human-class logit from frozen `qwen3-variable`, before the saved temperature. DistilBERT is mean P(human). The trigram ratio is the lowercased unique-trigram ratio. Word length is the mean whitespace word count.

A reward-judge win is signal greater than 7. Absolute signal of 7 or less is a tie. The win rate is the number of rollouts that beat the base, out of the four rollouts.

| Step | Samples | Soft score | DistilBERT P(human) | Reward-judge wins | Claude vs base | Mean trigram | Mean words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 saved base answers | 0.3447 | 0.3870 |  | not measured | 0.9616 | 162.5 |
| 10 | 4 rollouts, `train-02178` | 0.1789 | 0.7282 | 2/4, and 2 ties | 11/20 | 0.9119 | 457.8 |
| 20 | 4 rollouts, `train-04895` | 0.4979 | 0.7480 | 4/4, and 0 ties | 7/18 | 0.8688 | 591.8 |

Mean z, the logit inside the soft score, was -2.956 at step 0, -6.125 at step 10, and -0.207 at step 20.

At step 10 the four signals of rollout over base were 19.687, -0.625, -4.250, and 21.143. The two middle values are ties. At step 20 they were 18.228, 12.979, 10.866, and 21.304, all wins.

&nbsp;
## How the step-20 answers differ

The comparison here is the sheet written from the step-20 weights, on `validation-00001` through `validation-00020`, against the saved base answer to the same prompt. These are not the four `train-04895` rollouts in the table, and they are not the exact texts Claude judged. Claude's 7/18 remains the saved check.

The trained answers are longer, more outlined, and more willing to invent a concrete program. The base answers are shorter and stay on the generic request. Neither side is casual speech. The trained voice is a classroom answer: it announces what it will do, then lists principles. I did not find a surface trick aimed at the detector, such as hidden characters or a line addressed to a classifier. What did change in the training group is the soft score, from 0.1789 at step 10 to 0.4979 at step 20, while the answers got longer and the trigram ratio fell.

&nbsp;
### validation-00009

The question asks what a thorough explanation of state capital relocation should cover. The trained answer invents a program and names the wrong cities. The base answer stays with the request and does not invent a relocation.

Trained: "The State Capital Reclamation program is a comprehensive strategy implemented by the state of Michigan to address the issue of rapid population growth in certain areas, leading to overcrowding and urban sprawl. The program aims to relocate the state capital from downtown Ann Arbor to the more spacious and environmentally-friendly town of Dearborn."

Base: "A thorough explanation of state capital relocation attempts should cover several key aspects. First, it should provide a clear overview of the reasons behind the relocation, including historical, political, and economic factors."

&nbsp;
### validation-00018

The question asks for examples of ideas and tradeoffs in criminal justice supervision. The trained answer talks about the question. The base answer talks about supervision.

Trained: "In this question, I will analyze the key ideas and tradeoffs in criminal justice supervision strategies and provide examples to illustrate these concepts. One of the key ideas in criminal justice supervision strategies is the need for a balance between individual rehabilitation and societal safety."

Base: "Supervision strategies in criminal justice play a crucial role in managing offenders and ensuring public safety. Two key ideas that guide these strategies are the balance between deterring crime and preventing reoffending."

&nbsp;
### validation-00001

The question is about misconceptions in safe learning systems. The trained answer runs to 1,438 words and ends in named campus programs. The base answer is 199 words and stays with the misconceptions.

Trained: "Stanford University has regularly assessed the program’s effectiveness and made adjustments as needed. For example, the university has implemented new policies and procedures to address emerging safety concerns, and has increased funding for safety training for staff and learners."

Base: "One common misconception about safe learning systems is that they are solely the responsibility of teachers. While teachers play a crucial role in creating a safe learning environment, students also have a part to play in ensuring safety."
