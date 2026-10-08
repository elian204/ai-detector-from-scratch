&nbsp;
# Good lists against the frozen gate

No GRPO and no training. No API call. The rules are the ones in `results/grpo-junk-gate.md`, and they were not retuned. The prose PDF was not scored.

These are sixteen numbered answers, each to a different question. The phrasing is parallel on purpose. Some lists are only lightly parallel. Others repeat a frame, including "The role of X is", and each numbered item still adds a distinct fact. A careful reader should accept every list. None of the wording is taken from the junk rollouts.

The counts below are for the answer only. The question line is not part of the text that was scored. These answers are not rollouts, so the cap rule is not applied. A cap hit remains a property of a rollout that used the response token limit.
&nbsp;
## Counts

| list | words | 6-gram count | trigram ratio | rules |
| ---: | ---: | ---: | ---: | --- |
| 1 | 153 | 1 | 0.9934 | none |
| 2 | 162 | 5 | 0.8750 | loop: 6-gram count 5 |
| 3 | 148 | 1 | 0.8973 | none |
| 4 | 142 | 1 | 1.0000 | none |
| 5 | 147 | 1 | 1.0000 | none |
| 6 | 165 | 1 | 0.9448 | none |
| 7 | 163 | 1 | 0.9814 | none |
| 8 | 194 | 1 | 0.9427 | none |
| 9 | 177 | 1 | 0.9943 | none |
| 10 | 168 | 1 | 1.0000 | none |
| 11 | 172 | 1 | 0.9529 | none |
| 12 | 168 | 1 | 0.9880 | none |
| 13 | 140 | 1 | 0.9420 | none |
| 14 | 179 | 1 | 1.0000 | none |
| 15 | 174 | 5 | 0.8372 | loop: 6-gram count 5 |
| 16 | 179 | 1 | 0.9492 | none |

2 of the 16 lists fire. List 2 and list 15 each repeat a six-word frame five times, so the 6-gram count is 5 and the loop rule fires. On list 2 the frame is "in a heat wave, the role". On list 15 the frame is "for a textile in storage, the role". Each item under those frames still states a different fact.

The other 14 lists do not fire. Lists that use "The role of X is", with a different X in each item, keep a 6-gram count of 1, because the changed word sits inside the six-word span. The lowest trigram ratio is 0.8372, on list 15, which is above 0.70. No list repeats an eight-word sentence, none uses "the answer is", none repeats a line, and none is under 30 words.
&nbsp;
## List 1

What should a baker change when a sourdough loaf keeps coming out dense?

```text
A dense sourdough loaf usually fails in one of four places, and each one leaves a different trace in the crumb.

1. The starter may be too young. Feed it until it doubles in four to six hours at room temperature, and use it when the dome has just begun to fall.
2. The dough may be under-fermented. Give the bulk rise more time, and stop when the dough has grown by about half and shows a few large bubbles at the edge of the bowl.
3. The loaf may be under-proofed after shaping. A poke should spring back slowly. If the dent stays, the loaf needs longer in the basket.
4. The oven may be too cool, or the steam may vanish at once. Heat a Dutch oven for half an hour, and bake the loaf covered for the first twenty minutes so the crust does not set before the crumb expands.
```

153 words. 6-gram count 1. Trigram ratio 0.9934. Rules: none.

&nbsp;
## List 2

How can a city cut the risk of death during a heat wave?

```text
A heat wave kills by raising body temperature faster than people can shed heat. The useful work is specific.

1. In a heat wave, the role of street trees is to block sunlight before it is absorbed by asphalt and then radiated back at pedestrians.
2. In a heat wave, the role of drinking fountains is to replace water lost in sweat, which is the main way the body stays under a dangerous temperature.
3. In a heat wave, the role of cooled public rooms is to give people without air conditioning a place to spend the hottest hours, especially those who live alone.
4. In a heat wave, the role of overnight ventilation is to flush heat that masonry stored during the day, so apartments start the next morning cooler.
5. In a heat wave, the role of a check on older residents is to find someone who has stopped sweating or become confused before that person can call for help.
```

162 words. 6-gram count 5. Trigram ratio 0.8750. Rules: loop: 6-gram count 5.

&nbsp;
## List 3

What do the main classes of immune cells do in a first infection?

```text
In a first infection the cells divide the work, and no one cell does all of it.

1. The role of a neutrophil is to arrive within hours and engulf bacteria at the entry site.
2. The role of a dendritic cell is to carry pieces of the microbe to a lymph node and show them to T cells that have not seen that microbe before.
3. The role of a helper T cell is to license B cells and macrophages once it recognizes its matching fragment.
4. The role of a B cell is to secrete antibody that binds the microbe and marks it for clearance.
5. The role of a cytotoxic T cell is to kill the body's own cells when those cells are displaying viral peptides.
6. The role of a memory cell is to remain after the infection so the next encounter expands faster.
```

148 words. 6-gram count 1. Trigram ratio 0.8973. Rules: none.

&nbsp;
## List 4

How should a student read an empirical paper that is outside their field?

```text
Read the paper in passes, and do not start by underlining the introduction.

1. Read the abstract once and write the claim in one sentence of your own. If you cannot, you do not yet know what the authors think they showed.
2. Look at the main figure before the methods. Ask what comparison it makes, what the axes are, and whether the error bars are standard errors or confidence intervals.
3. Then read the methods only for the parts that could produce that figure: who was measured, what was manipulated, and what was held fixed.
4. Read the result that matches the figure, and ignore side analyses until the main comparison is clear.
5. Finish with the limitations the authors name, and add one they do not: a confound, a missing control, or a sample that cannot support the stated scope.
```

142 words. 6-gram count 1. Trigram ratio 1.0000. Rules: none.

&nbsp;
## List 5

Why do tomato seedlings wilt in the week after they are moved outdoors?

```text
Wilting in that first week has several ordinary causes, and they do not call for the same fix.

1. The leaves were raised indoors and have a thin cuticle, so sun and wind pull water out faster than the roots replace it. Harden the plants in shade for a few days first.
2. The potting mix may be dry an inch down even when the surface looks dark. Water until it runs from the drainage holes, then let the top dry.
3. A cold night below about 10 °C slows the roots, and the leaves still lose water in the morning sun.
4. Transplant shock tears root tips. Shade the plants at midday and do not feed them until new growth appears.
5. A fungal rot at the stem, unlike simple thirst, leaves a dark soft ring at the soil line and does not recover after watering.
```

147 words. 6-gram count 1. Trigram ratio 1.0000. Rules: none.

&nbsp;
## List 6

What does each stage of a simple compiler do with a source file?

```text
A simple compiler is a pipeline. Each stage consumes the previous representation and rejects a different class of error.

1. The role of the lexer is to turn characters into tokens, so a later stage never has to know that whitespace and comments were there.
2. The role of the parser is to turn tokens into a tree of expressions and statements, and to reject a missing parenthesis or a clause that is out of order.
3. The role of the name resolver is to bind each use of a variable to the declaration that is in scope, and to report a name that was never declared.
4. The role of the type checker is to reject an operation whose operands do not match, such as adding a number to a string if the language forbids it.
5. The role of the code generator is to emit instructions or bytecode from the checked tree, choosing registers or stack slots for the values that remain live.
```

165 words. 6-gram count 1. Trigram ratio 0.9448. Rules: none.

&nbsp;
## List 7

How should a researcher choose a statistical test for a two-group comparison?

```text
Choose the test from the data and the question, not from the test a neighboring paper used.

1. If the outcome is continuous and you care about a difference in means, start with a model for the mean. A two-sample t test is the small case of that model when the groups are independent and the variances are tolerable.
2. If a few extreme values would dominate the mean, compare medians or use a rank test, and say that the target changed.
3. If the outcome is a yes-or-no event, the comparison is a difference in proportions or a risk ratio, not a t test on coded zeros and ones.
4. If the same person is measured in both conditions, the pairs are not two independent samples, and the test has to use the within-person difference.
5. If you will look at many outcomes, decide the primary one before you see the p-values. The other comparisons are descriptive unless you adjust for them.
```

163 words. 6-gram count 1. Trigram ratio 0.9814. Rules: none.

&nbsp;
## List 8

What keeps food safe in a small restaurant between delivery and the plate?

```text
Safety here is a chain of cold, heat, and clean surfaces. A break in one link is not fixed by care in another.

1. The role of the delivery check is to refuse meat that arrives above 5 °C or produce that is already slimy.
2. The role of the refrigerator is to hold food at or below 5 °C, with raw meat on the lowest shelf so its juices cannot drip onto food that will be eaten raw.
3. The role of the cutting board is to keep raw poultry off the board used for salad, because washing the board later does not undo a salad that was already cut there.
4. The role of the cook step is to bring the center of poultry to 74 °C, which is a temperature, not a color.
5. The role of the holding pan is to keep cooked food above 60 °C until it is served, so spores that survived cooking cannot multiply on the line.
6. The role of the hand sink is to give staff soap and a place to wash after handling raw meat, separate from the sink where vegetables are rinsed.
```

194 words. 6-gram count 1. Trigram ratio 0.9427. Rules: none.

&nbsp;
## List 9

How did movable type change the way European scholars checked a claim?

```text
Movable type did not make a claim true. It changed how many people could compare a claim with a copy of the same page.

1. A scribe's copy drifted. Two monasteries could hold different sentences and both call them the author's. A printed edition gave distant readers the same line numbers and the same errata sheet.
2. Margins filled with printed notes that pointed to other books. A reader in Leiden and a reader in Padua could follow the same citation instead of trusting a teacher's memory.
3. A mistaken diagram could be corrected in the next issue, and owners of the first issue could be told which figure to redraw. Hand copying had no such common update.
4. Pamphlets made a local experiment visible to people who had not been in the room. They could repeat it, or they could write back that the apparatus was described too poorly to build.
5. The same press also multiplied forgeries and bad editions. Checking became possible at a distance, and so did the spread of a confident error.
```

177 words. 6-gram count 1. Trigram ratio 0.9943. Rules: none.

&nbsp;
## List 10

What should a psychology protocol record so another lab can repeat the study?

```text
Another lab can repeat the study only if the protocol fixes the choices that change the numbers.

1. Record the recruitment sentence, the exclusion rules, and the point at which a person is counted as enrolled. "University students" is not a sample.
2. Record every item the participant saw, in order, including practice trials and the feedback text. A paraphrase of the task is not the task.
3. Record the timing: how long a stimulus stayed on screen, how responses were collected, and what happened after a missed response.
4. Record the scoring rule in code or in a formula, including how missing trials were treated. Two honest analysts can otherwise turn the same keypresses into different means.
5. Record the planned sample size and the primary comparison. A later choice, made after the means were visible, is a different study.
6. Record the software versions and the random seed if the task uses one. A timing bug in one browser is enough to move a reaction-time effect.
```

168 words. 6-gram count 1. Trigram ratio 1.0000. Rules: none.

&nbsp;
## List 11

How do the main climate feedbacks change the warming from added carbon dioxide?

```text
Added carbon dioxide warms the surface on its own. Feedbacks then add to that warming or take some of it back.

1. The role of water vapor is to increase as the air warms, and because water vapor is itself a greenhouse gas, the extra vapor adds further warming.
2. The role of snow and ice is to reflect sunlight. When they shrink, darker ground or ocean absorbs more sun, which warms the surface again.
3. The role of low clouds is unsettled in sign. If warming produces more bright low cloud, more sunlight is reflected and the warming is damped. If those clouds thin, the damping is lost.
4. The role of the lapse rate is to move some warming aloft, where heat escapes to space more easily, which offsets part of the water-vapor increase.
5. The role of the land carbon store is to change the airborne share of emissions. A forest that grows faster takes up carbon. A forest that burns or a soil that thaws gives carbon back.
```

172 words. 6-gram count 1. Trigram ratio 0.9529. Rules: none.

&nbsp;
## List 12

What should a reviewer look for in a change that touches shared library code?

```text
A change to shared library code can break callers the author did not run. Review it for those callers, not only for style.

1. Check that the new behavior is what the tests actually assert. A test that only checks the happy path will stay green when an error code changes.
2. Look for a default that changed. Callers who omitted an argument were relying on the old default, and they will not show up in the diff.
3. Look for a lock, a file path, or a cache key that is now shared across threads or processes. A race in a library becomes a race in every program that imports it.
4. Read the names that leaked into the public module. A helper that was private yesterday is a promise today if its name is imported by other packages.
5. Ask for a note in the changelog when the observable behavior moved, including a behavior the author considers a bugfix. Callers may have depended on the bug.
```

168 words. 6-gram count 1. Trigram ratio 0.9880. Rules: none.

&nbsp;
## List 13

How do the four parts of a string quartet divide a musical line?

```text
The four parts are not four copies of the melody. They divide the line so the harmony and the pulse stay audible.

1. The role of the first violin is to carry the melody when the tune sits on top, and to step back into figuration when another part has the phrase.
2. The role of the second violin is to complete the inner harmony and to double a rhythm that would be too thin in one voice.
3. The role of the viola is to fill the register between the violins and the cello, often with the notes that decide whether a chord is major or minor.
4. The role of the cello is to supply the bass note that locates the harmony, and, in many slow movements, to take the melody when it drops into the lower octave.
```

140 words. 6-gram count 1. Trigram ratio 0.9420. Rules: none.

&nbsp;
## List 14

What limits yield in a rain-fed wheat field when the farmer cannot irrigate?

```text
Without irrigation the crop spends the water the sky and the soil provide. Yield is limited by whichever of these runs out first.

1. Rainfall in the weeks before flowering sets the number of grains. A dry spell then cannot be made up by rain after the heads have formed.
2. The soil's store of water depends on depth and on organic matter. A shallow soil over rock holds less than a deep loam, even in the same weather.
3. Nitrogen deficiency stops the canopy early. The plants transpire less, but they also build less grain, so saving water does not save the yield.
4. Weeds take up water and nitrogen the wheat would have used. A weedy field can look green and still thresh poorly.
5. Heat during grain fill shortens the time starch can move into the grain. Extra rain after that heat does not restore the days that were lost.
6. Disease on the flag leaf cuts the photosynthetic area that fills the grain. The roots may still have water while the leaf cannot use it.
```

179 words. 6-gram count 1. Trigram ratio 1.0000. Rules: none.

&nbsp;
## List 15

How should a museum store a dyed textile so the color and the fibers last?

```text
Light, damp, pests, and folds damage a dyed textile in different ways. Storage has to answer each of them.

1. For a textile in storage, the role of darkness is to stop light from breaking the dye molecules, which is why a windowed gallery shelf is a poor cupboard.
2. For a textile in storage, the role of a stable moderate humidity is to keep fibers from swelling and shrinking, because that movement cracks brittle threads and can bloom salts.
3. For a textile in storage, the role of an acid-free support is to keep wood and ordinary cardboard from releasing acids that weaken cotton and linen.
4. For a textile in storage, the role of a loose roll or a padded fold is to avoid a sharp crease, which becomes a break after years under the cloth's own weight.
5. For a textile in storage, the role of a sealed clean cabinet is to keep out moths and carpet beetles, whose larvae eat wool and silk whether or not the room looks tidy.
```

174 words. 6-gram count 5. Trigram ratio 0.8372. Rules: loop: 6-gram count 5.

&nbsp;
## List 16

What does each main part of a balance sheet tell a reader about a firm?

```text
A balance sheet is a statement of position on one date. The parts answer different questions, and none of them is a measure of profit.

1. The role of cash is to show what can be spent at once without selling a thing or collecting a bill.
2. The role of receivables is to show sales already recognized for which the customer has not yet paid, so the cash is still a claim rather than money in the account.
3. The role of inventory is to show goods bought or made and not yet sold, valued at cost or at a lower market value, not at the price the firm hopes to get.
4. The role of long-term debt is to show principal that is not due within a year. Interest that has not yet come due is not included in that line.
5. The role of equity is to show the residual after liabilities are subtracted from assets. It is not a pile of cash, and it can be large in a firm that is short of money.
```

179 words. 6-gram count 1. Trigram ratio 0.9492. Rules: none.
