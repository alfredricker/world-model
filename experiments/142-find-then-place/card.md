---
id: "142"
title: find, then place — which patches hold an object is read from its appearance, and where those patches lie is read from a map shared by every object
rung: 3
serves: [P1, P5, P8]
status: done
verdict: fail
arch_version: og4
date: 2026-11-14
---

# 142: find, then place

Roadmap step 2 (the perceptual side), rung 3. Second card of the
representation week.
[Card 141](../141-representation-test-bed/card.md) found that a
position read by a trained map from patches is right only where that
map was trained, whether contact points or true labels train it. It
proposed a reading that compares against remembered frames. Two
checks before drafting (2026-11-14, `tools/card142/check.py` and
`check_open.py`, 6 minutes and under 1 minute) changed that proposal.

The aim read-out answers with a sum over patches of *weight × (place +
offset)*: the weight says which patches hold the cube, the place is a
smooth map from a patch's row and column to a position, and the
offset is a correction computed from the patch's features.

1. **Finding the cube carries over; placing it does not.** With the
   cube's touches in the upper third of the table withheld from
   training, the read-out's median error there is 2.18 cm (0.62 when
   nothing is withheld). Taking the weights from the withheld fit and
   the place and offset from the full fit gives 0.72 cm; the other
   way round gives 2.03. The two fits put their largest weight on the
   same patch or its neighbour in 89% of those frames. The part that
   fails is the offset (2.16 cm with only the offset from the
   withheld fit, 0.89 with only the place map from it): a correction
   computed from features is free to depend on anything in them,
   and it was only ever trained where the cube was touched.
2. **The offset is not needed for precision.** A second-order fit
   from the centre of the weights in the image to the cube's position
   reads the cube on the table to 0.57 cm, beside 0.60 for the
   read-out's own answer.
3. **The cube in the open drawer is a different fault.** There the
   centre of the weights does no better than always answering the
   middle of the drawer (1.83 against 1.49 cm), and the largest
   score is half what it is on the table (3.27 against 6.77). The
   cube is not found. This card does not address it.

So recall over remembered frames is not what the first fault needs: a
neighbour can only answer between remembered positions, never past
them. What it needs is to keep the two steps apart.

Biological counterpart: the visual system answers "what" and "where"
in separate pathways (Ungerleider and Mishkin 1982). Where a seen
object lies in reaching coordinates is computed in the posterior
parietal cortex by a map that serves every object and is calibrated
by seeing one's own hand; nobody relearns where things are for each
new kind of object.

## 1. Question

If the position of an object is read as "find its patches by
appearance, then turn their location in the image into a position by
a smooth map that never sees the object's features", is the reading
right outside the range where the object was touched, at no cost
inside it? And can that map be one map for all objects? P1 (grounded
meaning), P5 (knowing where a prediction applies), P8 (composition:
a new object's appearance with an existing sense of place).

## 2. What changes

One component: how the aim read-out turns weights into a position.
Features, the score that gives the weights, the training signal (the
agent's own contact points) and the training schedule are og4's.

```
og4:      position = sum over patches of weight × (place(row, column) + offset(patch features))
          place is a free network; everything is trained together per object
142 own:  centre   = sum over patches of weight × (row, column)
          position = place(centre), second order in row and column, fitted on the object's
          own touches; no offset
142 shared: position = place(centre, height), one map for every object, fitted on the touches
          of all objects; each object brings only its score and the height at which it is
          touched (the mean height of its own contact points)
```

Arms: og4 (baseline); own map; shared map. Each is trained twice, on
all touches and with the cube's touches in the upper third of the
table withheld (card 141's split: 3,143 test frames in the kept
part, 1,650 in the withheld part), over 3 seeds.

**Amended before the run (2026-11-16, after
[card 143](../143-found-where-it-was-let-go/card.md)).** The card was
approved to run after 143, on its training set: og4's contact labels
plus the release labels (up to four frames per release). Three things
follow, none of which changes a criterion:

- Every arm, the baseline included, is trained on that set, so the
  gate measures the baseline again; "og4" below means og4's form of
  read-out.
- The cube now has labels at two heights (the table and, from
  put-ins, the drawer), and one image location means two positions
  at two heights. So the own map also takes the height, like the
  shared one: `position = Q(centre, height) · B`, second order in
  row, column and height, with one `B` per group (own) or one for
  all groups (shared). The height is the hand's own height at the
  label in training, and the group's usual height (the median of
  its labels) when answering.
- The maps are trained together with the score, as og4's read-out
  is, not fitted afterwards.

Reported without a bar: the cube in the open drawer, answered at the
height of the cube's labels above the table.

Not in this card, and named so that nothing is hidden: the cube in
the open drawer (the finding step fails, check 3), and the opening
of a slide (the appearance reading is coarse, card 141). Each is the
finding step, and each is a later card.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Where a patch lies in the image fixes where its content lies, given a height | no: the view moves, so the map must also take the agent's pose | holds (flat, top-down, centred on the agent) | holds (the grid) | holds for a fixed camera; a moving camera needs its pose | declared: one fixed camera. The pose is a further input to the same map, not a new part |
| An object is touched at a steady height | no (blocks at any height) | no height | no height | often (table-top), not in general | declared; the general form reads height from a second view or from depth |
| The agent's contact point is known | pose is given (C1) | holds | no body; the acted-on cell instead | holds (proprioception) | general |
| Objects are found by appearance | holds | weak on pixel art | holds | holds | og4's score, unchanged |

## 3. Dependencies

- The read-out and its training signal: cards 128 to 129.2.
- The test bed and the withheld split: card 141, with the two repairs
  of section 5.
- The centre of a softmax over image locations as an object's
  position is the spatial soft arg-max of Levine and others (2016,
  "End-to-end training of deep visuomotor policies"). It has no
  LITERATURE.md entry yet; one is added before the run.

## 4. Data check

Done 2026-11-14 from the read-out's stored touches (the first 80
episodes, as in card 129).

| Object group | Touches | Height of contact (10 cm units, 5th to 95th percentile) |
|---|---|---|
| Cube | 891 (the withheld fit drops those with y over 10.2 cm) | 0.18 to 0.30 |
| Drawer handle | 575 | 0.77 to 0.88 |
| Two buttons | 723 and 483 | 1.14 to 1.34 |
| Window handle | 591 | 2.30 to 2.38 |

The shared map is therefore fitted from five clusters at four
heights, 3,263 touches. That is thin for a map over the whole table:
the window's touches are the only ones of another object beyond the
withheld edge, and they lie about 21 cm higher than the cube. The shared arm is the uncertain one for this reason.

## 5. Feasibility gate

First the two repairs card 141 asked for, since its bed is the bar:
a gap in the smallest slide cell (same under 0.5 cm, different 2 to
4 cm), and the labelled probe over 5 seeds with the spread of its
"can the cube be located" answer reported.

- **Upper bound:** og4's read-out trained on all touches, scored in
  the upper third: 0.62 cm.
- **Trivial baseline:** og4's read-out with the third withheld: 2.18
  cm (0.64 in the kept part).
- The gate passes if these two hold over 3 seeds with a gap of at
  least 1 cm between them in the withheld third, and the repaired
  probe scores at least 95 in both smallest slide cells. Check 2
  already shows the own map can match og4 inside the range (0.57
  against 0.60 cm).

Result of the gate (2026-11-16): passed. Every fit here is trained on
card 143's labels.

- **Repairs.** With a gap, the probe scores 97.9 in the drawer's
  smallest cell (5 seeds: 97.8 to 98.0) and 95.3 in the window's
  (94.5 to 95.6; one seed is under 95). Its "can the cube be located"
  answer has a balanced accuracy of 96.4 on average but 90.2 to 98.3
  across seeds; with the drawer shut it says "locatable" in 3 to 29%
  of frames (the first card's seed was the 29%), and the hidden-cube
  pair cell runs from 82.4 to 94.6. That answer is not steady enough
  to be a bar; the cube's value cells are (0.16 to 0.17 cm on the
  table, 0.30 to 0.40 in the open drawer).
- **Upper bound:** og4's form trained on all labels reads the upper
  third to 0.50 cm (seeds 0.51, 0.54, 0.45).
- **Trivial baseline:** with the third withheld, 3.32 cm there (2.24,
  3.57, 4.15) and 0.45 in the kept part. The gap is over 1 cm in
  every seed.

## 6. Success criteria and prediction

Median error of the cube's position on the table, mean of 3 seeds.
Proposed, to be fixed with the user.

1. **Right outside the range:** with the upper third withheld, the
   own map reads the cube there to at most 1.0 cm (og4 2.18; all
   touches 0.62), and to at most 1.5 cm more than 5 cm past the edge
   (og4 4.36).
2. **No cost inside:** trained on all touches, the own map reads the
   cube to at most 0.75 cm (og4 0.60), and the bed's "cube, both on
   the table" cell stays at or above 98.5 (og4 99.4).
3. **One map for all objects:** the shared map is within 0.3 cm of
   the own map in both parts of criterion 1. This is what would make
   place a reusable part: a new object would need only its
   appearance learned.

Reported without a bar: the error of each arm for the other object
groups, and the cube cells of the bed for each arm.

**Prediction.** 1 passes: the weights carry over (check 1) and a
second-order map has too few parameters to bend at the edge. 2
passes narrowly. 3 is doubtful for the reason in section 4; if it
fails, the next source for the shared map is the agent's own hand,
which is seen with a known position in every frame and at every
height.

**Budget.** Three arms, two training sets, 3 seeds: 18 fits of about
3 minutes each, run as one capped run per arm of about 18 minutes
(under the 30-minute limit once the card is approved); the probe's 5
seeds take about 5 minutes more.

## 7. Result

Run 2026-11-16, `tools/card142/` (`fit.py`, `after.py`, `score.py`;
30 capped runs, each under 5 minutes), numbers in
[results.json](results.json). Median error of the cube on the table,
mean of 3 seeds; 4,793 test frames, 1,650 of them in the withheld
third and 634 more than 5 cm past its edge.

| Arm | All labels: whole table | Third withheld: kept part | Withheld third | Over 5 cm past the edge |
|---|---|---|---|---|
| og4's form (weight × (place + offset)) | 0.45 cm | 0.45 | 3.32 | 5.86 |
| Own map, trained with the score (declared) | 4.19 | 1.55 | 3.70 (8.11, 1.49, 1.49) | 5.04 |
| Shared map, trained with the score (declared) | 3.87 | 3.85 | 8.22 | 10.41 |
| Own map, fitted afterwards on og4-form weights | 1.08 | 1.05 | 1.84 | 2.66 |
| Shared map, fitted afterwards | 1.46 | 1.35 | 2.28 | 2.70 |
| Bars (own map) | at most 0.75 | | at most 1.0 | at most 1.5 |

The last two arms were added during the run and are not criteria.
They are the form of this card's own check 2: the weights of an
og4-form fit, their centre turned into a position by least squares on
the agent's own labels (`after.py`).

1. **Right outside the range: fail.** 3.70 cm in the withheld third
   (bar 1.0) and 5.04 cm beyond 5 cm (bar 1.5), no better than og4's
   form.
2. **No cost inside: fail.** 4.19 cm on the table against og4's 0.45
   (bar 0.75); the pair cell "both on the table" falls from 99.3 to
   63.1.
3. **One map for all objects: fail.** The shared map trained with the
   score is 4.5 and 5.4 cm worse than the own map.

What the run showed beyond the criteria:

- **Training the map together with the score is unsteady.** Two
  seeds of the own map reach 1.49 cm in the withheld third and one
  8.11; trained on all labels it is worse than with a third
  withheld. The amendment's choice to train them together was mine
  and was not checked first; these arms say little about the idea.
- **Fitted afterwards, the map does carry further than the offset,
  and costs precision where the offset was trained.** In the
  withheld third 1.84 against 3.32 cm, and beyond 5 cm 2.66 against
  5.86; inside the range 1.05 against 0.45. Check 2's "the offset is
  not needed for precision" (0.57 against 0.60 cm) does not hold
  here: that fit used true positions on 6,587 frames and og4's
  original weights.
- **One map for every object is nearly as good as a map per
  object**, fitted afterwards: 0.44 and 0.04 cm apart in the two
  parts of criterion 1 (bar 0.3), from five clusters of labels at
  five heights.
- **The cube in the open drawer is not helped:** answered at the
  drawer's height, 1.4 to 1.9 cm with the own map fitted afterwards
  and 2.2 to 4.0 with the shared one, against 1.5 to 1.7 for og4's
  form.

## 8. Decision

**Revise.** All three criteria fail, and nothing changes in the
agent: og4's form trained on card 143's labels stays the best
reading inside the range (0.45 cm). What the card leaves is a
sharper statement of the trade. The centre of the weights through a
smooth map is right to about 1 cm everywhere it was tested,
including 2.7 cm at more than 5 cm outside training; the offset from
features is right to 0.45 cm inside and 5.9 cm outside. Next, for
the user to approve: keep both, with the offset bounded so that it
can only correct within about one patch (a shift of the centre in
the image before the smooth map, not a free position), which cannot
grow outside the range by construction; fitted in two stages so that
the unsteady joint training is not in the way.
