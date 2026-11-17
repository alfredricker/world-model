---
id: "144"
title: a centre that means the middle — the read-out's weights are trained on, in a second stage, until their centre in the image is the object's middle under a fixed smooth map
rung: 3
serves: [P1, P5, P4]
status: done
verdict: fail
arch_version: og4
date: 2026-11-17
---

# 144: a centre that means the middle

Roadmap step 2 (the perceptual side), rung 3. Fourth card of the
representation week. Drafted and run on the user's "do what you think
is most important next" (2026-11-17), so the bars below are mine and
were written before the second stage was first run.

og4's aim read-out answers with a sum over patches of *weight ×
(place + offset)*. [Card 142](../142-find-then-place/card.md) showed
the trade: the offset, computed from a patch's features, is right to
0.45 cm where the cube was touched and 5.9 cm off well outside; a
smooth map from the centre of the weights is 2.7 cm off outside but
only 1.05 cm inside. [Card 143](../143-found-where-it-was-let-go/card.md)
met the same fault in the open drawer, where 66 put-ins are all the
training there is: 1.70 cm, growing with distance from a label.

Card 142 proposed adding a bounded offset to the map. The data check
below points at something simpler first. The weights are already
tight around the cube, but their centre is about 0.8 cm noisy even
under the best possible map, because nothing ever trained the centre
to mean anything: the offset absorbed whatever the weights did.

Biological counterpart: as card 142 (a map from retinal location to
reach coordinates, shared and calibrated by the hand; posterior
parietal cortex). What is added is that the "what" side learns to
hand over a location the "where" side can use.

## 1. Question

If, after og4's read-out is trained, its map is fixed as a smooth
function of the centre of the weights and the weights alone are
trained on, is the cube read as precisely as og4 reads it inside the
range of its labels, better outside that range, and better in the
open drawer? Rung 3. P1 (grounded meaning), P5 (knowing where a
reading applies), P4 (the reading holds when surroundings change).

## 2. What changes

One component: how the aim read-out turns weights into a position,
for the cube group (the only group whose aim reading is switched on).
Features, labels (og4's contact points plus card 143's release
points) and the first stage are unchanged.

```
og4 form:  position = sum over patches of weight × (place(row, column) + offset(patch features))
144:       stage 1  og4's form, as now; keep its trunk and score, drop place and offset
           stage 2  centre   = sum over patches of weight × (row, column)
                    position = map_h(centre): second order in row and column, one map per
                    height at which the group has labels (the table; above it), fitted by
                    least squares on the agent's own labels and held fixed
                    then train, on the same labels, 3,000 steps:
             tune   the group's score (the last layer; the trunk is frozen)
             off    a shift of each patch's location computed from its features, bounded to
                    one patch (card 142's proposal); the score frozen
             both   both
           the map is fitted once more at the end
answer:    the height is that of most of the 5 labels whose centres lie nearest in the image
```

Arms: og4's form (baseline); the map alone (card 142's
fitted-afterwards arm); tune; off; both. The card's arm is **tune**;
off and both are run because card 142 named the bounded offset.
Each is fitted on all labels and with the cube's labels in the upper
third of the table withheld, over 3 seeds.

Not in this card: the other groups, one map shared by all groups
(card 142 found it close to a map per group), the hidden cube, the
slides, and the agent's behaviour.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Where a patch lies in the image fixes where its content lies, given a height | no: the view moves, so the map must also take the agent's pose | holds | holds (the grid) | fixed camera only | declared, as card 142 |
| An object rests at one of a few heights, each with its own region of the image | no | no height | no height | table-top scenes only | declared; the general form reads height from depth or a second view |
| The agent's contact and release points are known | pose given | holds | the acted-on cell | holds | general |

## 3. Dependencies

- Stage 1 and the labels: cards 128 to 129.2 and 143
  (`runs/142/fit_og_*.pt`, already trained on 143's labels).
- The test bed and the withheld split: card 141.
- The centre of a softmax as a position: the spatial soft arg-max
  (Levine and others 2016), as card 142.

## 4. Data check

Done 2026-11-17 (`tools/card144/dump.py`, `check.py`; seed 0; under
2 minutes each). Median error of the cube on the table, 4,793 test
frames.

| Centre of the weights taken as | Map fitted on the agent's labels | Map fitted on true positions (the best any map can do) |
|---|---|---|
| all 1,024 patches (card 142's arm) | 1.08 cm | 0.78 cm |
| the 5 × 5 patches round the best one | 1.13 | 0.82 |
| the 3 × 3 patches round the best one | 1.22 | 1.05 |
| all patches, scores sharpened four-fold | 1.45 | 1.36 |

og4's form reads 0.45 cm from the same weights. So stray weight is
not the fault (the median weight outside the 5 × 5 window is 0.00,
and the best patch holds 0.48 of it), a better map cannot close the
gap (0.78), and sharpening makes it worse: the spread of the weights
carries the position, but not yet as their centre.

Labels for stage 2: 4,075 on the table and about 260 above it (66
put-ins, four frames each) in the training episodes; 2,771 on the
table with the third withheld.

## 5. Feasibility gate

- **Upper bound:** the tune arm on all labels, seed 0, reads the
  cube on the table to at most 0.75 cm. If the form cannot reach
  that where it was trained, there is no test.
- **Trivial baseline:** the map alone, 1.08 cm on the table and
  1.84 cm in the withheld third (card 142); og4's form, 3.32 cm in
  the withheld third.
- **The height rule:** it chooses the table for at least 98% of test
  frames with the cube on the table, and "above" for at least 90% of
  those with it in the open drawer.

Result of the gate (2026-11-17): passed. The tune arm on all labels,
seed 0, reads the table to 0.56 cm (bar 0.75). The height rule
chooses the table in 99.1% of table frames (bar 98) and "above" in
92.2% of open-drawer frames (bar 90).

## 6. Success criteria and prediction

Median error of the tune arm's one answer per frame, mean of 3
seeds.

1. **No cost inside:** fitted on all labels, the cube on the table
   is read to at most 0.75 cm (og4's form 0.45; the map alone 1.08),
   and the bed's "both on the table" cell is at least 98.5 (og4's
   form 99.3).
2. **Further outside:** with the upper third withheld, at most
   1.5 cm there (og4's form 3.32; the map alone 1.84) and at most
   2.5 cm more than 5 cm past the edge (5.86; 2.66).
3. **The open drawer:** fitted on all labels, at most 1.0 cm with
   the cube in the open drawer (og4's form 1.70; card 143's bar),
   over 985 test frames.

Reported without a bar: off and both; the open-drawer pair cell;
the size of the learned shift; the largest score in each cell.

**Prediction.** 1 passes: the score has 129 free numbers per group
and thousands of labels. 2 is uncertain: the weights carry over
(card 142), but a centre trained to be exact inside need not stay
exact outside. 3 is the doubtful one: one seed in three finds the
cube weakly in the drawer, and a frozen trunk may not be able to
mend that.

**Budget.** One capped run per training set and seed, about 4
minutes each, 6 runs; nothing over 10 minutes.

## 7. Result

Run 2026-11-17, `tools/card144/` (`fit.py`, `score.py`; 6 capped
runs of about a minute each), numbers in
[results.json](results.json). Median error of the cube, mean of 3
seeds; 4,793 test frames on the table (1,650 in the withheld third,
634 over 5 cm past its edge) and 985 in the open drawer.

| Arm | All labels: table | Cell "both on the table" | Third withheld: withheld third | Over 5 cm past the edge | All labels: open drawer |
|---|---|---|---|---|---|
| og4's form (weight × (place + offset)) | 0.45 cm | 99.4 | 3.32 | 5.86 | 1.70 (1.41, 1.51, 2.17) |
| The map alone | 1.09 | 95.8 | 1.84 | 2.66 | 1.90 (1.09, 1.37, 3.23) |
| **Tune: the score trained on (the card's arm)** | 0.67 | 98.3 | 1.94 | 3.08 | 1.76 (1.14, 1.27, 2.88) |
| Off: a shift bounded to one patch, score frozen | 0.61 | 99.1 | 1.21 | 1.89 | 1.75 (1.14, 1.23, 2.89) |
| Both | 0.51 | 99.2 | 1.32 | 2.15 | 1.63 (1.24, 1.24, 2.42) |
| Bars | at most 0.75 | at least 98.5 | at most 1.5 | at most 2.5 | at most 1.0 |

1. **No cost inside: fail, on the cell.** 0.67 cm passes its bar;
   the pair cell is 98.3 against 98.5.
2. **Further outside: fail.** 1.94 and 3.08 cm, no better than the
   map alone (1.84 and 2.66): training the score to be exact inside
   the range does not carry outside it.
3. **The open drawer: fail.** 1.76 cm against og4's form at 1.70.

What the run showed beyond the criteria:

- **The bounded shift, which card 142 proposed and this card
  demoted to a side arm, is the form that works.** With the score
  frozen it reads the table to 0.61 cm (cell 99.1) and the withheld
  third to 1.21 cm, 1.89 cm beyond 5 cm: inside both of this card's
  bars for criteria 1 and 2, in every seed (1.13, 1.33, 1.17 in the
  withheld third). Against og4's form it gives up 0.16 cm inside and
  gains 2.1 cm outside. The shift it learns is small: a median of
  0.4 of a patch.
- **Why the shift carries and the score does not** is not shown by
  this run. A reading consistent with it: the shift cannot move the
  answer more than one patch whatever the features say, while a
  retrained score can move the weights anywhere.
- **The open drawer is a fault of stage 1, in one seed.** Seeds 0
  and 1 read it to 1.1 to 1.3 cm in every stage-2 arm (og4's form:
  1.41 and 1.51); seed 2, the seed card 143 found to score the cube
  under zero there, reads 2.4 to 3.2 cm. No stage-2 arm mends it.
- **The height rule works from the image alone:** the table is
  chosen in 99.0% of table frames and "above" in 96% of open-drawer
  frames.

## 8. Decision

**Revise.** The card's arm fails all three criteria, so nothing is
adopted from it, and my guess that training the score was the
simpler fix was wrong. The side arm gives the read-out card 142 was
looking for: og4's weights, a smooth map, and a shift bounded to one
patch, fitted in two stages. It is a candidate for the agent's
read-out, at a stated loss of 0.16 cm inside the range for 2.1 cm
outside it; whether to adopt it is the user's call, and it has not
been tried in behaviour. What it does not fix is the open drawer,
where the remaining fault is stage 1 finding the cube weakly in one
seed of three, from 66 put-ins.
