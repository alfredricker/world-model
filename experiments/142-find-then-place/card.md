---
id: "142"
title: find, then place — which patches hold an object is read from its appearance, and where those patches lie is read from a map shared by every object
rung: 3
serves: [P1, P5, P8]
status: draft
verdict:
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

Result of the gate, before the main run:

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

## 8. Decision
