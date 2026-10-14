---
id: "082.1"
title: reasoning over recalled tries, trained on both kinds of query with combinations weighted equally
rung: 6
serves: [P3, P10, P21, P6]
status: done
verdict: fail
arch_version: 18
date: 2026-10-06
---

# 082.1: both kinds of query, combinations weighted equally

Card 082's revision, approved by the user (2026-10-06: "try option 1").

## 1. Question

Card 082's network failed its gate: with nothing removed from memory,
33–44 of 56 toggle cells right, against version 18's 56. Its diagnosis
found two gaps in the training, not the architecture:

- **No query beside its own combination.** Training never showed a
  query whose own (front, held) combination was in its context, though
  the agent meets such queries all the time.
- **The relation was drowned out.** The loss weighted every row
  equally, so about 70 door and key rows were outweighed by some 3,900
  rows where nothing happens.

With both fixed, does the network pass card 082's gate, and then its
criteria?

## 2. What changes

Only the training, against card 082:

| | Card 082 | This card |
|---|---|---|
| Queries per step | 32 combinations hidden whole | half the steps as card 082; half hide 128 single rows, leaving the rest of each row's combination in context |
| Loss | mean over query rows | mean over query combinations of the mean over their rows: each combination weighs the same (card 071's unit) |

The architecture, inputs, router, three actions, seeds and steps (3,000)
are unchanged.

## 3. Dependencies

Card 082 (`tools/card082/reason.py`).

## 4. Data check

As card 082.

## 5. Feasibility gate

As card 082: nothing removed, seeds 79, 80, 81, all cells of the
toggle, pick-up and drop tables right. Also at seed 79: no router, and
no memory.

## 6. Success criteria and prediction

Card 082's three criteria, unchanged. Each must hold at each of seeds
79, 80 and 81:

1. The left-out pair opens in 6 of 6 folds, with no other toggle error.
2. The pick-up and drop tables are right in every fold.
3. With the retrieved outcomes shuffled, the left-out pair opens in at
   most 2 of 6 folds.

The no-router and no-memory arms run in the gate only, to keep the
folds within about 25 minutes.

**Prediction.** The gate passes: with single rows hidden, a mismatched
pair's own failure can be copied when it is in context, and the door
and key combinations now carry real weight. Criterion 1 is uncertain.

**Decision rules.**
- **Keep / revise / stop** as card 082.
- **If the gate fails again, stop:** the next card gives the reasoner
  card 070's learned relation rather than making it learn one (card
  082's section 8).

**Budget.** Gate about 5 minutes; folds 6 × about 3.5 minutes.

## 7. Result

`runs/082/fold_none_082.1.{json,log}` (`tools/card082/reason.py` with
`WM_MIX=1 WM_COMBO_WEIGHT=1`). Only the gate ran.

**Gate: not passed.** Cells right with nothing removed (toggle of 56,
pick up of 28, drop of 42):

| Arm | Toggle | Pick up | Drop | Card 082's toggle |
|---|---|---|---|---|
| Router, seed 79 | 44 | 28 | 41 | 33 |
| Router, seed 80 | 49 | 28 | 42 | 37 |
| Router, seed 81 | 49 | 26 | 41 | 42 |
| No router (seed 79) | 49 | 28 | 42 | 44 |
| No memory (seed 79) | 48 | 24 | 40 | 48 |

- **Better, not enough.** Toggle rises by 7–12 cells at each seed, and
  pick up and drop are nearly right. Each stored combination is
  predicted right on 94–100% of its rows with that combination hidden.
- **The errors show no colour relation learned.**
  - Seeds 79 and 81 treat the blue door as opening with any key, and
    with nothing held.
  - Seed 80 predicts that no door opens even with its own key.
  - The network with no memory gets the same 48 cells as with memory.
  
  So memory adds nothing on toggle.

## 8. Decision

**Stop**, as declared for a second failed gate. With these two fixes the
network fits pick up and drop from memory, but it still does not learn
from about 36 door and key combinations that a key opens the door it
matches in colour. Learning the relation inside the reasoner, from one
world's few examples, is the step that fails. The next card, for the
user to approve, gives the reasoner card 070's already-learned relation,
trained with the encoder, and asks the reasoner only to read rules from
the tries it recalls.
