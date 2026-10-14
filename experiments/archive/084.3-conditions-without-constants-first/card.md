---
id: "084.3"
title: conditions over roles and relations, those that name no particular tile admitted first
rung: 6
serves: [P3, P12, P8, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 084.3: conditions without constants first

Card 084's third revision, approved by the user (2026-10-07: "Sure, try
084.3").

## 1. Question

In cards 084–084.2 the relation's cut was found and is clean: below
1.96, every door with a key opened and nothing else did, except toggling
the floor with an empty hand (the empty hand is drawn as floor). But
greedy admission took the door's identity first, and after identity no
condition that names no particular tile adds evidence: the role that
would set the floor apart ("forward moves onto it") was never admitted,
so the fallback cell "relation small" held 221,096 floor toggles beside
431 openings. If conditions that name no particular tile (roles and
cuts) are admitted first, and identity and view tuples only for what
they leave unexplained, does the left-out pair open in every fold? P3,
P12, P8, P19.

## 2. What changes

One thing against card 084.2 (whose back-off drops the condition with
most values first): the order of admission.

| | Cards 084–084.2 | This card |
|---|---|---|
| Admission | greedy over every candidate at once | two stages: greedy over roles and cuts only, until none pays its cost; then greedy over every remaining candidate (identity and view tuples among them) |

The cost per condition stays log(number of all candidates). Evidence
over every try, the back-off, the cells, own tries first, γ, β and the
tables are card 084.2's. This is the bias of Popper and the
Apperception Engine, which allow a constant only when rules without one
cannot explain the data. "Names no particular tile" is a declared prior
(C3): a role or a cut reads what a tile does or how two tiles relate,
an identity or a view tuple reads which tile it is.

## 3. Dependencies

Card 084.2 (`tools/card084/lifted.py`, with `WM_STAGES=1
WM_ORDER=specific`).

## 4. Data check

As card 084.2, plus, per fold, the cell the left-out pair reaches once
identity is dropped (what it holds), and the stored keys within 0.5 of
each admitted cut.

## 5. Feasibility gate

As card 084: nothing removed, every cell of the three tables right.

## 6. Success criteria and prediction

Card 084's three criteria, unchanged:
1. **Transfer:** the left-out pair opens in 6 of 6 folds, with every
   other cell of the three tables right.
2. **Tier 2's hand:** every cell of card 084's two tier 2 tables right.
3. **Own tries still overrule:** one failed try of the left-out pair
   makes it predicted to fail.

**Prediction.** Toggle admits a role that sets the floor apart and the
cut near 1.96 in the first stage, and identity in the second (for
closed and open doors, and the result tile). The left-out pair's
identity cell is empty, so it falls to "relation small, not walked
onto", which holds only openings: criterion 1 holds in five folds.
Risks:
- **Red** (1.80, above every other own pair): in card 084 the red
  fold's cut was 1.58, below it. If a stored key between 1.40 and 1.80
  pulls the cut down again, red fails. The keys near the cut are
  reported.
- **Gate:** without identity first, a cell may mix outcomes that only
  identity separated; the second stage should still admit it.
- **Tier 2's pick up:** whether "the hand holds something that can be
  picked up" enters before the cut that stood in for "the hand is
  empty".

**Decision rules.** As card 084: keep if 1–3 hold (next: into the
agent, decoy folds with trying and the three tiers); revise if 1 holds
and 2 or 3 fails; stop if the gate or 1 fails.

**Budget.** About 10 minutes, as card 084.

## 7. Result

`runs/084/*_084.3.{json,log}` (`tools/card084/lifted.py` with
`WM_STAGES=1 WM_ORDER=specific WM_TAG=_084.3`); about 12 minutes.
**Deviation:** the script capped admission at 12 conditions, a limit set
in card 084's code and never reached before. Toggle reached it in the
first stage, so the second never ran; the cap was raised to 64 and the
gate rerun, so that the procedure declared in section 2 ran in full.

**Gate: passed** (56, 28 and 42 cells right, both arms). Identity is
admitted for no action: roles and cuts on the relation explain every
table. Toggle admits the held and front tiles' "forward moves onto it"
and cuts at 2.53 and 6.04 first, then many finer cuts (25 conditions in
the declared arm, 16 in the diagnosis; one view tuple in the second
stage of the declared arm).

| Fold | Left-out pair opens: declared arm (P change) | Diagnosis (category) | Its fallback cell holds | Other cells |
|---|---|---|---|---|
| red (1.80) | no (0.36) | **yes** (0.53) | only openings (37; 422) | all right |
| green | no (0.36) | **yes** (0.55) | only openings (81; 444) | all right |
| blue | no (0.33) | **yes** (0.52) | only openings (69; 411) | all right |
| purple | no (0.33) | **yes** (0.53) | only openings (81; 420) | all right |
| yellow | no (0.42) | **yes** (0.55) | only openings (125; 430) | all right |
| grey | no (0.32) | **yes** (0.53) | only openings (79; 418) | all right |

Tier 2, both arms: pick up **9 of 9** (version 18: 6), toggle **3 of 3**
(version 18: 2). Criterion 3: declared arm trivially (already "no");
diagnosis met (opens before the failed try, not after, in every fold).

- **Criterion 1: not met** in the declared arm (0 of 6). The diagnosis
  meets it (6 of 6, every other cell right). **Criterion 2: met.**
  **Criterion 3:** met (trivially in the declared arm).
- **The structure is found.** With roles and cuts first, the cell the
  left-out pair reaches holds only openings in every fold and arm: "the
  held tile and the tile in front cannot be walked onto, and their
  relation is small". Red's pair (1.80) now falls below the cut (2.53).
- **What fails is how much the cell counts.** γ, fitted by leaving one
  try out as β is, is 8,103 for toggle in every fold and arm, the top
  of its grid (e^9): a cell of 81 tries carries about 1% of its level's
  weight, and the coarser cells, mostly failures, pull P(change) to
  0.32–0.42 in the declared arm and 0.52–0.55 in the diagnosis. With
  β near zero, γ is decided almost only by stored situations with a
  single try, and for those leaning on the coarser cells and version
  18's neighbours scores better than the fine cells (not traced further).
  In the declared arm the cells are finer (cuts that isolate one
  colour's pair, since the outcome class names the tile the door
  becomes), so they are diluted more.
- P19's curve: identity is admitted only with 5 colours in the
  declared arm (second stage), never in the diagnosis.

## 8. Decision

**Stop** (criterion 1, declared arm). The order of admission was the
obstacle the earlier cards named: with conditions that name no tile
first, the rule is a cell of its own in every fold, tier 2's hand table
is right for the first time, and the diagnosis passes all three
criteria. It does so by a thin margin, against a γ fitted to the wrong
question: the cells exist to answer combinations memory has not seen,
but γ is fitted by leaving one try out, which asks about combinations
it has. The next card, for the user, fits γ by leaving whole (front,
held) combinations out (card 071's unit).
