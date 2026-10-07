---
id: "084.4"
title: conditions over roles and relations, the cells' weight fitted by leaving whole combinations out
rung: 6
serves: [P3, P12, P8, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 084.4: γ by combinations left out

Card 084's fourth revision, approved by the user (2026-10-07: "yes run
084.4").

## 1. Question

In card 084.3, with conditions that name no tile admitted first, the
left-out pair's cell held only openings in every fold, but γ (how much
a cell counts against the coarser cells above it), fitted by leaving
one try out, sat at the top of its grid (e^9 = 8,103). The cell then
carried about 1% of the weight, and the pair opened in 0 of 6 folds in
the declared arm (P(change) 0.32–0.42; the diagnosis 0.52–0.55). The
cells exist to answer combinations memory has not seen, so γ should be
fitted on that question. Fitted by leaving whole (front, held)
combinations out, does the left-out pair open in every fold? P3, P12,
P8, P19.

## 2. What changes

One thing against card 084.3: the fit of γ.

| | Card 084.3 | This card |
|---|---|---|
| γ fitted by | leaving one try out (as β, card 050) | leaving out every try of the try's (front, held) combination: from its own situation (then empty), from every cell and from version 18's neighbours (card 071's unit) |

The tries are weighted as stored, as for β; the grid is β's. Admission
in two stages, the back-off most specific first, evidence over every
try, own tries first, β and the tables are card 084.3's. The diagnostic
arm (evidence on the category) is reported again.

## 3. Dependencies

Card 084.3 (`tools/card084/lifted.py`, with `WM_STAGES=1
WM_ORDER=specific WM_GAMMA=combination`); card 071's unit.

## 4. Data check

As card 084.3, plus both γ (one try out, combinations out) and the
held-out likelihoods at each.

## 5. Feasibility gate

As card 084: nothing removed, every cell of the three tables right.

## 6. Success criteria and prediction

Card 084's three criteria, unchanged:
1. **Transfer:** the left-out pair opens in 6 of 6 folds, with every
   other cell of the three tables right.
2. **Tier 2's hand:** every cell of card 084's two tier 2 tables right.
3. **Own tries still overrule:** one failed try of the left-out pair
   makes it predicted to fail.

**Prediction.** γ falls far below e^9: a combination never seen is
predicted better by its fine cell than by version 18's neighbours,
which weigh the door's identity. The pair's P(change) rises toward 1 in
both arms and criterion 1 holds. Risks:
- The fit is weighted by stored tries, so huge combinations (221,096
  floor toggles) may dominate it.
- With a small γ, table cells that have no own tries rely on fine cells
  of a few tries, which may break the gate.

**Decision rules.** As card 084: keep if 1–3 hold (next: into the
agent, decoy folds with trying and the three tiers); revise if 1 holds
and 2 or 3 fails; stop if the gate or 1 fails.

**Budget.** About 12 minutes, as card 084.3.

## 7. Result

`runs/084/*_084.4.{json,log}` (`tools/card084/lifted.py` with
`WM_GAMMA=combination WM_STAGES=1 WM_ORDER=specific WM_TAG=_084.4`);
diagnosis `runs/084/none_084.4diag.{json,log}` (`WM_GAMMA_DIAG=1`);
about 14 minutes.

**Gate: passed** (56, 28 and 42 cells right, both arms).

| | γ, one try out (card 084.3) | γ, combinations out (this card) |
|---|---|---|
| Toggle | 8,103 | 8,103 (red fold: 6,003 and 2,981) |
| Pick up, drop | 0.0001 | 8,103 |

8,103 is e^9, the top of the grid. Held-out log-likelihood of toggle's
tries: −20,829 nats with the cells at that γ, −64,169 with version
18's neighbours alone.

| Fold | Left-out pair opens: declared arm (P change) | Diagnosis (category) | Other cells |
|---|---|---|---|
| red | no (0.45) | **yes** (0.85) | all right |
| green, blue, purple, yellow, grey | no (0.32–0.42) | **yes** (0.52–0.55) | all right |

Tier 2, both arms: pick up 9 of 9, toggle 3 of 3. Criterion 3:
declared arm trivially; diagnosis met in every fold.

- **Criterion 1: not met** in the declared arm (0 of 6); the diagnosis
  meets all three criteria, as in card 084.3. **Criterion 2: met.**
  **Criterion 3:** met (trivially in the declared arm).
- **Why γ stays at the top** (diagnosis). The likelihood γ is fitted
  on is over outcome classes, and an outcome class names the tile a
  place becomes ("the hand holds the grey key", "the red door is
  open"). Left out, a combination's classes occur nowhere else in
  memory, so only version 18's neighbours, smoothed, give them any
  probability, and the largest γ wins. The combinations that pull
  hardest are picking up each key empty-handed (about 8,000 tries each),
  dropping each key on the floor, and, for toggle, each door with its
  key and each open door closed. Weighting every combination equally
  gives the same γ (8,103 for every action), so the size of the floor
  combinations was not the cause.

## 8. Decision

**Stop** (criterion 1, declared arm). Leaving combinations out asks the
right question, but scores it on the wrong quantity: what a cell must
predict for a combination never seen is whether and how the action
changes something (the category, which recall's prediction of an effect
reads at one half), while the tile it becomes is carried over by card
038's ways. A class that names the result can never be predicted from
other combinations. The next card, for the user, fits γ with
combinations left out on the category.
