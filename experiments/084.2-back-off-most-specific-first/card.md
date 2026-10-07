---
id: "084.2"
title: conditions over roles and relations, with the back-off dropping the most specific condition first
rung: 6
serves: [P3, P12, P8, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 084.2: back-off, most specific condition first

Card 084's second revision, approved by the user (2026-10-07: "run the
fallback instead of 3-hue", in place of the three-hue card the user had
first set as 084.2).

## 1. Question

In cards 084 and 084.1 the door's identity was admitted before card
070's relation, and the back-off dropped conditions in the reverse
order of admission. So the cells a query fell back to were "this door
with a small relation", then "this door", and no cell held other
colours' openings without identity: the left-out pair opened in 0 of 6
folds. If the back-off instead drops the most specific condition first
(the one with most values among the stored tries), "a small relation"
becomes a cell of its own below "this door with a small relation". Does
the left-out pair then open, with every other cell right? P3, P12, P8,
P19.

## 2. What changes

One thing against card 084 (evidence over every try, as card 084): the
order of the back-off.

| | Card 084 | This card |
|---|---|---|
| Levels, finest to coarsest | all admitted; then drop the condition admitted last, …; neighbours | all admitted; then drop the condition with most distinct values among stored keys (identity before a cut or a role), …; neighbours |

Admission, candidates, evidence, cells, own tries first, γ, β and the
tables are card 084's. The diagnostic arm (evidence on the category) is
reported again. Specificity by number of values is a declared prior
(C3), the same for every action and world.

## 3. Dependencies

Card 084 (`tools/card084/lifted.py`, with `WM_ORDER=specific`).

## 4. Data check

As card 084, plus the levels in order and, for the left-out pair, the
cell it falls back to and what it holds.

## 5. Feasibility gate

As card 084: nothing removed, every cell of the three tables right.

## 6. Success criteria and prediction

Card 084's three criteria, unchanged:
1. **Transfer:** the left-out pair opens in 6 of 6 folds, with every
   other cell of the three tables right.
2. **Tier 2's hand:** every cell of card 084's two tier 2 tables right.
3. **Own tries still overrule:** one failed try of the left-out pair
   makes it predicted to fail.

**Prediction.** Toggle's levels become (identity, cut), then (cut),
then neighbours. The left-out pair's identity cell is empty, so it
falls to "relation below 1.96". The risk is what that cell holds:
toggle admitted no role, so it also holds pairs that are not a door and
its key (a key on the floor with a key of its colour held), and if
those tries outnumber the openings it predicts "nothing", and criterion
1 fails. Red's pair (1.80) also falls above the red fold's cut (1.58).

**Decision rules.** As card 084: keep if 1–3 hold; revise if 1 holds and
2 or 3 fails; stop if the gate or 1 fails.

**Budget.** About 10 minutes, as card 084.

## 7. Result

`runs/084/*_084.2.{json,log}` (`tools/card084/lifted.py` with
`WM_ORDER=specific WM_TAG=_084.2`); about 10 minutes, and the green fold
rerun once to record what the fallback cell holds.

**Gate: passed** (56, 28 and 42 cells right). Admission is card 084's
(toggle: front identity, then rel:P > 1.96); the levels are now (cut,
identity), then (cut), then neighbours.

| Fold | Left-out pair (both arms) | P(change) in its fallback cell | Other cells |
|---|---|---|---|
| red | no; its relation (1.80) falls above the fold's cut (1.58) | 0.000 | all right |
| green, blue, purple, yellow, grey | no | 0.002 | all right |

Tier 2: toggle 3 of 3, pick up 7 of 9 (card 084's). Criterion 3:
trivially met.

**What the fallback cell holds** (green fold, relation below 1.96,
toggle tries):

| Front, held | Changed | Unchanged |
|---|---|---|
| floor, floor (the empty hand is drawn as floor) | 0 | 221,096 |
| each locked door with its own key (5 colours) | 69–88 each, 394 in all | 0 |
| the closed blue door with the blue key | 37 | 0 |

- **Criterion 1: not met. Criterion 2: not met.** Criterion 3:
  trivially met.
- **The relation is clean; the cell is not.** Below the cut, every
  door with a key opened and nothing else did, except toggling the
  floor with an empty hand, whose relation is 0 because the empty hand
  looks like the floor. Those 221,096 tries outweigh the 431 openings.
  The front tile's role "forward moves onto it" separates them, but it
  was never admitted: once identity is in, a role adds nothing to the
  evidence.

## 8. Decision

**Stop** (criterion 1). Across cards 084–084.2 the rule is in memory
and the evidence finds its cut, but greedy admission takes identity
first, and identity makes every condition without a constant redundant.
The change these results point to is the order of admission, not of the
back-off: conditions without constants (roles, the relation's cut)
admitted first, and identity only for what they leave unexplained (the
bias of Popper and the Apperception Engine, which allow constants only
when needed). With this card's back-off, toggle would then have the cell
"relation small and cannot be walked onto", which holds only openings.
For the user to decide.
