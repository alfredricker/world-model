---
id: "084.1"
title: conditions over roles and relations, with evidence counted once per stored situation
rung: 6
serves: [P3, P12, P8, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 084.1: evidence per stored situation

Card 084's revision, approved by the user (2026-10-07: "let's do 84.1
first, if it fails, we can try 3-hue as 84.2").

## 1. Question

In card 084 the door's identity was admitted before card 070's
relation, so the relation's correct cut (1.96, between 1.80 and 2.94)
became a rule per door, and the left-out pair opened in 0 of 6 folds.
The evidence there counted every stored try, about 528,000 per action,
so the cost of identity's many cells was negligible beside its
likelihood gain. Under the agent's own repeated play, only a prior
separates a rule from per-colour facts (Tenenbaum and Griffiths). If
each distinct stored situation counts once, does the relation come
before identity, and does card 084's test pass? P3, P12, P8, P19.

## 2. What changes

One thing against card 084: the counts the evidence reads.

| | Card 084 | This card |
|---|---|---|
| Evidence for admission | the stored outcome counts of every key (every try) | each stored key's outcome counts scaled to sum to 1: one situation, one count |

Everything else is card 084's: the candidates, the greedy admission
and its cost, the back-off chain in order of admission, the prediction
(raw counts in the cells, own tries first, γ and β), and the tables.
The diagnostic arm (evidence on the category, not the outcome class)
is reported again.

## 3. Dependencies

Card 084 (`tools/card084/lifted.py`, with `WM_EVIDENCE=key`).

## 4. Data check

As card 084, plus the distinct stored keys per action (about 1,300).

## 5. Feasibility gate

As card 084: nothing removed, every cell of the toggle (56), pick-up
(28) and drop (42) tables right.

## 6. Success criteria and prediction

Card 084's three criteria, unchanged:
1. **Transfer:** the left-out pair opens in 6 of 6 folds, with every
   other cell of the three tables right.
2. **Tier 2's hand:** every cell of card 084's two tier 2 tables right.
3. **Own tries still overrule:** one failed try of the left-out pair
   makes it predicted to fail.

**Prediction.** With about 1,300 counts, identity's dozens of cells
cost tens of bits each against a smaller gain, so toggle admits the
front tile's roles and the relation's cut before identity, and
criterion 1 holds. Risk: identity is still the only candidate that
tells locked doors from closed doors and walls, so it may still come
first; then criterion 1 fails as in card 084, and card 084.2 (three
more hues in memory) follows, as the user set.

**Decision rules.** As card 084: keep if 1–3 hold (next: into the agent,
decoy folds with trying and the three tiers); revise if 1 holds and 2
or 3 fails; stop if the gate or 1 fails.

**Budget.** As card 084: about 10 minutes (seven fold setups and tier
2's, three at a time).

## 7. Result

`runs/084/*_084.1.{json,log}` (`tools/card084/lifted.py` with
`WM_EVIDENCE=key WM_TAG=_084.1`); about 10 minutes.

**Gate: passed** (56, 28 and 42 cells right; version 18 the same).

**Admitted** (gate; the same in every fold):

| Action | Declared arm (outcome class) | Diagnosis (category) |
|---|---|---|
| Toggle | front identity (772 bits); nothing else | front identity (656), rel:P > 1.96 (52) |
| Pick up | front identity, held "forward moves onto it" | the same |
| Drop | held identity, front "forward moves onto it" | front "forward moves onto it", rel:P > 7.50, front "toggle changes it" |

The cost of a condition is about 7 bits; each action has about 1,300
distinct stored keys.

| | Card 084 | This card |
|---|---|---|
| Left-out pair opens (6 folds) | 0 of 6 | **0 of 6** (both arms) |
| Other toggle, pick-up and drop cells | all right | all right |
| Tier 2 toggle (of 3) | 3 | 3 (version 18: 2) |
| Tier 2 pick up (of 9) | 7 | **5** (version 18: 6); diagnosis 7 |
| One failed try (criterion 3) | already "no" | already "no" |

- **Criterion 1: not met. Criterion 2: not met** (pick up 5 of 9:
  the box with something held, a ball with a ball held, a key with a
  key held). **Criterion 3:** trivially met.
- **Counting each situation once did not change the order.** Toggle's
  front identity is still admitted first, and in the declared arm the
  relation no longer pays its cost at all: the outcome class names the
  tile a door becomes, which identity already predicts. P19's curve
  (memory with 1, 2, 3 or 5 colours) admits identity first, and only
  identity, at every point.
- **What this shows.** Identity is first because it is the one
  candidate that tells locked doors, closed doors and walls apart, and
  that holds however the evidence is weighted and however many colours
  memory holds.†  The relation can only enter inside identity's cells,
  and the back-off in order of admission then keeps the rule per door.

## 8. Decision

**Stop** (criterion 1). The weight of the evidence was not the cause;
the order of the back-off is. Card 084.2 (three more hues in memory)
was set by the user to follow a failure, but this result predicts it
fails the same way: identity was admitted first with one colour as with
five, so a few more colours will not move it. The change the two cards
point to is the back-off: cells that drop identity before the relation
(most specific condition first), so that "a door and a key whose
relation is small" exists as a cell of its own. The choice is the
user's.

† Correction (card 084.3): identity is not the only such candidate.
Admitted before identity, roles and cuts on the relation also tell them
apart, and identity was then admitted for no action. It is the one
condition that does so at once, which is why greedy admission took it
first.
