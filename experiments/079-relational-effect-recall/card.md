---
id: "079"
title: a relational router as toggle recall's metric, for a key never seen opening its door
rung: 6
serves: [P3, P10, P6]
status: done
verdict: fail
arch_version: 18
date: 2026-10-06
---

# 079: a relational router for effect recall

The user's choice (2026-10-06, after cards 077–078): take the relational
router to effect recall, where the agent's transfer failures have been.
Report only; plugging it into the agent is a later card.

## 1. Question

In card 070's decoy folds (one colour's door openings removed from
memory), version 18's recall gets 35 of 36 door-and-key colour pairs
right in every fold, and the one it gets wrong is always the left-out
colour's own pair: it predicts the door will not open
(`runs/074.2/decoy_*.json`). Its weights favour the door's identity
about 40 times the relation (card 070), and memory holds only failures
at that door. If toggle recall's metric is a router that reads the tiles
only through what they do under the other actions and card 070's
relation between them, does it predict the left-out pair opens, without
new errors? P3 (a known relation applied to new participants), P10, P6.

## 2. What changes

One component: toggle recall's similarity between a query and stored
tries (the level after "own tries first", which stays: an identical
stored try is still the evidence).

| | Version 18 (cards 049–050, 070) | This card |
|---|---|---|
| A stored try | front tile, held tile, view set; outcome counts | the same |
| Similarity | admitted conditions: the front tile's four parts, rel:P, …, each with a weight | a **router** embedding: per tile its role under the *other* actions (forward moves onto it; a pick up changes it, empty-handed), card 070's **\|P(z_front − z_held)\|**, and for each tile in view \|P(z_v − z_front)\|, \|P(z_v − z_held)\| and its role, summed; no tile's vector read directly; stored tries count by exp(−‖w ⊙ (e_q − e_j)‖₁) |
| Fit | leave-one-out likelihood of stored outcomes | the same, leaving out every try with the query's (front, held) combination (card 071), so the fit is rewarded only for carrying to combinations it has not seen |

Roles under the other actions are what the agent's own recall predicts
(card 077's affordances), so "a door" is "a tile that cannot be walked
onto or picked up", not a code. Literature: Webb et al., the relational
bottleneck; card 070 (a relation-only path carried to unseen hues).

## 3. Dependencies

Card 070's encoder, P and decoy folds (memory with every hue, less the
fold's openings; `run.py decoy --hold opening`); card 069's recall
table; card 077's roles.

## 4. Data check

Per fold: stored toggle tries (distinct keys), how many open a door, and
how many involve the fold's door.

## 5. Feasibility gate

- **Upper bound:** the router with nothing left out (the fold's
  openings kept) predicts every cell of the table right.
- **Trivial baseline:** version 18's recall (0 of 6 fold pairs).

## 6. Success criteria and prediction

The table (per fold, from the decoy world's start view, the evaluator's
truth): the six locked doors × held {six keys, nothing}, and a wall and
the floor in front × the same; a door opens only with its own key.

1. **Transfer:** the left-out pair predicted to open in 6 of 6 folds
   (version 18: 0 of 6).
2. **No new errors:** every other cell right in every fold.

Also reported: version 18's recall on the whole table, the router's
leave-one-combination-out accuracy on stored tries, time per query.

**Prediction.** With identity unreadable, the relation decides, and it
separates matching pairs on unseen hues (card 070), so 6 of 6. The risk
is the wall: it shares a locked door's role, and only its relation to
the key tells them apart.

**Decision rules.** Keep (next card: the router inside the agent's
toggle recall, on the decoy folds with trying and the tiers) if 1 and 2
hold. Revise if only 2 fails. Stop if 1 fails.

**Budget.** Six fold setups of about 70 s, a fit of about 30 s each.

## 7. Result

`runs/079/` (`tools/card079/effects.py`, one process per fold). The table
has 56 cells (six locked doors, a wall and the floor, each with six keys
or nothing held).

**Data check.** Per fold: 1,280 distinct stored toggle tries in 129
(front, held) combinations, 25 of which open a door (30 with nothing held
out: 5 per colour; the red fold removed 77 opening tries); 10–14 at the
fold's door, all failures.

**Gate.** With nothing held out, the router gets all 56 cells right (and
so does version 18).

| Fold | Own pair's relation \|P(z_door − z_key)\|₁ | Version 18: the pair | Router: the pair | Router's other errors |
|---|---|---|---|---|
| red | 1.80 (largest of the six) | no | no | 0 |
| green | 1.11 | no | **yes** | 1 (purple door, blue key) |
| blue | 1.37 | no | **yes** | 3 (two own pairs missed, one mismatch) |
| purple | 0.76 (smallest) | no | no | 0 |
| yellow | 1.40 | no | **yes** | 0 |
| grey | 0.93 | no | **yes** | 0 |

- **Criterion 1: not met** (4 of 6; version 18: 0 of 6).
  **Criterion 2: not met** (new errors in the green and blue folds).
- **Why the two folds fail.** The router carries "fits" to a left-out
  pair whose relation lies inside the range of the pairs it trained on
  (0.93–1.40), and not to the two outside it: red's 1.80 is above every
  other own pair, purple's 0.76 below. A vote over stored tries weighted
  by similarity interpolates between remembered values; it does not
  extrapolate "the closer the relation, the more it fits" beyond them,
  so a pair outside the range falls back on the prior (nothing happens).
  The new errors in the green and blue folds are probably the same vote
  drawing a boundary between remembered values in the wrong place (not
  traced; one fit per fold).
- Version 18's recall fails all six for another reason, the door's
  identity outweighing the relation (card 070); the router, with no
  identity to read, fixes that in the four interior folds.
- Cost: a fit of about 3 s; 0.05 ms per query.

## 8. Decision

**Stop** (criterion 1). Reading tiles only through roles and card 070's
relation removes the identity bias that makes version 18 fail every fold,
but a similarity vote over remembered relations cannot extrapolate along
the relation. Carrying "fits" to a new pair outside the remembered range
needs the relation to enter as an ordered quantity, where closer means
more likely, for example a monotone readout of the relation beside the
vote; that is a new card for the user.
