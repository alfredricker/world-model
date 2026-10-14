---
id: "080"
title: the relation as an ordered quantity, a monotone readout as the vote's prior
rung: 6
serves: [P3, P10, P6, P21]
status: done
verdict: fail
arch_version: 18
date: 2026-10-06
---

# 080: the relation as an ordered quantity

Approved by the user (2026-10-06), after card 079.

## 1. Question

Card 079's router (roles and card 070's relation, no identity) predicted
a key never seen opening its door in 4 of 6 colour folds, and failed
exactly the two whose relation lies outside the remembered range (red
1.80 above every other own pair, purple 0.76 below): a similarity vote
interpolates between remembered values and does not extrapolate. If the
relation also enters a readout that is linear in it, so that "the closer,
the more it fits" holds beyond the remembered values, and that readout is
the vote's prior, does the left-out pair open in every fold, without new
errors? P3, P10, P6, P21 (a weight carrying a rule, the vote as
fallback).

## 2. What changes

One component: card 079's prior.

| | Card 079 | This card |
|---|---|---|
| Vote | the router's similarity vote over stored tries, other (front, held) combinations | the same |
| Prior | the overall outcome rates, weight a | a **readout**: per category, a logit linear in r = \|P(z_front − z_held)\|₁ for each combination of the two tiles' roles (logit_c = u_c · roles + (v_c · roles + v0_c) · r + b_c), so monotone in r; its weight β |
| Combination | (Σ k C + a · rates) / (Σ k n + a) | (Σ k C + β · readout) / (Σ k n + β) |
| Fit | leave-combination-out likelihood | the same, router, readout and β together |

β is fitted on held-out combinations, so the data decide how far the
readout speaks; extreme combinations (the smallest and largest relations)
are held out in training too. The back-off follows card 050 (own tries,
then neighbours as their prior) and MacKay and Peto 1995 (LITERATURE.md).
A network inside a component, used where it improves on the current
mechanism in a test (CHARTER, One encoder, as amended 2026-10-06).

## 3. Dependencies

Card 079's folds, table, roles and router (`tools/card079/effects.py`).

## 4. Data check

Card 079's: 1,280 stored toggle tries per fold, 25 opening a door.

## 5. Feasibility gate

With nothing held out, every cell right (as card 079).

## 6. Success criteria and prediction

Card 079's table and criteria, judged on the fit with card 079's seed
(79); seeds 80 and 81 reported for the spread between fits:

1. **Transfer:** the left-out pair predicted to open in 6 of 6 folds.
2. **No new errors:** every other cell right in every fold.

Also reported: the readout alone and card 079's vote, per fold and seed;
the fitted β; the readout's slope on r for a locked door and a key.

**Prediction.** The readout's slope makes small relations open doors
beyond the remembered range, and the held-out extremes in training give
β enough weight for it to decide where the vote is far from every stored
try. Risk: where the vote is confident and wrong (card 079's red fold:
P(nothing) = 1.0), a prior cannot overturn it.

**Decision rules.** Keep (next: the router with this prior inside the
agent's toggle recall, on the decoy folds and the tiers) if 1 and 2
hold; revise if only 2 fails; stop if 1 fails.

**Budget.** Six fold setups of about 70 s; nine fits of a few seconds
each per fold.

## 7. Result

`runs/080/` (`tools/card080/ordered.py`, one process per fold; nine fits
per fold, three arms × seeds 79, 80, 81). Each cell: the left-out pair
opens (yes/no) / other errors out of 55.

| Fold (pair's relation) | Vote 79 / 80 / 81 | Readout 79 / 80 / 81 | Both 79 / 80 / 81 |
|---|---|---|---|
| red (1.80) | no/1, no/3, no/6 | **yes/2, yes/0**, no/5 | no/1, no/4, yes/11 |
| green (1.11) | yes/2, yes/5, yes/11 | **yes/2, yes/0**, no/5 | yes/6, yes/2, yes/13 |
| blue (1.37) | yes/3, yes/2, yes/6 | **yes/2, yes/0**, no/5 | yes/3, yes/2, yes/15 |
| purple (0.76) | yes/3, yes/6, yes/13 | **yes/2, yes/0**, no/5 | yes/1, yes/5, yes/17 |
| yellow (1.40) | yes/1, yes/7, yes/7 | **yes/2, yes/0**, no/5 | yes/3, yes/8, yes/16 |
| grey (0.93) | no/5, no/4, yes/9 | **yes/2, yes/0**, no/5 | no/5, no/2, yes/13 |
| nothing held out (errors of 56) | 2, 8, 7 | 2, **0**, 6 | 2, 6, 10 |

- **Gate: not passed** for the judged arm: with nothing held out, the
  vote with the readout as prior (seed 79) gets 2 cells wrong. The gate
  ran in the same pass as the folds, not before them.
- **Criterion 1: not met** (seed 79, both: 4 of 6; red and grey fail).
  **Criterion 2: not met** (1–6 other errors in every fold).
- **The readout alone transfers.** A logit linear in the relation, per
  combination of the two tiles' roles, opens the left-out pair in 6 of 6
  folds at seeds 79 and 80, and at seed 80 makes no other error in any
  fold or with nothing held out. Its slope on the relation for a locked
  door and a key is negative ("closer, more likely to open": −1.9 to
  −2.1 at seeds 79 and 80).
- **But its fit is unstable.** Seed 79 confuses the blue door with the
  purple key, both ways, in every fold and with nothing held out (their
  relation is small); seed 81 opens no own pair anywhere, nothing held
  out included: a fit failure, not a transfer failure.
- **As prior, the vote overrules it.** The fitted weight of the prior
  is about 0.35 tries against vote sums of many tries, so where the vote
  is confident and wrong (red, grey) the readout cannot correct it, and
  the readout's own errors add to the vote's. The risk the card named.
- **Card 079's explanation was partly wrong.** Refitted here, the vote
  opens purple's pair in 3 of 3 seeds (card 079: no) and grey's in 1 of
  3 (card 079: yes), and makes 1–13 other errors per fold. Card 079's
  single fit was noise in two folds; only red (above every other own
  pair) fails in every seed. Its "outside the range" reading holds for
  red, not purple; a † note is added there.
- Cost: nine fits of about 3 s per fold.

## 8. Decision

**Stop** (criterion 1, and the gate, for the vote with the readout as
prior). What the result supports: a readout linear in the relation, per
role combination, is the part that carries "fits" to a pair never seen
opening, including outside the remembered range, where no vote did; it
needs a fit that is stable, and it should decide where it agrees with
memory rather than be outweighed by the vote. CHARTER's System 1 rule
(weights take over only where they agree with recall on every case) is
the principled form of that: a card for the user.
