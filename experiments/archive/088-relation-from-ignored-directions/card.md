---
id: "088"
title: the relation from the directions properties ignore, in place of a relation trained for colour
rung: 6
serves: [P3, P4, C5, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 088: the relation from ignored directions

Drafted at the user's request (2026-10-08: "replace the named color with
this 'ignored direction' principle"); CHARTER's "Patterns are
discovered". Runs after card 087, whose properties it reads, and
before card 086's main runs (the user, 2026-10-08).

Approved by the user (2026-10-08: "yes, they do").

## 1. Question

Recall's relation `rel:P = ‖P(z_front − z_held)‖₁` uses card 070's
projection P, trained with an objective that said "a key matches a door
of its colour": the attribute was named in advance. Card 087's property
networks must read what separates kinds (a key from a ball, an open door
from a closed one) and, trained on fresh hues, learn to ignore colour.
The directions in which tiles of one kind still differ, but which no
property reads, are then what varies within a kind: here colour, found
without being named. Does "equal along those directions", formed by the
agent from its own properties, do what card 070's P does: tell a matching
key and door from another for colours left out and hues never seen, and
give card 084.3's admission the cut that explains every opening with one
condition? P3, P4, C5, P19.

## 2. What changes

One component: where P comes from. Its use (`rel:P` in recall, card
084.3's cuts and admission, card 086's level) is unchanged.

```
card 070:  P trained so that same-colour pairs are near           (colour named by the objective)
this card: G  = Σ_properties E[J_f(z)ᵀ J_f(z)]                      what the properties read (J: f's Jacobian)
           C  = covariance of z among tiles of the same property profile    what varies within a kind
           P  = top r solutions of C v = μ (G + εI) v               within-kind, ignored by every property
```

- **Tiles** for G and C: property play's (card 087), fresh hues, no
  MiniGrid hue. A tile's profile is f's rounded outputs (walkable, ends,
  pick up, toggle); C pools the covariance within each profile.
- **Candidates, not one answer:** the leftover may hold more than one
  attribute, so `rel` over the top 1, 2, 4 and 8 directions are each a
  candidate, and card 084.3's evidence (cost log(candidates)) admits the
  one that explains stored tries most simply. Nothing names colour; a
  world where size or pattern varied within kinds would give those.
- **Unchanged:** the encoder (card 070's, frozen), recall, version 19's
  level when card 086 runs, the planner.

The encoder itself was trained with card 070's relation term, so colour
is still named once, inside it. Card 070's step A found the same relation
readable on card 054's encoder, trained without it (63 of 66, 8 of 9). If
this card is kept, moving to an encoder trained without the relation term
is the next card.

## 3. Dependencies

Card 087 (the property ensemble f, property play); card 070 (the encoder,
the hue sampler, card 069's probe of pairs); card 084.3 (`rel` cuts and
admission); card 086 (version 19's gate tables). The relational
bottleneck (Webb et al.; LITERATURE): a relation is a comparison in one
part. Invariance as what an outcome does not depend on: invariant causal
prediction (`1501_01332`, LITERATURE).

## 4. Data check

G's eigenvalues and C's within each profile; the share of C's variance
that hue explains (report only, a probe never read by the agent); that
keys' and doors' leftover directions line up (the angle between the top
directions computed from keys alone and from doors alone).

## 5. Feasibility gate

- **Upper bound:** card 070's trained P: 66 of 66 left-out-colour pairs,
  9 of 9 unseen-hue pairs (card 069's probe, only the threshold fitted).
- **Trivial baselines:** card 054's vectors under any weighting, 47 and
  4; P as random directions of the same rank; P from the directions the
  properties read most (G's top), which should fail.
- **Gate:** the leftover P at its admitted rank, on card 069's probe:
  ≥ 64 of 66 and ≥ 8 of 9 (card 070's gate).

**Gate result: failed** (`tools/card088/leftover.py`, `runs/088/gate.json`;
G and C from 20,000 property-play tiles and card 087's ensemble, seven
property profiles; about a minute).

| P (card 069's probe, only the threshold fitted) | Left-out colour pairs (≥ 64 of 66) | Unseen-hue pairs (≥ 8 of 9) |
|---|---|---|
| Card 070's trained P (upper bound) | 66 | 9 |
| Leftover, rank 1, 2, 4, 8 | 35, 41, **53**, 42 | 5, 6, **7**, 6 |
| Random directions, rank 1, 2, 4, 8 | 45, 46, 33, 32 | 6, 6, 5, 5 |
| G's top directions, rank 1, 2, 4, 8 | 42, 42, 39, 31 | 3, 5, 4, 5 |
| Card 054's vectors, best weighting (card 069) | 47 | 4 |

ε from 10⁻³ to 10 × trace(G)/32 changes nothing (best 53 of 66).

**Data check, which explains it.** The leftover is what the user
expected: hue explains 78% of the top four leftover directions' variance
among keys and 85% among locked doors. But keys' and locked doors'
leftovers point different ways: their top two directions are 70° and 88°
apart. This encoder codes colour separately for each kind (ARCHITECTURE,
"Parts mix kind and colour"), so a red key and a red door are not equal
along any leftover shared by both, and "equal along the leftover" cannot
compare across kinds. Card 070's P works because its objective aligned
the two.

## 6. Success criteria and prediction

1. **Recall's tables** (card 086's gate with this P in place of card
   070's): card 079's toggle table 56 of 56, pick up 28, drop 42; tier
   2's hand tables 9 of 9 and 3 of 3.
2. **The left-out pair:** in each of card 086's six colour folds, the
   cell the fold's pair reaches holds only openings, as with card 070's
   P (card 084.3).
3. **CHARTER's tiers** and card 086's lower bar on transfer, against the
   same version with card 070's P: no tier worse; the fold's door opened
   within its first two toggles.

**Prediction.** The leftover is mostly hue: property play varies hue and
little else within a kind. Risks: noise directions enter C (rank 8 picks
them up; the rank is admitted, which should prefer the smallest that
works); keys' and doors' colour directions do not line up on this
encoder (card 070's step A suggests they do).

**Decision rules.** Keep if the gate and 1–3 hold (card 070's P is then
retired). Revise if the gate passes and 2 fails. Stop if the gate fails
at every rank.

**Budget.** Gate a few minutes (G, C and the probe on stored vectors).
Criteria as card 086's, handed to the user.

## 7. Result

The gate failed at every rank (section 5); the criteria were not run.

## 8. Decision

**Revise** (the user, 2026-10-08: "keep 088"; the card's own rule said
stop). The principle found the attribute: what the properties ignore is
mostly hue (78–85%), without colour being named. It cannot yet turn it
into a relation, because this encoder codes colour in different
directions for each kind (70° and 88° apart for keys and doors), so
being equal along the leftover is not the same as having the same
colour. The direction stays; what changes is the encoder: card 089
trains it so that what is left after the kind means the same in every
kind, and this card's gate is then rerun on it. Card 070's P stays in
the agent until then.
