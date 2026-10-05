---
id: "067"
title: closeness by propagation over the believed map
rung: 1
serves: [P17, P12, P21, P18]
status: draft
verdict:
arch_version: 14
date: 2026-10-05
---

# 067: closeness by propagation over the believed map

Asked for by the user on 2026-10-05: "The model should only know where it is
stepping in the immediate and learn whether or not it is more likely this
step brings them closer to their desired destination." Run on CHARTER's
three MiniGrid tiers ([card 066](../066-minigrid-tiers/card.md)).

## 1. Question

Can walking take each step from a closeness worked out locally on the
agent's own believed map, with nothing kept for pairs of placements, and
do as well as version 14 on the three tiers at a cost that grows with the
map, not with its square? P17 (planning stays usable as the world grows;
Crafter's map is 64 × 64), P12 (a door on the way still becomes a
condition), P21 (walking's compute is reported), P18.

## 2. What changes

One component: walking (card 045's System 1 and System 2, card 051's
search over pairs of tokens).

| | Version 14 | This card |
|---|---|---|
| How far a placement is from another | System 1, a network trained on open space, read for every pair of placements once and kept | not kept: per situation and need, a closeness field |
| Which routes are clear | each approach's route tokens, worked out for every pair of placements and kept; chains of clear approaches over the standing placements (System 2) | the field itself: a forward step counts only onto a walkable token |
| A blocked way | each token memory has seen made walkable, counted walkable one at a time, then pairs of them (card 051), each giving a new chain | one field where those tokens can be crossed at one step more; the first token a least route crosses becomes the condition ("walk", j) |
| Kept per world | 676 placements: about 457,000 pairs of V, first moves and routes | nothing beyond card 044's placements and moves |

The field. Over the placements the agent could stand at (on a token
recall's forward kind says moves it), the need's placements are 0 and
every other placement is one more than the least of the placements its
three moves lead to (card 044's learned transformations give those).
Only the moves into a placement whose closeness just changed are updated,
so each move is relaxed a few times, not once per sweep (value iteration,
Bellman 1957, on the agent's own map as in VIN, `1602_02867`). The agent
takes a move that brings it closer, forward, then left, then right among
equals (card 023's order). If the first crossed token's condition cannot
be met, that token is left out and the next least route is taken.

**Which parts still imagine.** The field uses the move transformations to
find each placement's neighbours, as card 045's routes did once per pair;
no recall is called on an imagined state.

**Declared exception (rule 8, P21).** Version 14's System 1 for walking,
a trained network, is removed: the field is computed, not learned end to
end, from learned parts (walkability from recall, neighbours from the
learned moves). The user chose this over a learned closeness network
(card 012: learned walking 30% against 95% for exact walking). A network
may later take over where it agrees with the field on every case
(CHARTER, "weights take over from recall").

## 3. Dependencies

- Version 14 and the three tiers' harness, memory and seeds (card 066).
- Card 045's "exact" walking, the same search with sweeps over every
  placement, used there as the upper bound; the field here equals it
  (checked on 300 random maps with and without crossable tokens: 0
  disagreements, 0.85 relaxations per move edge).
- Card 044's transformations and placements; card 047's openable things.

## 4. Data check

Card 066's memory (random play, about 3.2 million steps per tier): doors
opened 1,565 (tier 1), 416 (tier 2), 10,571 (tier 3); keys taken out of
boxes 2,889 (tier 3).

## 5. Feasibility gate

- **Upper bound:** the tasks are solvable from every start (MiniGrid;
  tier 3 is the v1 environment, which fixes v0's unsolvable layouts).
- **Trivial baseline:** random actions, from card 066's memory: 3.3%
  (tier 1), 0.45% (tier 2), 5.9% (tier 3) of episodes succeed.
- **Smoke test** (before the main run): tier 1, 4 episodes: the same steps
  as version 14 on every seed (22, 28, 13, 18).

## 6. Success criteria and prediction

Card 066's seeds and episode counts; version 14's scores from card 066.

1. **Tier 1:** success ≥ 99% (CHARTER).
2. **Tiers 2 and 3:** not worse than version 14 (paired over the same
   seeds, McNemar's test, p < 0.05).
3. **Cost:** nothing is kept per pair of placements, and the work per
   field is linear in the map: on average at most 4 relaxations per move
   edge of the field, over every field of the three tiers. Reported:
   time per step, fields per step, placements per field.

**Prediction.** Tier 1 as version 14. Tier 2 no different: both meet card
058's "clear the way" problem (a ball in front of the door). Tier 3
better or no different: version 14's walking reads System 1's V across
the whole 16 × 16 map, beyond the 12 tiles it was trained on, and its
pair tables grow with the square of the lattice.

**Decision rules.** Keep if all three hold. Revise once if criterion 2
fails on one tier with a diagnosis in walking. Stop if tier 1 falls below
99%.

**Budget.** Set from card 066's timings (section 7 of that card).
