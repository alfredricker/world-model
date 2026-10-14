---
id: "067"
title: closeness by propagation over the believed map
rung: 1
serves: [P17, P12, P21, P18]
status: done
verdict: pass
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

Approved by the user on 2026-10-05, to be decided on tiers 1 and 2: tier
3's memory cannot be built yet (recall's weight fit compares every pair of
stored views at once, 698 GB there; a later card), so tier 3 counts as no
different. Card 066's seeds: 200 episodes of tier 1, 100 of tier 2; an
episode still running after 5 minutes of wall-clock is stopped and counted
as a failure, for both versions alike (reported).

1. **Tier 1:** success ≥ 99% (CHARTER).
2. **Tier 2:** not worse than version 14 (paired over the same seeds,
   McNemar's test, p < 0.05).
3. **Cost:** nothing is kept per pair of placements, and the work per
   field is linear in the map: on average at most 4 relaxations per move
   edge of the field, over every field of tiers 1 and 2. Reported:
   time per step, fields per step, placements per field.

**Prediction.** Tier 1 as version 14. Tier 2 no different: both meet card
058's "clear the way" problem (a ball in front of the door). Tier 3
better or no different: version 14's walking reads System 1's V across
the whole 16 × 16 map, beyond the 12 tiles it was trained on, and its
pair tables grow with the square of the lattice.

**Decision rules.** Keep if all three hold. Revise once if criterion 2
fails on one tier with a diagnosis in walking. Stop if tier 1 falls below
99%.

**Budget.** Tier 1 about a minute; tier 2 at most about 25 minutes (100
episodes on 20 workers, 5 minutes at most each).

## 7. Result

`runs/067/tier1.json`, `runs/067/tier2.json`; version 14's from card 066,
on the same seeds.

| | Tier 1: version 14 | Tier 1: this card | Tier 2: version 14 | Tier 2: this card |
|---|---|---|---|---|
| Success | 100% (200) | **100%** (200) | 0% (100) | **0%** (100) |
| Episodes one solved and the other did not | – | 0 | – | 0 |
| Steps when successful | 20.8 | 20.8 (199 of 200 the same; one 2 shorter) | – | – |
| Time per step | 2.9 ms | 2.1 ms | 65 ms | 55 ms |
| Setup (memory, recall, walking) | 47 s | 21 s | 171 s | 143 s |
| Relaxations per move edge | – | 1.00 | – | 0.93 |
| Fields per step; placements per field | – | 1.1; 80 | – | 1.6; 63 |
| Time per field | – | 0.27 ms | – | 0.28 ms |

- **Criterion 1: pass,** 200 of 200.
- **Criterion 2: pass.** Tier 2: no episode solved by one version and
  not the other (McNemar's test, p = 1); both stop in the first room for
  card 066's reasons, none of them walking's. Tier 3: no different (its
  memory cannot be built for either).
- **Criterion 3: pass.** Nothing is kept per pair of placements; each
  move edge is relaxed 0.93–1.00 times per field, against the bound of 4.
  Walking is 14% of the agent's time in tier 1 and under 1% in tier 2,
  where recall and the condition search take the rest.

## 8. Decision

**Keep: version 15.** Walking takes each step from a closeness worked out
over the agent's own believed map, with the same decisions as version 14
where both can be compared and nothing stored per pair of placements; the
token lattice can now be sized to the map (card 066's declared prior).
What this card could not show: walking on a large map, since tier 3's
memory cannot be built yet.
