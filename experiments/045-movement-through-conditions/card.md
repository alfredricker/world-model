---
id: "045"
title: movement through conditions
rung: 0
serves: [P21, P12, P6, P17, C1, C5]
status: approved
verdict:
arch_version: 7
date: 2026-10-01
---

# 045: movement through conditions

## 1. Question

Walking is the last part of the planner that works by imagining steps
instead of reasoning over conditions (ARCHITECTURE.md, known limits). To
face a thing, version 7 moves closer by a hand-set distance (grid
distance, then turns) and checks every step by playing it forward in the
head with recall. When that fails it searches "move ways", one and two
moves deep, from every place the agent could stand. In card 044's run,
moving closer gave 79% of all moves and move ways 3–4%; walking took about
45% of planning time (card 038).

Suppose instead that walking has two levels, as GOAL.md P21 asks:
- **System 1, the approach.** A small network, trained on the agent's own
  moves, predicts how soon the agent can stand at any placement (a place
  and heading relative to the tokens) and which move to take.
- **System 2, its conditions.** The approach works only if every tile it
  will step onto is walkable. An unmet condition becomes a subgoal: make
  that tile walkable (open the door), or reach a waypoint from which the
  rest of the way is clear.

Does the agent reach its goals as before, in rooms it has never seen,
with no imagined step checked by recall? Before rung 1. Serves P21
(moving as System 1, obstacles as System 2 over conditions), P12 (how
soon a condition can be met), P6, P17 (planning cost), C1 (the distance
learned, not supplied) and C5 (new rooms).

## 2. What changes

One component, walking (ARCHITECTURE.md's components table). It replaces
card 041 (how soon only, which kept the imagined checks) and the planned
obstacles card.

```
version 7:  reach a placement facing u
            moving closer: grid distance, then turns (hand-set), each step imagined with recall
            if that fails: move ways one and two moves deep, from every standing placement (imagined)
            a door in the way: would imagined walking reach u if it opened?
card 045:   targets  P = placements standing on a walkable token with u ahead (as now)
            how soon V(g) = min_m Q(g, m), g = a target placement in the agent's frame   (System 1)
            route    the tiles the approach steps onto: Q's moves applied to g by the learned
                     transformations, with no recall
            condition every tile on the route is walkable (recall's forward kind, per token)
            unmet:   "tile v walkable" has achievers like any condition (toggle the door), or
                     a waypoint q: reach q, then reach p; q a standing placement with both
                     routes clear and V(g_q) + V(q → p) least
```

- **How soon (System 1).** Q has four inputs (the target's offset from
  the agent, and its heading relative to the agent's as a unit vector)
  and three outputs (left, right, forward). It is trained by
  fitted Q-iteration: Q(g, m) = 1 + min Q(T_m g, ·), and 0 at the target.
  Every place in view and every heading in every stored move is a target
  (hindsight, `1707_01495`), with one network for all targets
  (`universal-value-function-approximators`). It trains only on moves that
  had their effect. So V is how soon in open space, and whether the way is
  open is a condition, not averaged into Q.
- **The route.** Starting from the target's placement in the agent's
  frame, apply Q's move and its learned transformation until the target is
  reached, collecting the tokens stepped onto. This is the one designed
  procedure: it predicts what the approach will need from its own learned
  moves. It calls no recall and searches no sequences.
- **Obstacles are conditions.** "Tile v is walkable" is met when recall's
  forward kind predicts that stepping onto v moves the agent. When it is
  unmet, it is pursued like any condition (cards 029, 038). An achiever
  that makes v walkable, such as toggling a door, is checked in the
  situation its effect produces (card 043). Otherwise the approach is
  chained through a waypoint (skill chaining; `1906_05253` plans over
  waypoints with a learned distance). The waypoint is itself a placement
  to reach, so chains nest up to card 029's depth, with backtracking.
- **Choosing between alternatives** (the key or the switch; which side to
  face a thing from) uses V along the chain, not the hand-set closeness.
- **Removed:** the hand-set closeness, moving closer in the head, move
  ways, the check that places connect, and the imagined check of a door
  in the way. Nothing in walking calls recall on an imagined move.
- **Declared exceptions** (CHARTER rule 8): the route procedure above;
  placing the view and recall's appearance set (card 044).
- **Unchanged:** card 044's tokens, card 043's planner and conditions,
  card 042's recall, arm A's encoders (seeds 400–404), the data.

## 3. Dependencies

- Card 044 (kept): tokens with a where; move transformations exact.
- Recall's forward kind, moved or blocked per token: held-out effects 1.0
  in every seed (card 044).
- Card 029's means-ends procedure and card 043's checking in produced
  situations (kept).
- Methods: `universal-value-function-approximators`, `1707_01495`,
  `1906_05253`; skill chaining (Konidaris and Barto 2009, not in papi).

## 4. Data check

- **Moves:** about 200,000 stored move transitions per world; every
  offset within 6 tiles occurs at every heading.
- **Obstacles** (`logicdoor.make_layout`): a dividing wall with one door,
  the goal beyond it, and up to four objects in the start room (two keys,
  the switch, a vase). Version 7 needs move ways in 2.7–4% of moves: the
  cases conditions must now cover.
- **Unseen rooms,** built on the tools side: sizes 6 and 7 (training
  uses 8) from `make_layout`, and a mirrored 8 with the start room right
  of the wall. A rotated room is not a test: egocentric views make it a
  familiar room at another heading.

## 5. Feasibility gate

- **Upper bound:** the same procedure with the evaluator's exact steps
  (a search of the simulator) in place of V, on familiar and unseen rooms.
- **Trivial baseline:** card 044's walking on the same layouts. Random
  moves within walking are reported as a floor.
- **Mechanism, before acting:**
  - Q against the evaluator's open-space steps, on held-out placements;
  - the route's clear-or-blocked prediction against whether the approach
    actually reaches the target, on held-out states;
  - imagined moves checked by recall per real move in card 044's walking
    (counted, the figure this card brings to zero).
- **Shakedown:** spare seed 399, 30 layouts per world and room.

## 6. Success criteria and prediction

Arm A, seeds 400–404; each criterion in at least 4 of 5 seeds.

1. **Nothing lost.** Familiar rooms, four worlds, 500 layouts: goal in
   ≥ 99%, mean steps within 5% of card 029's; the key world with yellow
   and with purple keys ≥ 99%.
2. **Unseen rooms (C5).** Sizes 6 and 7 and the mirrored 8, four worlds,
   100 layouts each: goal in ≥ 99%, mean steps at most 1.15 times the
   shortest route (card 029's familiar ratio: 1.03–1.11).
3. **How soon, predicted (P12).** On states from unseen-room episodes,
   for every thing in view that can be reached, the predicted steps until
   it can be faced (the chain's V, waypoints included) against the
   evaluator's fewest steps: rank correlation ≥ 0.95, walls between
   included.

Reported: imagined moves checked by recall in walking (card 044's figure
against 0); walking's share of planning time against card 038's 45%;
seconds per layout (P17); how often waypoints and doors as conditions are
used.

**Prediction.** All three pass. V is exact in open space, since the moves
are exact transformations. The doorway becomes a waypoint because it is
the only placement with both routes clear. The risk is the start room's
clutter. Q's single route can be blocked where another route of the same
length is clear; a waypoint at the corner then finds it at the same
length. Planning time falls by about walking's share.

**Budget.**
- Building: Q and its training (minutes, once per world), the route and
  waypoints in the planner, the unseen rooms, the upper bound.
- Acting: 2,000 familiar and 1,200 unseen layouts per seed. Card 044 took
  about 3.5 minutes per seed for the familiar ones, so about 30 minutes
  for five seeds. If over 30 minutes, the main run goes to the user.
