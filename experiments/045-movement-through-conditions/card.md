---
id: "045"
title: movement through conditions
rung: 0
serves: [P21, P12, P6, P17, C1, C5]
status: done
verdict: pass
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

Result of the gate, after building (`tools/card045/movement.py`,
`tools/card045/route_check.py`; spare seed 399, arm A):
- **How soon is exact where it is used.** Q trains in about 20 seconds,
  once per run, on all 2,500 target placements within 12 tiles (6 beyond
  the view, so that moves leaving the view bootstrap from trained values).
  Read in whole steps, V equals the evaluator's open-space steps at all
  676 placements in view and all 2,500 trained (Q's largest error 0.48
  steps). Held out of the loss: with 5% of the placements held out (108),
  V is exact on every one; with 20% (497), V is exact on 30% of the held
  out and 30% of the trained placements alike (mean error 1.0 step for
  both). So with few anchors the whole fit drifts; it does not fail on
  unseen placements in particular. In use Q trains on every placement it
  is asked about (`runs/045_route_check.json`).
- **The route predicts the approach.** At the start of 30 layouts per
  world, over 356,515 pairs of placements the agent can stand at, the
  route was clear exactly when the simulator, taking the approach's moves,
  got through (agreement 1.0 in all four worlds). Getting through always
  took exactly V moves.
- **Upper bound, baseline and floor**, on the same 30 layouts per
  familiar world and per unseen room (`runs/045_shakedown_399_*.json`,
  `runs/045_floor_random_399.json`):

  | Walking | Goal, familiar and unseen | Unseen rooms: steps / shortest | Imagined moves checked by recall, per real move |
  |---|---|---|---|
  | card 045 | 100% everywhere | 1.00–1.13 | 0 |
  | exact (evaluator's fewest steps over the agent's own tokens) | 100% everywhere | the same steps as card 045 in every test | 0 |
  | card 044 (baseline) | 100% everywhere | 1.00–1.16 (the mirrored key room) | 775–868 familiar, 225–911 unseen |
  | random moves within walking (floor) | 7–13% (familiar) | – | – |

  How soon in unseen rooms (criterion 3's measure): rank correlation 1.0
  between the chain's V and the evaluator's steps, over 25,046 pairs.
- **Four faults in my code, found and fixed on the shakedown worlds
  before the runs above.** Counting every thing that could be opened as
  walkable at once made the agent pick up and drop a key forever; each is
  now tried alone. Chains with tied waypoints broke into one-step hops and
  missed the door; a chain now hops from k waypoints to k − 1. A goal
  right behind the door can be faced only from the door's tile, so the
  hypothetical's target placements are taken with the door counted
  walkable. The simulator's view assumed rooms of size 8 (evaluator side;
  smaller rooms are padded with wall, which the view shows beyond the
  grid anyway).
- **Time.** The shakedown ran 9–29% slower per layout than card 044's
  walking (0.54–0.71 s against 0.42–0.64). Two changes that leave every
  decision the same (route tables shared by every need in a situation;
  each thing's chance to be opened kept until memory changes) brought it
  to within 5%: 0.632 and 0.428 s per layout in the key and both worlds,
  against 0.602 and 0.411 (`runs/045_speed_*.json`).
- **The first main run and an exact filter.** The first main run
  (`runs/045_armA_before_filter.json`) passed every criterion in seeds
  400–402. But seed 401's key world took 27 s per layout (seed 400: 0.37)
  with decisions identical to seed 400's. Seed 403's new-colour tests then
  ran for over 17 minutes, and the run was stopped at 60 minutes, past the
  30-minute rule. The cause: in seed 401, recall judges the wall openable
  (memory has seen things like it made walkable, read at the similarity
  level). So each of about 40 wall tiles in view became a "what if this
  were open" chain, about 119 per real step against 5. A filter that is
  exact (a tile can matter only if a route it alone blocks leaves what the
  agent reaches now, and one ends where the target is reached) changed no
  decision (16 of 16 layouts compared step by step, seeds 401 and 399).
  It cut seed 401's key world from 13.3 to 3.9 s for 6 layouts, and seed
  403 ran in 6.8 minutes. The main run below uses it. Card 044 had the
  same weakness at a smaller cost: 1.07 s per layout in seeds 401 and 404
  against 0.37.

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

## 7. Result

Main run, arm A, seeds 400–404, one code version (with section 5's
filter): `runs/045_armA_400-402.json`, `runs/045_seed403_filter.json` and
`runs/045_armA_404.json`, 28 minutes in all. Compact numbers are in
[results.json](results.json); card 044's are from `runs/044_armA.json`.

| Seed | Familiar, four worlds | Key world, yellow / purple | Unseen rooms, 12 | Unseen steps / shortest | How soon, rank correlation | Criteria 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 400 | 100% | 100% / 100% | 100% | 1.01–1.11 | 1.0 | pass / pass / pass |
| 401 | 100% | 100% / 100% | 100% | 1.01–1.11 | 1.0 | pass / pass / pass |
| 402 | 100% | 100% / 100% | 100% | 1.01–1.11 | 1.0 | pass / pass / pass |
| 403 | 100% | 100% / 100% | 100% | 1.01–1.11 | 1.0 | pass / pass / pass |
| 404 | 100% | 100% / 100% | 100% | 1.01–1.11 | 1.0 | pass / pass / pass |

- **Criterion 1: pass, 5 of 5 seeds.** Mean steps were 16.28, 17.21,
  15.35 and 21.69 in the key, switch, either and both worlds: 0.4–0.6%
  fewer than card 044's (16.38, 17.27, 15.42, 21.77), and 0.992–0.997 of
  card 029's. The five seeds took identical steps, as in card 044.
- **Criterion 2: pass, 5 of 5.** Every unseen room in every world, 100
  layouts each: goal 100%, mean steps 1.01–1.11 times the shortest route
  (bar 1.15; the both world's 7 × 7 rooms are the highest).
- **Criterion 3: pass, 5 of 5.** Over 83,710 pairs per seed (every thing
  in view, every state of the unseen-room episodes), the chain's predicted
  steps until a thing can be faced rank exactly as the evaluator's
  fewest steps (correlation 1.0), walls between included; it predicted no
  reachable thing unreachable.
- **Reported:**
  - imagined moves checked by recall while walking: 0 in every seed, world
    and room, against card 044's 775–868 per real move (section 5);
  - steps taken, familiar worlds: 74–77% by the approach, 5–9% toward a
    waypoint, the rest acts (key world: 6,263, 376 and 1,500). 67–77% of
    all steps serve a door made walkable as a condition (key world: 5,618
    of 8,139);
  - time per layout: within 4% of card 044's in 18 of 20 seed-world
    pairs (for example 0.373 against 0.367 s in seed 400's key world). In
    seeds 401 and 404's key worlds it is 2.9 s against card 044's 1.1
    (other seeds 0.37): recall judges the wall openable there, and every
    "what if this wall were open" that passes the filter is pursued as a
    condition and fails;
  - walking's share of planning time, timed as the approach and chain
    computations: 28–46% (90% in those two key worlds), against card 038's
    45%. Card 044's walking timed the same way took 7–19% in the
    shakedown, but part of its work ran outside the timed calls, so the
    comparable figure is time per layout, above;
  - the switch world with a new-colour door (reported only): goal 8–74%,
    within one point of card 044's in every run. Time per layout is
    within 10% of card 044's except seed 403 with purple (20.6 against
    11.9 s);
  - the first main run without the filter (section 5) took the same
    decisions in seeds 400–402: goal, moves, kinds of steps, refusals and
    wrong predictions match in every world and test.
- **Prediction:** met for all three criteria; not met for time. Planning
  did not get faster by walking's share: time per layout is unchanged, and
  2.7 times card 044's where recall misjudges walls.

## 8. Decision

**Revise** (the user, 2026-10-01: "we need to fix your wall filter patch,
using recall codes sounds like a good revision"). Walking through
conditions works as the card asked: a learned approach as System 1, and
its conditions (the tokens on its route walkable, a door made walkable, a
waypoint) as System 2, with no imagined step checked by recall, in the
exact upper bound's steps in familiar and unseen rooms. But it needed a
filter to stay fast, because recall judges "can be made walkable" by
similarity alone, and in two seeds the wall looks openable. The filter
works around a weak judgement instead of fixing it.
[Card 046](../046-openable-by-identity/card.md) reads that judgement at
recall's same-thing level (codes, card 042), removes the filter, and
reruns these criteria. Also open: Q is trained on every placement it is
used for; with a fifth held out, the fit drifts.
