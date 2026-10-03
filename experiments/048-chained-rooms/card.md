---
id: "048"
title: version 8 in chained rooms
rung: 0
serves: [P21, P8, P2, P17]
status: done
verdict: pass
arch_version: 8
date: 2026-10-01
---

# 048: version 8 in chained rooms

## 1. Question

Can architecture version 8, unchanged, reach the goal in three chained
rooms? The test is a world larger than the view, where rooms leave sight as
the agent walks and the goal can take two doors in a row. The user asked
for it (2026-10-01) as the first step toward CHARTER.md's chained rooms
and, later, Crafter. Before rung 1. Serves:
- P21, chains of conditions longer than in training;
- P8, known effects composed in a new layout;
- P2, tokens remembered once out of view;
- P17, planning cost in a larger world.

## 2. What changes

Nothing in the agent. A new test world (evaluator side), with the agent
and its memory as for the key world:

```
key world (training, 8 x 8):      one wall, one locked door, its key and a second key, the goal behind
chained rooms (this card, 13 x 8): two walls at x = 4 and 8, three rooms of 3 x 6, a locked door in each
                                   wall in its own colour; the agent starts in the middle room with the
                                   switch and the vase (both do nothing, as in the key world); goal in an
                                   outer room, left or right at random
  one door:  both keys in the middle room; open the goal room's door with its key
  two doors: the goal room's key is in the other outer room, behind the other door, whose key is in
             the middle room: key A, door A, put key A down, key B, door B, the goal
```

- **Why this size.** Version 8 places the agent only within 6 tiles of
  where it started (676 placements: 13 × 13 places × 4 headings), and
  shows a place never seen as wall. A world with any place more than 6
  tiles from every start cannot be passed by construction, so it is not
  an experiment here. In this world every place is within 6 tiles of the
  middle room. The outer rooms still leave the view, for example while
  the agent stands in the other outer room.
- **Memory** is reset to the key world's memory for every layout (the
  familiar tests carry what is learned across a chunk of layouts).
- **Code:** `tools/card048/chain.py` (world, views, shortest routes,
  acting loop).

## 3. Dependencies

- Card 047's agent (kept, version 8) and the key world's memory and
  encoders (arm A, seeds 400–404).
- The view renderer (card 016's egocentric view, generalised to any grid:
  checked by eye on a rendered layout, the agent's view facing up equal to
  the top-down map).

## 4. Data check

- **Layouts** (`runs/048_layouts.pkl`, seeds 48000–48001), 100 per
  variant. Fewest steps by breadth-first search over full states: one
  door 17.5 on average (10–40), two doors 37.1 (25–50).
- **Training** (the key world): one door; a second key of another colour
  in every layout. Never two doors, never two doors in view, never a
  needed key behind a door.

## 5. Feasibility gate

- **Upper bound:** the evaluator's search reaches every layout within the
  budget of 200 steps (by construction).
- **Trivial baseline:** random actions, 20 runs per layout, reach the goal
  in 0.15% (one door) and 0% (two doors).
- **Shakedown,** spare seed 399, 10 layouts per variant
  (`runs/048_shakedown_399.json`, 2 minutes):
  - one door: 10 of 10, at 1.09 times the shortest route (9 of 10
    exactly shortest), 0.42 s per layout;
  - two doors: 0 of 10; 93% of steps random (no plan), 7.1 s per layout.
- **Why two doors fail** (traced on layout 3; placing the view was right
  at every step, 0 mismatched places):
  1. At the start the planner finds the whole chain: the goal needs door
     B open, which needs key B in hand, which needs facing key B, which
     needs door A open, which needs key A in hand. It walks to key A and
     picks it up.
  2. Now it holds key A, and "hold key B" has no achiever. Achievers are
     the actions whose predicted effect makes the condition true *from
     the present*; picking up key B with a full hand is predicted to do
     nothing. The planner does not look for the situation in which the
     pick up works (an empty hand) and make that a need (drop key A). At
     the start the hand was empty, so "hand empty" was never a need, and
     nothing protected or restored it.
  3. With no plan, the agent acts at random for the rest of the budget.
     A random drop can then put key A in front of a door, so that two
     tokens block the way; walking makes only one token at a time a
     condition, and finds nothing.
  - The search depth is not the cause: 8 and 10 levels instead of 6 gave
    the same steps on layouts 0, 3 and 5.

## 6. Success criteria and prediction

Arm A, seeds 400–404; 100 one-door and 30 two-door layouts per seed.

1. **One door:** goal reached in ≥ 95% of layouts at ≤ 1.15 times the
   shortest route, in at least 4 of 5 seeds.

Reported: two doors (success, share of random steps); time per layout.

**Prediction** (made after the shakedown on seed 399): one door passes in
5 of 5 seeds; two doors reach the goal in under 5%, for the reason traced
in section 5.

**Budget.** About 6 minutes per seed, the five seeds in parallel; under 10
minutes in all.

## 7. Result

Main run, arm A, seeds 400–404 (`runs/048_seed40{0..4}.json`; numbers in
[results.json](results.json)). It took 19 minutes, against the budget's
10, which also broke the rule that runs over 10 minutes need an approved
card. Seed 401's failing layouts each ran the full 200 steps at 5–10 s;
the estimate came from seed 399 alone.

| Seed | One door: goal | Steps over the shortest | Solved by plan alone | Two doors: goal | Two doors: random steps | Criterion 1 |
|---|---|---|---|---|---|---|
| 400 | 97% | 1.10 | 90 of 100 | 0 of 30 | 91% | pass |
| 401 | 46% | 2.33 | 35 of 100 | 0 of 30 | 36% | fail |
| 402 | 97% | 1.10 | 90 of 100 | 1 of 30 | 91% | pass |
| 403 | 97% | 1.10 | 90 of 100 | 1 of 30 | 46% | pass |
| 404 | 98% | 1.08 | 90 of 100 | 0 of 30 | 3% | pass |

- **Criterion 1: pass, 4 of 5.** With one door, the agent crosses the
  three-room world by plan in 90 of 100 layouts in four seeds, from the
  key world's memory, with the far rooms out of view part of the time.
  Random actions reach the goal in 0.15%.
- **Seed 401** fails with one door. Traced on layout 0's first step:
  with two doors in view, which training never shows, recall no longer
  predicts that the green key opens the green door with the view as it
  is. The planner then needs the view changed as well as the hand, which
  is card 043's declared exception, checked in a spliced situation. The
  agent soon has no plan (185 of 200 steps random). The other seeds solve
  the same layouts by plan.
- **The 2–3% one-door failures in the good seeds** were not traced. In
  one (layout 5) the agent starts boxed in by a key and the switch, so
  that two tokens block the way; the others are similar or untraced.
- **Two doors: 2 of 150.** Door A was opened in 0–5 layouts per seed.
  Traced on seed 399's layout 3 (section 5), with recall checked
  directly:
  1. The planner finds the whole chain at the first step and picks up
     key A.
  2. Holding key A, it asks whether picking up key B *with an empty hand*
     would let door B open. It imagines this by writing key B into the
     hand, so key A vanishes from the imagined room. No key is then left
     on the floor in view.
  3. Recall, asked to toggle the blue door holding the blue key in that
     view, predicts nothing happens. With key A put back on the floor in
     view it predicts the door opens; removing the second door from view
     changes nothing. In the key world a second key always lies in view,
     so recall has never seen a toggle without one.
  4. So "hold key B" gets no achiever, and the agent acts at random. In
     seeds 401, 403 and 404 it instead keeps acting on wrong predictions
     (3,258–5,695 against about 830).
- **Prediction:** one door met in four seeds, not five (seed 401); two
  doors met (under 5%).

## 8. Decision

**Proposed: keep** (awaiting the user), as a measurement: nothing in the
agent changed. Version 8 crosses chained rooms with one locked door. It
cannot yet chain two, and one seed fails with one door. Both failures
come from recall's view part, which compares the whole set of things in
view:
- the planner splices the hand in its imagination (key A vanishes) and so
  asks recall about a situation that cannot occur. This is card 043's
  splicing, now for the hand inside achievers' conditions;
- a new combination in view (two doors; no key on the floor) sways
  predictions about actions that do not depend on it.

Two next cards, one component each:
1. **Hand changes imagined, not spliced** (planner): the hand an action
   needs is produced by imagining the action that makes it (put key A
   down, then pick up key B), as card 043 checks needs.
2. **What in view matters, per action** (recall): each thing in view
   weighed by how much it changes the outcome, as attention over the
   view's tokens, instead of the set compared whole (the user's
   relations direction; card 044's declared exception).
