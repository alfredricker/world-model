---
id: "041"
title: where things are, and how soon they can be faced
rung: 0
serves: [P12, P21, P6, P17, C1]
status: approved
verdict:
arch_version: 6
date: 2026-09-30
---

# 041: where things are, and how soon they can be faced

## 1. Question

Walking is the part of the planner that is not yet conditional
(ARCHITECTURE.md, known limits). "Moving closer" is a hand-set distance:
grid steps, then turns. It is checked by imagining one step at a time, and
it takes about 45% of planning time (card 038). Suppose instead:
- every thing carries a where-vector, its place relative to the agent;
- each move is a learned transformation of where-vectors;
- a small network, trained on the agent's own transitions, predicts how
  soon a thing can be faced, and picks the move that lowers that most.

Does the agent then reach its goals as well as before, in rooms it has
never seen? Does it predict how soon a thing can be faced? Serves:
- P12, how soon a condition can be met;
- P21, moving as fast System 1;
- P6, how moves change the abstract state;
- P17, planning cost;
- C1, the distance learned, not supplied.

## 2. What changes

One component, walking, in two parts that the user asked for together
(2026-09-30). Obstacles as conditions are card 044.

```
before: closeness(pose, target) = grid distance, then turns          (hand-set)
        move = the first of forward, left, right that lowers closeness in imagination
after:  where(thing) = its tile's offset from the agent, turned to the agent's heading
        T_m: where → where after move m (learned, one affine map per move)
        Q(w, m) = steps until the thing at w is ahead, after move m   (a small network)
        closeness = V(w) = min_m Q(w, m);  move = argmin_m Q(w, m)    (no imagined step)
```

- **Where-tokens.** Each thing in view becomes a token with a what part
  (the encoder's vector and its codes, as in card 042) and a where part
  (its offset in tiles). The offsets come from the view's grid, a
  declared prior like the tiles themselves.
- **Moves as transformations.** T_m is fitted to the place
  correspondences before and after each move, which card 028 already
  counts from pixel transitions. Forward's map applies when recall's
  forward kind predicts "moved"; a blocked forward leaves where unchanged.
  An affine map holds at offsets and in rooms never seen, which the
  current table of places does not.
- **How soon, learned from experience.** Q is a small network: two inputs
  (the offset) and three outputs (left, right, forward). Every thing in
  view in every stored move transition is a target (hindsight,
  `1707_01495`). One network covers all targets (`universal-value-function-approximators`).
  It is trained by fitted Q-iteration: Q(w, m) = 1 + min Q(w′, ·), with
  w′ = T_m(w) or w as observed, and 0 once the thing is ahead. Taking the
  minimum gives the competent how-soon P12 asks for, from random play.
- **In the planner:**
  - the facing need's closeness and the choice of nearest target use V;
  - a real step toward a target takes argmin Q.

  Imagined walks stop being the way to choose a move.
- **Declared exceptions** (CHARTER rule 8), removed by card 044: the
  check that an approach reaches its target, the move ways (one and two
  moves deep) and the check that places connect still imagine moves.
  Placing the view still uses card 028's pose table, until views stop
  covering the whole room.
- **Unchanged:** card 043's planner and card 042's recall (two levels:
  codes for the same thing, vectors for similar things), arm A's encoders
  (seeds 400–404), the three-colour data.

## 3. Dependencies

- **Card 043's planner with card 042's recall.** Card 042's effects were
  exact in every seed; its one failing world is card 043's subject. Card
  041 runs after card 043's main run passes, so that walking's failures
  cannot be confused with the planner's.
- Card 028's move correspondences and poses (passed); recall's forward
  kind (moved or blocked; held-out effects 1.0 in card 042's shakedown).
- Methods: `universal-value-function-approximators`, `1707_01495`;
  transformations acting on states (`1812_02230`).

## 4. Data check

Per world: about 200,000 stored move transitions (1.6 million weighted
tries), with 5.2 things in view on average, so about a million targets
per world. Every offset within the view (±6 tiles) occurs.

**Unseen rooms** (built from the tools side, nothing in `src/`):
- rooms of size 6 and 7 (training uses 8);
- a mirrored 8, with the start room right of the dividing wall and the
  goal left of it.

A rotated room is not a test: in egocentric views it looks the same as a
familiar room with another heading.

## 5. Feasibility gate

- **Upper bound:** the same planner with the evaluator's exact steps until
  a thing is ahead (a search of the simulator) in place of V, on familiar
  and unseen rooms.
- **Trivial baseline:** the hand-set walking (card 043's planner as it
  is) on the same layouts. Moves chosen at random within walking are
  reported as a floor.
- **Mechanism:** Q's error against the evaluator's steps on held-out
  familiar states. T_m's fit to the correspondences.
- **Shakedown:** spare seed 399, 30 layouts per world and room.

## 6. Success criteria and prediction

Arm B, seeds 400–404; each criterion in at least 4 of 5 seeds.

1. **Nothing lost.** Familiar rooms, four worlds: goal in ≥ 99% of 500
   layouts, mean steps within 5% of card 029's.
2. **Unseen rooms.** Sizes 6 and 7 and the mirrored 8, four worlds, 100
   layouts each: goal in ≥ 99%, mean steps within 5% of the hand-set
   walking's on the same layouts.
3. **How soon, predicted (P12).** On states from unseen-room episodes,
   for every thing in view, V against the evaluator's fewest steps until
   it is ahead. Rank correlation ≥ 0.95 where no wall stands between the
   agent and the thing; all states are reported. Walking's share of
   planning time is reported against card 038's 45% (P17).

**Prediction.** All three pass. How soon depends only on the offset in
open space, and training views cover every offset from both sides.
Where a wall stands between, V underestimates. The move ways still rescue
those cases, and card 044 makes them conditions. Planning time falls,
since no imagined steps choose moves.

**Budget.**
- Building: T_m, Q and its training, and the unseen rooms.
- Q trains once per world, independent of the encoder seed; minutes.
- Acting: 2,000 familiar and 1,200 unseen layouts per seed, at about 1.5–2
  seconds each: about an hour per seed. The main run goes to the user.
