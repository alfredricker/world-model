---
id: "041"
title: where things are, and how soon they can be faced
rung: 0
serves: [P12, P21, P6, P17, C1]
status: abandoned
verdict:
arch_version: 6
date: 2026-09-30
---

# 041: where things are, and how soon they can be faced

## 1. Question

Walking is the part of the planner that is not yet conditional
(ARCHITECTURE.md, known limits). "Moving closer" is a hand-set distance:
grid steps, then turns. It is checked by imagining one step at a time, and
it takes about 45% of planning time (card 038). After
[card 044](../044-state-as-tokens/card.md), every thing is a token with a
where (its offset from the agent), and each move is a transformation of
where. Suppose a small network, trained on the agent's own transitions,
predicts from a thing's where how soon it can be faced, and picks the move
that lowers that most.

Does the agent then reach its goals as well as before, in rooms it has
never seen? Does it predict how soon a thing can be faced? Serves:
- P12, how soon a condition can be met;
- P21, moving as fast System 1;
- P6, how moves change the abstract state;
- P17, planning cost;
- C1, the distance learned, not supplied.

## 2. What changes

One component, walking's closeness. Revised 2026-10-01 with the user:
the where-tokens and the moves as transformations (T_m) moved to card 044,
since tokens need a way to move. Obstacles as conditions are card 045.

```
before: closeness(where, target) = grid distance, then turns         (hand-set)
        move = the first of forward, left, right that lowers closeness in imagination
after:  Q(w, m) = steps until the thing at w is ahead, after move m   (a small network)
        closeness = V(w) = min_m Q(w, m);  move = argmin_m Q(w, m)    (no imagined step)
        w is card 044's where; T_m is card 044's transformation
```

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
- **Declared exceptions** (CHARTER rule 8), removed by card 045: the
  check that an approach reaches its target, the move ways (one and two
  moves deep) and the check that places connect still imagine moves.
  Placing the view stays card 044's, until views stop covering the whole
  room.
- **Unchanged:** card 044's state as tokens, card 043's planner and card
  042's recall (two levels: codes for the same thing, vectors for similar
  things), arm A's encoders (seeds 400–404), the three-colour data.

## 3. Dependencies

- **Card 044's state as tokens,** on card 043's planner with card 042's
  recall. Card 041 runs after card 044's main run passes, so that
  walking's failures cannot be confused with the state's.
- Recall's forward kind (moved or blocked; held-out effects 1.0 in card
  042's shakedown).
- Methods: `universal-value-function-approximators`, `1707_01495`.

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
  familiar states. (T_m's fit is card 044's gate: exact.)
- **Shakedown:** spare seed 399, 30 layouts per world and room.

## 6. Success criteria and prediction

Arm A, seeds 400–404; each criterion in at least 4 of 5 seeds.

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
those cases, and card 045 makes them conditions. Planning time falls,
since no imagined steps choose moves.

**Budget.**
- Building: Q and its training, and the unseen rooms.
- Q trains once per world, independent of the encoder seed; minutes.
- Acting: 2,000 familiar and 1,200 unseen layouts per seed, at about 1.5–2
  seconds each: about an hour per seed. The main run goes to the user.

## 7. Result

Not run.

## 8. Decision

**Stop** (the user, 2026-10-01). Replaced by
[card 045](../045-movement-through-conditions/card.md): its where-tokens
and move transformations became card 044's, and its learned how-soon
became card 045's System 1, without the imagined checks this card kept.
