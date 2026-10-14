---
id: "023"
title: closer in view, exact check
rung: 0
serves:
  - P12
  - P21
  - P9
status: done
verdict: fail
arch_version: 3
date: 2026-09-28
---

# 023: closer in view, exact check

## 1. Question

In the key world, does the condition tree reach the goal if "can get there
by walking" means "can get there by moving closer in view, one move at a
time", and does discovery then find the extra subgoals this needs (such as
going through the doorway) by itself? A theory check with exact
computation, before any learning (CHARTER, "theory first"). Serves P12
(the hierarchy reaches primitive actions: "closer" is the lowest subgoal),
P21 (a fast choice per move, no route search) and P9 (any layout).

This replaces walking as a computation backwards from the target (cards
014–016). The agent never counts steps or compares routes: it
predicts each move's effect on where the target appears and takes a move
that brings it closer.

## 2. What changes

One component: what "reachable by walking" means in the exact conditions
of card 012 (`Exact` in `discover_logic.py`).

```
before: a way's condition holds if a state where its action works can be reached
        by any walking path (a path search)
after:  ... can be reached by moving closer in view:
          view offset of the target spot: (a ahead, b to the right) of the agent
          closeness: first |a - 1| + |b|, then 0 ahead / 1 to a side / 2 behind
          repeatedly take a move whose exact effect lowers closeness
          (order: forward, left, right); if none does, not approachable
```

When the target is not approachable, the chooser takes an accessible
subgoal, as it already does for a closed door. Nothing about doorways is
written in: if stepping through the open doorway makes the goal
approachable, card 012's exact discovery should find that as a new way
(forward, from the doorway). Everything else is card 012's exact
discovery and acting.

## 3. Dependencies

Card 012's exact discovery (`run_exact`) and exact acting; the key-world
simulator. Nothing learned.

## 4. Data check

Card 012's exact discovery data. Report how often a way's old condition
("a path exists") holds while the new one ("approachable") does not, per
way, and what was in the way: the wall, the vase, another object.

## 5. Feasibility gate

- **Upper bound:** card 012's exact tree with path-search walking, rerun
  here as the control (card 012 reported 95% acting with its learned
  conditions and exact walking; the exact tree's own rate is measured).
- **Trivial baseline:** the old tree with moving closer in view but the old
  "a path exists" conditions (it gets stuck where the two differ); random
  moves.

Result of the gate, before the main run:

## 6. Success criteria and prediction

1. **Acting:** with the rediscovered tree choosing subgoals and every move
   chosen by "closer in view", the agent reaches the goal square in ≥ 95%
   of 500 new layouts, and no more than 2 points below the path-search
   control.
2. **Discovery finds the detours:** the rediscovered tree has ways the
   control tree lacks (reported by path), within the same goal budget (16
   goals, card 012's), and acting with it beats the baseline (control
   conditions, moving closer) by ≥ 10 points.

If criterion 1 fails, the remaining stuck cases name what discovery could
not find. If both pass, the next card learns the move effects that "closer in
view" uses.

Prediction: discovery finds a "through the doorway" way; a few traps behind
the vase or another object may need their own ways or stay unsolved.
Budget: exact computation, about 10 minutes on the CPU pool (card 012's
exact discovery took minutes).

## 7. Result

`runs/023closer.out`, 3.5 minutes. Both discoveries used the same 5,000
random-play episodes; acting used 500 new layouts, 200-step budget (the
shortest route from the start averages 15.9 steps).

| Arm | Reaches the goal | Mean steps | Random moves (no rule applied) |
|---|---|---|---|
| Control: card 012's conditions, path-search walking | 100% | 16.3 | 0 |
| Baseline: card 012's conditions, moving closer | 86.2% | 23.1 | 14,245 |
| New: approachable conditions, rediscovered tree, moving closer | **39.6%** | 87.8 | 75,792 |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Acting ≥ 95% and within 2 points of the control | 39.6% (control 100%) | Fail |
| 2. New ways, and acting ≥ 10 points above the baseline | New ways found, but acting 46.6 points below the baseline | Fail |

The control's tree is card 012's. The new tree fills its 16-goal budget
with moves: turn right, turn left, forward, nested inside one another
(`forward/right/left`, ...). Toggle and pickup are pushed down or out by
the budget, and acting falls apart.

Data check (sequence frames with the parent condition off): where a path
exists, moving closer gets stuck in 55% of frames for the goal square
(1,051 of 1,329 with the internal wall between) and in 60% for the door
with the key (5,569 of 8,299 facing a wall, the rest facing the switch,
the other key or the vase).

Two reasons. First, the closeness measure has a flaw: "ahead" counts
anything in front of the agent's side, so an agent facing a wall with the
target diagonally ahead sees no gain from turning towards it and stops.
Most of the "facing a wall" cases are this, not real traps. Second, when
"approachable" is false that often, a single turn or step rescues it from
very many frames, so discovery adds each move as a new way, at every level.
The real detour (the internal wall, 1,051 frames) is buried among them.

## 8. Decision

**Revise** (2026-09-28, with the user). The closeness measure, not the
idea, stopped most walks, and discovery then filled the tree with moves.
[Card 024](../024-closer-by-a-free-step/card.md) changes only the measure:
turn towards a direction where one step really brings the agent closer.
