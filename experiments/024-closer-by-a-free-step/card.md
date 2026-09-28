---
id: "024"
title: closer by a free step
rung: 0
serves: [P12, P21, P9]
status: done
verdict: fail
arch_version: 3
date: 2026-09-28
---

# 024: closer by a free step

## 1. Question

With a closeness measure that stops only at real obstacles, does the
condition tree built on "approachable by moving closer" reach the goal, with
discovery supplying the detours as subgoals? A theory check with exact
computation, before any learning. Serves P12 (the hierarchy reaches single
moves), P21 (one fast choice per move, no route search) and P9 (any layout).
Revises [card 023](../023-closer-in-view/card.md), which reached the goal in
39.6% of new layouts because its measure stopped agents that were merely
facing a wall, and discovery then filled the tree with moves.

## 2. What changes

One component: the closeness measure behind "moving closer".

```
card 023: target = the tile in front of a ready pose
          closeness = (tiles away, target ahead 0 / beside 1 / behind 2)
          facing a wall with the target diagonally ahead: no move is closer -> stuck
card 024: target = the ready pose itself (the tile to stand on, the way to face)
          closeness = (tiles away from that tile,
                       turns until facing a direction where one step forward goes
                       onto a free tile and brings the agent closer;
                       on the tile: turns until facing the pose's direction)
          no such direction -> stuck: something really is in the way
```

The agent still never counts steps to the target or compares routes. It
looks at where the target is relative to itself and which of the four tiles
around it it can step onto; later, that second part is what the learned
effects of moves supply. Everything else is card 023's: conditions mean
"approachable by moving closer", card 012's exact discovery, the same data,
layouts, chooser and 200-step budget, 16 goals.

## 3. Dependencies

Card 023's pipeline (`tools/card023/closer.py`, now with `--measure step`),
its control and baseline; card 012's exact discovery and chooser; the
key-world simulator. With the old measure the pipeline reproduces card
023's results. Nothing learned.

## 4. Data check

Before any discovery, on card 023's frames (5,000 random-play episodes'
sequence frames, the way's parent condition off), card 012's tree: where a
path exists, how often moving closer gets stuck, old measure → new:

| Way | Stuck, card 023 | Stuck, card 024 | Main cause now |
|---|---|---|---|
| Goal square | 55% | 45% | internal wall between (967 of 1,107) |
| Door, with the key | 60% | 11% | the switch, other key or vase; corners |
| Key | 36% | 12% | walls and corners |

What is left is what subgoals should handle: the internal wall (a doorway
detour) and objects in the way.

## 5. Feasibility gate

- **Upper bound:** card 012's conditions with path-search walking: 100% of
  500 new layouts, 16.3 steps on average (card 023's control).
- **Trivial baselines:** card 012's conditions with moving closer by the new
  measure, falling back on a random action when stuck; random play (0.4%,
  card 012).

Result of the gate, before the main run:

## 6. Success criteria and prediction

Same 500 new layouts as card 023, new conditions, rediscovered tree:

1. **Acting:** reaches the goal square in ≥ 98% of layouts (the control:
   100%; card 023: 39.6%).
2. **Subgoals, not luck:** random actions, taken when no way of the tree
   applies, are ≤ 1% of all moves (the baseline's share reported alongside).
   This shows the detours come from discovered subgoals.
3. **No wandering:** mean steps when successful ≤ 1.5 × the control's
   (16.3; the shortest route averages 15.9).

Declared diagnostic, not part of the verdict: the same run with a budget of
32 goals, reported whether or not 16 was reached. If the tree hits the
budget at 16, it says whether the budget is the only limit.

Prediction: discovery finds a step through the doorway and toggles or
pickups that clear objects in the way; turns should mostly disappear from
the tree. The risk is the budget: each level of the chain (goal square,
door, key) may need its own detours, and at 16 goals the key's way may be
crowded out again. Budget: about 4 minutes per run, two runs.

## 7. Result

`runs/024closer_g16.out` and the diagnostic `runs/024closer_g32.out`, about
4.5 minutes each. 500 new layouts, 200-step budget.

| Arm | Reaches the goal | Mean steps | Random moves (share of all moves) |
|---|---|---|---|
| Control: card 012's conditions, path-search walking | 100% | 16.3 | 0% |
| Baseline: card 012's conditions, moving closer | 99.0% | 24.2 | 35.6% |
| New, 16 goals | **96.4%** | 23.9 | **50.1%** |
| New, 32 goals (diagnostic) | 96.4% | 23.9 | 50.1% |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Acting ≥ 98% | 96.4% | Fail |
| 2. Random moves ≤ 1% of all moves | 50.1% (baseline 35.6%) | Fail |
| 3. Mean steps ≤ 1.5 × control (24.5) | 23.9 | Pass |

Twice the goal budget changes nothing: the extra goals are turns and steps
under the goal square that acting never uses. In 200 of the layouts, every
random move but 92 (3,170 of 3,262) happens with no key, the door closed and
the agent on the key's side, with no way of the tree applying. The key's way
("picking up the key makes the door approachable") holds at 82% of start
states (100% with card 012's conditions), so card 012's stopping rule makes
it a leaf (a condition on at more than half of the starts is not expanded).
In the other 18% of starts the tree has nothing below it, and the agent
wanders until the key becomes approachable by chance. The goal-square and
door levels work: their ways carry most steps (937 and 1,145), and turns
are used for 14 steps in all.

## 8. Decision

**Revise** (2026-09-28, with the user). The measure now does its job; what
fails is how discovery grows the tree: it stops below a condition that is
on at most starts, and its goal budget runs out on turn and step ways
before it reaches the key's detours.
[Card 025](../025-tree-growth/card.md) changes that one component.
