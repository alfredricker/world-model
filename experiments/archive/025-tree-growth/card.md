---
id: "025"
title: growing the tree below common conditions
rung: 0
serves: [P12, P21, P9]
status: done
verdict: fail
arch_version: 3
date: 2026-09-28
---

# 025: growing the tree below common conditions

## 1. Question

With conditions meaning "approachable by moving closer" (card 024's measure),
does the tree reach the goal without random wandering once discovery keeps
breaking down a condition that is usually, but not always, true at the start?
A theory check with exact computation. Serves P12 (subgoals down to single
moves), P21 (one fast choice per move) and P9 (any layout). Revises
[card 024](../024-closer-by-a-free-step/card.md): acting 96.4%, but 50% of
moves random, nearly all with no key, the door closed and no way applying.

## 2. What changes

One component: how discovery grows the tree (card 012's `run_exact`).

```
card 024: a condition on at > 50% of start states is a leaf (not broken down);
          16 goals
card 025: a leaf only if on at > 99% of start states;
          64 goals
```

Why both parts of the rule: the key's way ("picking up the key makes the
door approachable") holds at 82% of starts, so under the old rule it is a
leaf, with nothing below it for the other 18%. Its detours would sit at
depth 4, but depth 3 alone takes about 36 goals (six ways under each of the
six ways of the goal square), so with 16 or 32 goals the tree is full before
depth 4 and this test could not pass. With 64, depth 4 starts with the door's
children, the key's way first (breadth-first order, as card 012).

Not changed, although proposed: spending the budget where conditions are
most often false at the start. On inspection that favours exactly the turn
and step ways (false at almost every start) over the key's way (false at
18%). Everything else is card 024's: the measure, data, chooser, 500 new
layouts, 200-step budget.

## 3. Dependencies

Card 024's pipeline (`tools/card023/closer.py --measure step`, now also
`--start-share`), its control and baseline; card 012's exact discovery and
chooser; the key-world simulator. Nothing learned.

## 4. Data check

From card 024: the key's way is on at 81.8% of start states (card 012's
conditions: 100%). Random moves in 200 layouts: 3,170 of 3,262 with no key,
the door closed and no way applying.

## 5. Feasibility gate

- **Upper bound:** card 012's conditions with path-search walking, rerun
  with the same rule and budget (card 024: 100%, 16.3 steps).
- **Trivial baselines:** card 012's conditions with moving closer (card
  024: 99.0%, 35.6% random moves); card 024's tree: 96.4%, 50.1% random.

Result of the gate, before the main run:

## 6. Success criteria and prediction

Card 024's criteria, unchanged, same 500 layouts:

1. **Acting:** reaches the goal square in ≥ 98% of layouts.
2. **Subgoals, not luck:** random moves (no way of the tree applies) ≤ 1%
   of all moves.
3. **No wandering:** mean steps when successful ≤ 1.5 × the control's.

Reported with them: the tree, and which ways acting uses.

Prediction: the key's way gets ways below it (steps and turns around an
object, toggling the vase, picking up the other key) and random moves fall
sharply. Risk: some of those detours need a further level (depth 5), which
the budget may not reach; the turn and step ways under the goal square
still take most of the tree. Budget: discovery grows with the number of
goals, about 10–15 minutes in all, under 30.

## 7. Result

`runs/025tree.out`, 6.7 minutes. 500 new layouts, 200-step budget.

| Arm | Reaches the goal | Mean steps | Random moves (share of all moves) |
|---|---|---|---|
| Control: card 012's conditions, path-search walking | 100% | 16.3 | 0% |
| Baseline: card 012's conditions, moving closer | 99.0% | 24.2 | 35.6% |
| Card 024's tree | 96.4% | 23.9 | 50.1% |
| **Card 025's tree** | **100%** | **16.5** | **1.27%** |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Acting ≥ 98% | 100% | Pass |
| 2. Random moves ≤ 1% of all moves | 1.27% (105 of 8,258) | Fail, narrowly |
| 3. Mean steps ≤ 1.5 × control (24.5) | 16.5 (shortest route 15.9) | Pass |

The key's way is now broken down (its ways: turn or step around an obstacle,
toggle, drop the other key; each on at 82–89% of starts). All the remaining
random moves (99 in a rerun of acting) come from 3 of the 500 layouts, with
no key, the door closed and no way applying: the key is blocked in a way
that needs a detour one level deeper than the tree reaches. The tree used
all 64 goals. Acting used 8 of its ways: the goal square (2,521 steps), the
door (3,016), the key (2,247) and five detours (398); the turn and step ways
under the goal square were never used.

## 8. Decision

**Revise** (2026-09-28, with the user). In exact form, "move closer, and
take a discovered subgoal when stuck" reaches the goal as reliably and
almost as directly as path search; criterion 2 is missed by 0.27 points,
all from 3 layouts that need one more level of detour. The cost is the
tree: grown breadth-first before acting, it filled 64 goals, of which
acting used 8, and it still ran out one level short.
[Card 026](../026-depth-first-subgoals/card.md) grows it only where the
agent gets stuck, depth first with backtracking.
