---
id: "026"
title: depth-first subgoals
rung: 0
serves: [P12, P21, P17]
status: done
verdict: fail
arch_version: 3
date: 2026-09-28
---

# 026: depth-first subgoals

## 1. Question

Does the agent reach the goal as well as card 025 if it grows its subgoals
only when it is stuck, walking down from the goal one yes/no question at a
time and backing up when a branch leads nowhere, and does it then keep far
fewer conditions? A theory check with exact computation. Serves P12
(subgoals down to single moves), P21 (deliberation "over chains of
conditions ... backtracking when one fails") and P17 (the cost of that
deliberation, reported). Revises
[card 025](../025-tree-growth/card.md): 100% reached, 1.27% random moves,
but a 64-goal tree of which acting used 8, and still one level short.

## 2. What changes

One component: when the tree grows and how acting walks it.

```
card 025: grow the whole tree before acting, breadth-first: every way the
          evidence supports under every goal, up to 64 goals;
          acting takes the shallowest goal that has a way possible now
card 026: nothing grown in advance. At each move, from the goal square down:
            is one of this goal's ways possible now? (in order of evidence)
              yes -> act on the first that is (move closer, or do the action)
              no  -> go down into its ways one at a time (depth first);
                     nothing possible below a way (to depth 6) -> back up, try the next
          a goal's ways are discovered the first time the search needs them
          (card 012's evidence test and self-check on the same experience),
          then kept; no goal budget, no "on at start" leaf rule
```

The ways of a goal come from the stored experience, not from the layout
being played, so discovering them on demand gives the same ways as
discovering them in advance. To keep the computation batched, acting runs
in rounds: a round replays all 500 layouts from the start with the ways
found so far, and a layout pauses at the first goal its search needs that
has not been expanded. Those goals are expanded from the experience, and
the next round replays. Play up to a pause is exactly what discovery on
demand would do, so the goals expanded are exactly the ones it would
expand. The last round, when no layout pauses, is the result.

Unchanged from card 025: conditions mean "approachable by moving closer"
(card 024's measure), the evidence test and self-check, the 5,000-episode
experience, the 500 new layouts, the 200-step budget, depth cap 6.

## 3. Dependencies

Card 025's pipeline (`tools/card023/closer.py`: conditions, measure,
moving closer, acting) and its result as the comparison; card 012's
exact discovery (evidence test, self-check); the key-world simulator.
Nothing learned.

## 4. Data check

The same experience as cards 023–025. Card 025's acting used 8 ways: the
goal square, the door, the key and five detours, 8,258 moves in all.

## 5. Feasibility gate

- **Upper bound:** card 012's conditions with path-search walking: 100%,
  16.3 steps (cards 023–025).
- **Trivial baseline:** card 025's breadth-first tree: 100%, 16.5 steps,
  1.27% random moves, 64 goals stored.

Result of the gate, before the main run: both rerun in the same script on
the same 500 layouts, rebuilt from card 025's saved trees, and reproduced
exactly: control 100%, 16.3 steps; card 025's tree 100%, 16.5 steps,
1.27% random moves.

## 6. Success criteria and prediction

Same 500 new layouts:

1. **Acting, without luck:** reaches the goal square in ≥ 98% of layouts,
   with random moves (no way applies) ≤ 1% of all moves.
2. **No wandering:** mean steps when successful ≤ 1.5 × the control's
   (16.3). Depth first takes the first chain that works, not the shortest.
3. **Fewer conditions:** conditions stored (every way of every goal the
   search expanded) ≤ 32, half of card 025's 64. Counted as card 025's 64
   was: every node of the tree, including the goal square and the ways the
   self-check rejected (card 025: 49 accepted, 14 rejected).

Reported with them: the goals expanded, the rounds needed, and the cost per
move (conditions checked, mean and worst).

Prediction: acting 100% with random moves near 0, since nothing now stops
growth one level short. Steps close to card 025's. Stored conditions about
25–35: each expanded goal brings all its ways (about six), so criterion 3
is the risk. Budget: a few rounds of acting and discovery, about 5–10
minutes, under 10.

## 7. Result

`runs/026dfs.out`, 4.2 minutes. 500 new layouts, 200-step budget.

| Arm | Reaches the goal | Mean steps | Random moves | Conditions stored | Conditions checked per move (mean / worst) |
|---|---|---|---|---|---|
| Control: card 012's conditions, path-search walking | 100% | 16.3 | 0% | 18 | 7.4 / 13 |
| Card 025: breadth-first tree, grown in advance | 100% | 16.5 | 1.27% | 64 | 25.6 / 49 |
| **Card 026: depth first, grown when stuck** | **100%** | **16.4** | **0%** | **58** | **4.1 / 50** |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Acting ≥ 98%, random moves ≤ 1% | 100%, 0 of 8,216 moves random | Pass |
| 2. Mean steps ≤ 1.5 × control (24.5) | 16.4 (shortest route 15.9) | Pass |
| 3. Conditions stored ≤ 32 | 58 (goal square, 53 accepted, 4 rejected) | Fail |

The search expanded 11 goals over 12 rounds. Growing only where needed
removed the "one level short" failure: the 3 layouts that made card 025's
random moves now find a detour. Per move, the search checks 4.1 conditions
on average, against 25.6 for card 025's chooser.

Criterion 3 fails because of one layout. Replaying the final tree and
recording which goals each layout's search opened: 405 of 500 layouts
opened 3 goals (reach the goal square, step onto it, open the door), 63
also "pick up the key", 28 also "drop the other key", 3 also "turn beside
the key". Layout 202 opened all 11. Its key is in a pocket against the wall
that can only be reached from one side, with the other key and the switch
next to it. Nothing under "pick up the key" works there down to depth 6, so
the search opened all five of its detours, about six ways each, before
backing up to "turn at the door" and finding a chain under it. Without that
layout the tree would hold 6 goals and 30 conditions. So what is stored is
set by the hardest layout, not the typical one: once a branch fails,
backtracking searches all of it. Many of the stored ways also look
duplicated: under the key's way, "turn right" and "turn left" have
children with identical evidence scores.

Prediction: acting and steps as predicted. Stored conditions predicted
25–35, actual 58; the typical layout needs 30.

## 8. Decision

**Keep** (2026-09-28, with the user), although criterion 3 failed. Growing
subgoals only when stuck and walking them depth first, with backtracking,
reaches the goal in every layout with no random moves and checks about a
sixth as many conditions per move as card 025. The failed storage criterion
is understood: one layout of 500 searched a whole failed branch before
backing up, and several stored ways look like duplicates. Neither blocks
the next step. The next sessions are theory: how the agent discovers the
conditions and objects themselves, which the exact checks here still
supply.
