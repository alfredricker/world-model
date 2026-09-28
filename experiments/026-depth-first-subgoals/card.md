---
id: "026"
title: depth-first subgoals
rung: 0
serves: [P12, P21, P17]
status: draft
verdict:
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
in rounds: a round plays all 500 layouts with the ways found so far and
records every goal the search needed but had not expanded; those are
expanded from the experience, and the next round plays again. The last
round, when nothing new is needed, is the result.

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

Result of the gate, before the main run:

## 6. Success criteria and prediction

Same 500 new layouts:

1. **Acting, without luck:** reaches the goal square in ≥ 98% of layouts,
   with random moves (no way applies) ≤ 1% of all moves.
2. **No wandering:** mean steps when successful ≤ 1.5 × the control's
   (16.3). Depth first takes the first chain that works, not the shortest.
3. **Fewer conditions:** conditions stored (every way of every goal the
   search expanded) ≤ 32, half of card 025's 64.

Reported with them: the goals expanded, the rounds needed, and the cost per
move (conditions checked, mean and worst).

Prediction: acting 100% with random moves near 0, since nothing now stops
growth one level short. Steps close to card 025's. Stored conditions about
25–35: each expanded goal brings all its ways (about six), so criterion 3
is the risk. Budget: a few rounds of acting and discovery, about 5–10
minutes, under 10.

## 7. Result

## 8. Decision
