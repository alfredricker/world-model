---
id: "105.5"
title: planning both ways over learned conditions — backward from the goal, forward through imagined transitions
rung: 1
serves: [P12, P21, P14]
status: done
verdict: revise
arch_version: 27
date: 2026-10-13
---

# 105.5: planning both ways over learned conditions

Series [105](../105-one-transition-model/card.md), approved for the overnight
session of 2026-10-13. Agents in breaking world 1 and Crafter; MiniGrid's
tiers come later, when this planner enters version 27's agent.

## 1. Question

Version 27's planner works backward through five hand-written forms. Suppose
the planner works backward through the conditions 105.4 reads from the
transition model, and forward through the model's own imagined next states to
check that a chain holds together, with walking as the model's move
transitions chained. Does it then reach goals no form was written for? Rung 1;
P12, P21 (search from both ends), P14.

## 2. What changes

The planner. Version 27's means-ends search is kept as the procedure:
- **Conditions:** 105.4's interventions on the model, not the five forms.
- **Achievers:** a condition is a token state. Its achievers are the
  transitions the model predicts make it true, found among the actions
  available in imagined states.
- **Forward check:** each chain is imagined one transition at a time from
  the present. The chain is refused if a condition it needs is undone on the
  way (what is used up, side effects).
- **Walking:** the same model's move transitions, chained toward the place
  a condition names (closeness propagated, as card 067, over the model's
  predicted moves). There is no separate walker.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The goal is given as a token state | yes | yes | yes | supplied (C1, until P20) |
| Search depth bounded | yes | yes | yes | declared (6, as version 27) |

## 3. Dependencies

105.1–105.4; card 067's propagation; version 27's search and intention
(cards 029, 097.2).

## 4. Data check

- **World 1:** test worlds, each episode's goal "lamp L lit" for an unlit
  lamp.
- **Crafter:** the curriculum's test starts (worlds 8–9), with the goal "a
  wood pickaxe in the inventory".

## 5. Feasibility gate

- **Upper bound:** the same planner with the evaluator's rules as
  conditions.
- **Trivial baseline:** random play's success in the same step budget.

## 6. Success criteria and prediction

1. **World 1:** the goal lamp is lit within 50 steps in at least 90% of test
   episodes (lamps at new columns).
2. **Crafter:** a wood pickaxe is made within 100 steps in at least 70% of
   test starts with wood available. That needs a table placed or found, and
   wood collected.
3. **Compute:** per-step planning time is reported (P17).

**Prediction:** world 1 passes. Crafter's main risk is walking through the
model's own moves at night.

**Decision rules:**
- **Keep** if 1 and 2 hold.
- **Revise** if one fails within 15 points.
- **Stop** otherwise.

**Budget:** episodes in parallel, each run under 20 minutes, capped.

## 7. Result

Run 2026-10-13, world 1 only ([results.json](results.json);
`tools/card105.5/world1.py`). The planner works in perception's codes.
- **Statements:** learned before acting from 6,000 training steps. On each
  event the transition model predicts, 105.4's interventions give the
  tokens the change depends on. A statement is their offsets and codes,
  with the action; 30 statements were seen at least twice.
- **Backward:** means-ends from the goal over those statements, at most
  depth 6, with the agent's own condition achieved last.
- **Walking:** closeness propagated over the moves the model predicts in
  the imagined state.
- **Forward:** every action imagined one effect at a time. The plan is
  made again only if what is seen differs from what was imagined; it never
  did.
- **Upper bound:** the same planner with the world's step function in
  place of the model, for both statements and imagination.

The learned statements are the world's rules, though none was written:
- **Entering a cell:** the agent comes from the neighbouring cell.
- **Pushing:** the agent stands behind the box and the cell beyond is
  floor.
- **A lamp lighting:** a box is pushed into a cell beside it.

| Arm (200 test worlds, lamps at columns 5–7) | Lamp lit within 50 steps | Steps when lit | Planning per episode |
|---|---|---|---|
| Planner on the transition model | **65.5%** | 8.8 | 0.05 s, 1 call |
| Planner on the world's rule (upper bound) | 65.5% | 8.9 | 0.02 s, 1 call |
| Random play | 13% | 20.7 | |

1. **World 1: fail** (65.5% against 90%). **The gate fails too:** the
   upper bound gets the same 65.5%.
   - **45 worlds cannot be solved.** Of the 69 worlds without a plan, a
     breadth-first search over the true world finds no way to light the
     lamp within 30 steps in 45. Usually the boxes sit against walls.
     About 155 of 200 worlds are solvable, so the 90% bar could not be
     met.
   - **The rest are a coverage gap.** On the solvable worlds the planner
     succeeds in 131 of 155 (84.5%), with the model and with the rule
     alike. The 24 it misses need a way of lighting the lamp that training
     never showed twice. Only 6 of the 12 (push direction, side of the
     lamp) pairs were learned. Pushing right into a lamp's left side, for
     instance, needs the lamp at column 4 or beyond, and training lamps
     stand at columns 1–3.
2. **Crafter: not run.** Its moves and their conditions are not yet right
   (105.8, 105.9).
3. **Compute:** 0.02–0.05 s of planning per episode.

## 8. Decision

**Revise** (2026-10-13). The planner on the learned model succeeded as
often as the planner on the world's own rule (65.5% each, over the same 200
worlds). It found plans in 84.5% of the solvable worlds (counted on the
rule's arm), from conditions it read itself and statements nobody wrote. The world-1 bar was infeasible: about a quarter of
the test worlds have no solution. Two changes for the next card:
- **Count only solvable worlds**, checked by search on the true world
  (scoring only).
- **Find the missing variants in imagination.** The model already predicts
  lamps at new columns and from every side (105.6). Variants never seen
  twice in training should come from interventions on imagined states
  around the goal, not only from remembered events.

Crafter waits for moves to be right.
