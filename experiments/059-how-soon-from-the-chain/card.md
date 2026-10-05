---
id: "059"
title: how soon a goal is, from the chain of conditions
rung: 1
serves: [P12, P21, P6, P17]
status: approved
verdict:
arch_version: 10
date: 2026-10-05
---

# 059: how soon a goal is, from the chain of conditions

Drafted and started overnight on 2026-10-05 (AGENTS.md, "Overnight
sessions"), as rung 1's second criterion after card 056 tested its
first. The numbers below were fixed before any run.

## 1. Question

From any state, can the agent predict how many steps a goal given as
frames is away, by following its chain of conditions in imagination
one condition at a time, and do those predictions rank states as the
evaluator's fewest steps do? Rung 1 (CHARTER: "the rank correlation
between predicted and true steps-to-goal over fork states"); P12 ("from
any state the agent predicts how likely it is to reach the goal, and how
soon"), P21 (over conditions, not imagined steps), P6, P17.

## 2. What changes

One component: a reading of the planner, added; acting is unchanged.

| | Version 10 | This card |
|---|---|---|
| How soon | walking's cost to face one thing (card 045's how-soon check) | for a goal: repeat (ask the planner for its chain; take the chain's next act on a thing; add walking's cost to face the thing, plus one step; imagine the situation after the act, the agent where walking brings it) until the goal holds; at most 8 conditions; no chain means "not reachable" |

Nothing is searched over action sequences: each round is one call of the
planner as in acting, and walking is card 045's learned approach cost.

## 3. Dependencies

Card 056's goals from frames (with its revision: one frame per
episode); card 045's walking costs; card 054's encoder; version 10.
The evaluator's fewest steps: breadth-first search on the simulator.

## 4. Data check

150 states per familiar world from random play on the 30 test layouts,
each with card 056's four goals: about 600 pairs per world, minus those
where the goal already holds.

## 5. Feasibility gate

- **Upper bound:** the goal written in (supplied arm).
- **Trivial baseline:** the goal-swapped control: the prediction for
  another goal's frames, against this goal's fewest steps.

## 6. Success criteria and prediction

Encoder of seed 399 (the four encoders behave identically, card 056).
1. **Rank:** rank correlation between predicted and true steps ≥ 0.8 in
   every familiar world, goals from frames.
2. **Above the control** by ≥ 0.4 in every world.
3. **Coverage:** a prediction for ≥ 95% of states where the goal can be
   reached and does not yet hold.

Reported: mean absolute error, the ratio predicted / true, per goal, and
seconds per estimate (P17).

**Prediction.** Rank about 0.9: walking costs ranked placements at
card 045's check, and the chains are short (one to three acts).
Predictions run low where the chain picks a detour or ignores turns at
arrival.

**Decision rules.** Keep (version 11: how soon from the chain) if all
three hold. One declared revision if one fails. Stop if the rank
correlation is below the control's.

**Budget.** About 15 minutes.

## 7. Result

## 8. Decision
