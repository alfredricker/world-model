---
id: "059"
title: how soon a goal is, from the chain of conditions
rung: 1
serves: [P12, P21, P6, P17]
status: done
verdict: fail
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

Seed 399's encoder (`runs/059/main_399.json`; revision
`rev_k10_399.json`). Pairs are states × goals where the goal can be
reached and does not yet hold: 497–512 per world.

| | Key | Switch | Either | Both |
|---|---|---|---|---|
| 1. Rank correlation, goals from five frames | 0.977 | 0.990 | 0.988 | 0.981 |
| 1. The same, goal written in | 0.978 | 0.990 | 0.989 | 0.981 |
| 2. Goal-swapped control | −0.28 | −0.05 | −0.10 | −0.23 |
| 3. Coverage, five frames | 96.4% | 95.9% | 94.9% | 93.8% |
| 3. Coverage, goal written in | 99.2% | 99.8% | 98.4% | 99.2% |
| Mean absolute error (steps), five frames | 0.29 | 0.32 | 0.29 | 0.43 |
| Predicted / true | 1.04 | 1.03 | 1.03 | 1.05 |
| Revision, ten frames: rank / coverage | 0.973 / 98.8% | 0.984 / 99.4% | 0.982 / 98.2% | 0.971 / 89.0% |

Seconds per estimate: 0.003–0.015. Per goal, the rank correlation is
0.95–0.99 except the switch goal in the key and both worlds (0.88),
where the chain walks round a key it could pick up (card 058's case),
so predictions run a few steps long.

Criteria 1 and 2 hold everywhere. Criterion 3 failed in two worlds with
five frames: some inferred goals carried a companion feature that cannot
be reached, so no chain exists (card 056). **The declared revision**,
ten frames, fixed three worlds and made the both world worse: random
play opened that world's door in only 1–8 of 600 episodes, so ten frames
from different episodes cannot be drawn and the inferred door goals are
mostly companions.

## 8. Decision

**Revise.** How soon a goal is, read from the chain of conditions, ranks
states almost exactly as the evaluator's fewest steps do (0.97–0.99,
about 0.3 steps off, a few milliseconds per estimate), with goals from
frames and written in alike, and without imagining a single primitive
step. It is not kept only because criterion 3 depends on goal examples
that random play cannot supply in the both world. With rung 1's first
criterion (card 056's forks, 82–89%) still short, rung 1 is not passed;
the two open pieces are walking that keeps its choice of clearing the
way (after card 058) and goal examples from demonstrations where random
play is too thin (pinned in STATUS).
