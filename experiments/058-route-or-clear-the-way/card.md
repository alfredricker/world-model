---
id: "058"
title: walking compares a route with clearing the way
rung: 1
serves: [P21, P12, P17]
status: done
verdict: fail
arch_version: 10
date: 2026-10-05
---

# 058: walking compares a route with clearing the way

Drafted and started overnight on 2026-10-05, from card 056's fork misses
(AGENTS.md, "Overnight sessions": no waiting for the user at a split).
The numbers below were fixed before any run.

## 1. Question

When a route to the next condition exists but clearing a key or door out
of the way would be shorter, does pricing both chains and taking the
cheaper raise rung 1's interaction-decisive forks to ≥ 90% without
costing version 10 anything elsewhere? Rung 1 (its first criterion);
P21 (System 2 compares candidate chains of conditions), P12, P17.

Card 056's misses, traced: at forks where the fastest first action is
picking up a key that lies between the agent and the switch or the door,
the agent walks round (82–89% of forks right, the same with the goal
written in). Card 045's walking takes any route that exists and counts
a token walkable (the condition ("walk", j)) only when none exists.

## 2. What changes

One component, walking (card 045, with card 051's pairs).

| | Version 10 | This card |
|---|---|---|
| A route exists | take it | also price every chain that clears one token on the way: the route's cost with that token walkable, plus one step for the act; pursue the cheapest such ("walk", j) if it beats the route, else the route |
| No route | ("walk", j), then pairs | unchanged |

The clearing act's own needs (an empty hand for a pick up) are not
priced; if they cannot be met, the route stands.

## 3. Dependencies

Card 056's goals and fork test (with its revision: one frame per
episode); card 054's encoder with identity up to noise; version 10.

## 4. Data check

As card 056 (forks from play on the test layouts, 30 per world) and
card 054's step D (chained rooms, cluttered world, 100 layouts each).

## 5. Feasibility gate

- **Upper bound:** the evaluator's search (every fork's fastest actions).
- **Trivial baseline:** version 10's walking (card 056: 82–89%).

## 6. Success criteria and prediction

Encoder of seed 399 (card 056: the four encoders behave identically).
1. **Forks:** with the goal written in, top-1 ≥ 90% in every familiar
   world, above version 10's walking in every world.
2. **No regression on the episode's end:** familiar worlds 100% with
   mean steps no higher than version 10's; chained rooms with one door
   and the cluttered world ≥ 98%, steps within 2% of version 10's.
3. **No regression on goals from frames:** card 056's acting ≥ 99% per
   world, steps no higher.

**Prediction.** Forks rise to about 95%; the remaining misses are
clearing acts that need the hand emptied first. Steps fall slightly.

**Decision rules.** Keep if all three hold (version 11 walking). Stop
if criterion 2 fails.

**Budget.** About 30 minutes.

## 7. Result

Seed 399's encoder (`runs/058/planner_399.out`; version 10 on the same
encoder: 100% in every familiar world, card 054).

| | Key | Switch | Either | Both |
|---|---|---|---|---|
| 2. Success, episode's end | 90% | 100% | 96.7% | 73.3% |
| 2. Mean steps (version 10) | 15.4 (15.9) | 16.2 (16.7) | 14.2 (14.4) | 21.4 (21.0) |

Criterion 2 fails in three of four worlds. Traced on the both world
(`tools/card058/trace.py`): in all eight failed layouts the agent turns
left, then right, for all 200 steps. Each step the clearing chain is
priced again; a turn changes the route's cost and the clearing chain's
by different amounts, so the cheaper chain flips every step (for
example ("walk", 123) through the hand one step, through facing the
door the next). Card 051's commitment keeps the achiever chosen for a
condition, but not walking's choice between a route and a clearing
chain. Criteria 1 and 3 were not measured: those runs ran out of GPU
memory while ten jobs shared it, and were not rerun once criterion 2
had decided the card.

## 8. Decision

**Stop.** As declared, criterion 2 failed. Comparing candidate chains by
cost is still the right principle (P21), but a comparison remade every
step needs the choice kept until it fails, as card 051 keeps achievers;
otherwise near-equal chains alternate. A later card can price clearing
inside the committed chain, with the clearing act's own needs (an empty
hand) in its cost.
