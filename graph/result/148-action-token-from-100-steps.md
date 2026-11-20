---
id: result/148-action-token-from-100-steps
type: result
title: an action token settles to the action that gives a wanted effect after 100 steps of experience
card: '148'
model: model/og4
env: environment/sprite-world
measure: percent of test steps in which the completed action gives exactly the wanted effect, pairwise couplings, 100 steps of experience, mean of 5 streams
value: 95.6
against: 80.0
n: 6000
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/action-as-output-token
---
Criteria 1 to 3 of [card 148](../../experiments/148-the-action-as-an-output-token/card.md): with couplings learned by pseudolikelihood the settled action gives the wanted effect in 95.6% of 6,000 test steps after 100 steps of experience and in 100.0% after 1,000 (the commonest action 5.2%, a random one 1.1%); the settled action is the lowest-energy of the 162 in 100.0% of queries from 100 steps on. State and effect tokens come from the simulator and every query is a single step.
