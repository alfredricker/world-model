---
id: result/156-the-agents-takes-always-move
type: result
title: the agent that grips and moves at once leaves no record of a take without a move, and its take action fails the test
card: '156'
model: model/og4
env: environment/sprite-world
measure: percent of never-met objects for which the take action is right, agent arm, mean of 10 worlds
value: 75.0
against: 80.0
n: 10
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/appearance-couplings-from-own-action
---
From [card 156](../../experiments/energy-modules/156-appearance-couplings-taught-by-the-agents-own-action/card.md): the script arm reaches 93.9. Checked: of 103 takes in 50 episodes of the agent none is without a move (script: 35 of 150), and the test asks for a take without a move; with the true classes given, couplings fitted to the agent's transitions also score 0% on takeable objects. The binding and the classes are as good as the script's. Card 149's lesson again: couplings have no answer for a combination their records lack.
