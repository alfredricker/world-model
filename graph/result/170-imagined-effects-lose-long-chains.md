---
id: result/170-imagined-effects-lose-long-chains
type: result
title: effects imagined by the scene model keep short chains and lose long ones
card: '170'
model: model/og5
env: environment/ogbench-visual-scene
measure: locked put-away self-made problems solved, percent of 100
value: 58.0
against: 81.0
n: 100
outcome: fail
lost_points: 23
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: instance
  to: problem/relations-between-variables
  weight: 1.0
  weight_basis: counted
---
With each imagined next state taken whole from SceneWNet, every kind of touch offered and no guards, the agent solves 58% of locked put-away problems against og5's 81% (100 problems each from one generator, not the same draws: card 171's correction), and 87% against 83% on mixed problems. 21 of 42 failures repeat or alternate touches; 12 give up. Search costs 9-15 ms per decision against 2 ms ([card 170](../../experiments/170-effects-imagined-by-the-scene-model/card.md)).
