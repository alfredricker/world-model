---
id: result/131-belief-left-behind
type: result
title: belief left behind
card: '131'
model: null
env: environment/ogbench-visual-scene
measure: belief within 4 cm of the cube, closed drawer
value: 0.0
against: 0.975
n: 99
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/belief-keeps-last-value
- rel: instance
  to: problem/relations-between-variables
  weight: 0.6
  weight_basis: estimated
- rel: instance
  to: problem/object-permanence
  weight: 0.4
  weight_basis: estimated
---
With the cube in a closed drawer the belief was within 4 cm of it in 0% of 99 frames (median error 11.2 cm), because the drawer carried the cube while the belief kept the old place ([card 131](../../experiments/131-belief-out-of-view/card.md)). Weights are estimates.
