---
id: result/174-factors-hold-in-two-seeds-of-three
type: result
title: the label-free factoring holds in two seeds of three and loses the cube in the third
card: '174'
model: model/og5
env: environment/ogbench-visual-scene
measure: reading of the cube through its own factor (AUROC), mean of three seeds at six coordinates, 100 fresh episodes
value: 0.944
against: 0.95
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
Seeds 0 and 1 give five factors, one per variable, read at 0.973-0.991 with 79-96% of each variable's change in its factor and 2-8% stray movement; seed 2 gives four factors, the drawer on three coordinates and the cube on none (cube read at 0.872, stray movement 110%). The cube is the weak variable in every fit: never over 89%, lost at size 4, spread thin at 12 and 16. The held-out loss ranks the three seeds in the order of their factoring (0.931, 0.954, 0.989) ([card 174](../../experiments/174-do-the-five-factors-hold/card.md)).
