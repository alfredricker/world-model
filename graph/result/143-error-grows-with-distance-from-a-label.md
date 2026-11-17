---
id: result/143-error-grows-with-distance-from-a-label
type: result
title: in the drawer the error grows with distance from a label
card: '143'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube's position in the open drawer, over 3 cm from the nearest release label, cm
value: 11.4
against: 0.9
n: 123
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
Within 1 cm of a release label the cube in the open drawer is read to 0.81 to 0.98 cm; 1 to 3 cm away, 2.3 to 3.8 cm; over 3 cm away, 7.6 to 15.8 cm (3 seeds). The cube is found; placing it is right only near the 66 positions training held ([card 143](../../experiments/143-found-where-it-was-let-go/card.md)).
