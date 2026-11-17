---
id: result/143-cube-read-in-the-open-drawer
type: result
title: cube read in the open drawer once releases are labels
card: '143'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube's position with the cube in the open drawer, cm
value: 1.7
against: 27.7
n: 985
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/release-point-labels-the-object
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
With release points added to the read-out's training signal, the cube in the open drawer is read to 1.70 cm (3 seeds; og4 27.7; bar 1.0; the same frames with true positions 0.93). On the table the reading improves from 0.60 to 0.45 cm. og4 had 2 training frames with the cube in the open drawer, because recorded play puts the cube in 119 times and takes it out once ([card 143](../../experiments/143-found-where-it-was-let-go/card.md)).
