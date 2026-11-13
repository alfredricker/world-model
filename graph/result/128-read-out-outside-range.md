---
id: result/128-read-out-outside-range
type: result
title: read out outside range
card: '128'
model: model/og2
env: environment/ogbench-visual-scene
measure: read-out median error in cm outside the range of remembered grasps
value: 9.2
against: 0.43
n: null
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/aim-read-out-from-own-contact
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
Outside the range of remembered grasps the error was 9.2 cm, against 0.43 cm inside ([card 128](../../experiments/128-a-touch-moved-to-where-the-cube-is/card.md)).
