---
id: result/141-frozen-features-hold-what-the-readings-lack
type: result
title: the frozen features hold the cube in the open drawer and a slide's opening, which the current readings miss
card: '141'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error locating the cube in the open drawer (cm), labelled probe over the frozen features
value: 0.39
against: 27.7
n: 985
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/frozen-pretrained-patch-features
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
A small attention probe trained on the true state, over the same frozen features the agent already uses, locates the cube in the open drawer to a median of 0.39 cm. The agent's current reading is off by a median of 27.7 cm there. The same probe reads a slide's opening to a median of 0.12 cm, where the current readings give no value at all. The 985 is the number of test frames with the cube in the open drawer ([card 141](../../experiments/141-representation-test-bed/card.md)).
