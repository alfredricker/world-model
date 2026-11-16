---
id: result/141-current-readings-under-the-probe-where-the-agent-fails
type: result
title: the current readings sit 17 to 28 points under the probe in the cells where the agent fails today
card: '141'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy on same or different, cube in the open drawer, current readings (%)
value: 70.6
against: 99.0
n: 2767
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
- rel: instance
  to: problem/reading-resolution
  weight: 1.0
  weight_basis: counted
---
With one cube in the open drawer, the current readings score 70.6 on telling whether the cube differs between two frames, against 99.0 for the labelled probe over the same features (350 same and 2,417 different pairs). For slides differing by 4 to 10 cm the drawer scores 81.8 against 98.7, and the window 76.6 against 99.2. The gaps are 28, 17 and 23 points, so the test shows the known faults ([card 141](../../experiments/141-representation-test-bed/card.md)).
