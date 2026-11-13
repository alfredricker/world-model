---
id: result/129-handles-not-a-position
type: result
title: handles not a position
card: '129'
model: model/og2
env: environment/ogbench-visual-scene
measure: handles within 3 cm of the closing point (group read-out)
value: 0.655
against: 0.9
n: 800
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/one-position-variable-for-everything
- rel: instance
  to: problem/heterogeneous-state
  weight: 1.0
  weight_basis: estimated
---
Where the gripper meets a slide handle was read within 3 cm in 65.5% of frames, against a bar of 90% ([card 129](../../experiments/129-one-skill-one-kind-of-variable/card.md), gate); a position cannot be a slide's variable.
