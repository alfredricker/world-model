---
id: result/129-mixture-read-out
type: result
title: mixture read out
card: '129'
model: model/og2
env: environment/ogbench-visual-scene
measure: cube within 1 cm, mixture read-out
value: 0.4
against: 0.9
n: 800
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/mixture-read-out-over-all-targets
- rel: instance
  to: problem/heterogeneous-state
  weight: 0.5
  weight_basis: estimated
- rel: instance
  to: problem/object-discovery
  weight: 0.5
  weight_basis: estimated
---
A single mixture over all targets put the cube within 1 cm in 40% of 800 held-out frames, against a bar of 90% ([card 129](../../experiments/129-one-skill-one-kind-of-variable/card.md), gate).
