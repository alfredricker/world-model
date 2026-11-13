---
id: result/124-slots
type: result
title: slots
card: '124'
model: null
env: environment/ogbench-visual-scene
measure: mean overlap of the best token with each object (best arm)
value: 0.28
against: 0.4
n: null
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/object-slots
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
Slot tokens trained on the frozen features reached a mean overlap of 0.28 (bar 0.40, k-means 0.29); they found the cube but no button, and no token carried the lock ([card 124](../../experiments/124-tokens-from-consequences/card.md)).
