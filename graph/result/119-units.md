---
id: result/119-units
type: result
title: units
card: '119'
model: null
env: environment/ogbench-visual-scene
measure: overlap of the best group of change with the drawer
value: 0.17
against: 0.3
n: null
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/frozen-pretrained-patch-features
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
Grouping patches by where they change split each object into four groups covering only the strip a slide sweeps (drawer overlap 0.17, k-means 0.30; buttons 0.00) ([card 119](../../experiments/119-variables-from-lasting-change/card.md)). Against value is the k-means overlap.
