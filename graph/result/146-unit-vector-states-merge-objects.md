---
id: result/146-unit-vector-states-merge-objects
type: result
title: with perfect couplings, unit-vector states merge objects and beliefs over objects do not
card: '146'
model: model/og4
env: environment/tetrominoes
measure: FG-ARI of the settled grouping with perfect couplings, unit-vector state, K = 8
value: 85.2
against: 96.4
n: 320
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/grouping-as-settling
- rel: instance
  to: problem/object-discovery
  weight: 0.5
  weight_basis: estimated
---
From the gate of [card 146](../../experiments/146-grouping-as-settling-on-a-known-benchmark/card.md). Couplings of +1 between patches of one object and -1 otherwise. With each patch's state a unit vector the settled grouping scores 68.3 (K = 4) and 85.2 (K = 8) against the grid's ceiling of 96.4: a repelling pair of unit vectors is best satisfied by pointing opposite ways, and only two groups can be opposite. With each patch's state a belief over the K objects it scores 96.4 with K at the count and at twice it. The weight is an estimate.
