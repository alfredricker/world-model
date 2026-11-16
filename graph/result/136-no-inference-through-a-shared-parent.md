---
id: result/136-no-inference-through-a-shared-parent
type: result
title: recall cannot infer a parent's state from another child's outcome, so unseen parents leave it below the base rate
card: '136'
model: model/og4
env: environment/memory-world
measure: log loss on cold touches of the chance, two-parent and rewired children (recall, worst case)
value: 0.708
against: 0.693
n: 722
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/conditions-by-contrast-over-touches
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
This is from a breakdown run after the card, not a declared measure. On cold touches of the chance, two-parent and rewired children, recall is worse than the base rate: 0.675 against 0.617, 0.558 against 0.515 and 0.708 against 0.693. The best possible answer gains over the base rate (0.532, 0.391 and 0.510) by reasoning from one child's outcome to a parent that another child shares. Recall reads each object's own record, so it cannot carry belief about an unseen parent from one child to another. The weight is an estimate. Numbers are in [results.json](../../experiments/136-memory-world/results.json); see [card 136](../../experiments/136-memory-world/card.md).
