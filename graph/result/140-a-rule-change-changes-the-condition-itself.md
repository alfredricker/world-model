---
id: result/140-a-rule-change-changes-the-condition-itself
type: result
title: when the rules change, both which stored episodes answer and which records are in the key must change
card: '140'
model: model/og4
env: environment/memory-world
measure: gap closed (%), warm touches of the rewired child, mean of three worlds (check run afterwards, both changed)
value: 85
against: 62
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/conditions-by-contrast-over-touches
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
This is a check run afterwards, not a declared arm of [card 140](../../experiments/140-which-period-of-the-past-is-the-present-like/card.md). On warm touches of the rewired child, card 138's answer closes 62% of the gap. Letting only the stored episodes wired as now answer gives 72%, a key made of the object's own record plus the true parent now gives 73%, and both together give 85%. Which stored episodes answer and which records are in the key each cost about 10 points, and both together 23. The card says the reference of its gate was badly built, because it kept a key chosen on both wirings together, and that mixing the answers of periods cannot recover the second half, since each period would need its own choice of records. The weight is an estimate. Numbers are in [results.json](../../experiments/140-which-period-of-the-past-is-the-present-like/results.json) under the post-hoc block; see [card 140](../../experiments/140-which-period-of-the-past-is-the-present-like/card.md).
