---
id: result/132-ten-episodes-too-few
type: result
title: ten stored episodes are too few even with the true key
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), true-value key, store of 10 episodes, on reopenings and on pulls
value: 58.9
against: 85.0
n: 2226
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
Criterion 3 of card 132 asked for at least 85% on both measures from a store of 10 episodes. With the true hidden values as the key, a store of 10 episodes gave 58.9% on reopenings and 65.0% on pulls, against the 85% bar. A store of 30 gave 73.2% and 81.8%, and 100 gave 98.4% and 93.8%, against 100.0% and 94.8% at 400 episodes. Ten episodes hold about one reopening with the cube, so a majority of five neighbours cannot return it. No problem node covers learning from few examples, so this is filed under the recall key, and the weight is an estimate. Numbers are in [results.json](../../experiments/132-memory-test-bed/results.json).
