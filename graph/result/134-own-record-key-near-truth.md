---
id: result/134-own-record-key-near-truth
type: result
title: a key of the touched object's own record nearly reaches the true lock where the lock was seen
card: '134'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock touched earlier, key of the touched object's own record, 400 episodes
value: 95.0
against: 95.9
n: 1534
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/object-codes-with-per-object-memory
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
Post hoc (a check run after the result, not a declared arm): a key made only of the touched object's own record (identity, last seen value, last effect, and the two locks as last seen) lifts the 1,534 pulls with the lock touched earlier from 88.5 to 95.0, beside 95.9 for the true lock, with nothing trained. It reaches only 61.2 in the middle group, and reopenings fall to 59.4 from 93.4 because they depend on the cube's record. The weight is an estimate. Numbers are in [results.json](../../experiments/134-unseen-state-from-effects/results.json); see [card 134](../../experiments/134-unseen-state-from-effects/card.md).
