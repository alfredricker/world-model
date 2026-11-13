---
id: result/132-last-seen-read-badly-by-recall
type: result
title: last-seen contents are right but recall reads them badly
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), narrow view, last-seen key, on drawer reopenings and on pulls (does the slide move)
value: 70.2
against: 94.8
n: 2226
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: counted
- rel: tested
  to: principle/belief-keeps-last-value
---
In the narrow view, a key made of the last seen values scored 79.7% on 198 drawer reopenings and 70.2% on 2,226 pulls, against 100.0% and 94.8% for a key with the true hidden values. The remembered lock was right in 1,955 of the 2,226 pulls, so the loss is not in what was kept. It is in recall's fixed weighting of the key: every remembered variable counts equally, and the irrelevant ones (the cube's place, the other slide) outvote the lock. A rule that reads only the true lock scores 96.1%. On reopenings, last seen scores 89.5% when the cube was seen earlier in the episode (137 events) and 50.0% when it was not (61 events), and no memory of this episode can answer the latter. The weight is a count from the card. Numbers are in [results.json](../../experiments/132-memory-test-bed/results.json).
