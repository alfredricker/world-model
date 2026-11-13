---
id: result/133-learned-key-misses-the-bars
type: result
title: a learned recall key falls just short of the accuracy bars
card: '133'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls (does the slide move)
value: 87.5
against: 88
n: 2226
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/recall-key-learned-from-recall-outcome
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
The learned key scored 87.5% on 2,226 pulls against a bar of 88, and 93.2% on 171 drawer reopenings where the cube was seen earlier against a bar of 95, while recall's fixed weights score 96.5% on those reopenings. So the learned key gains about 11 points on pulls over fixed weights (76.7%) but loses about 3 points on reopenings, and misses both bars narrowly. The weight is an estimate. Numbers are in [results.json](../../experiments/133-key-learned-from-recall-outcome/results.json); see [card 133](../../experiments/133-key-learned-from-recall-outcome/card.md).
