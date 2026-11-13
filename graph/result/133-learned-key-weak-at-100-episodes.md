---
id: result/133-learned-key-weak-at-100-episodes
type: result
title: learned recall keys are weak when trained on only 100 episodes
card: '133'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%) on pulls, trained on 100 episodes
value: 80.8
against: 85
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
Trained on 100 episodes, the per-item weights scored 80.8% on pulls and the network 73.3%, against a bar of 85 and against 71.5% for fixed weights. The per-item weights then broke at 400 episodes (64.1% on pulls) because one scale grew without bound, a training fault and not a finding about the idea. The weight is an estimate. Numbers are in [results.json](../../experiments/133-key-learned-from-recall-outcome/results.json); see [card 133](../../experiments/133-key-learned-from-recall-outcome/card.md).
