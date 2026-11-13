---
id: result/133-learned-key-finds-the-lock
type: result
title: a key learned from recall's outcome finds the lock
card: '133'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%) on pulls with the lock variable shuffled (right-reason test)
value: 54.5
against: 65
n: 2226
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/recall-key-learned-from-recall-outcome
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
Shuffling the remembered lock value in the learned key (the network arm, trained on 400 episodes) dropped pull accuracy to 54.5%, under the bar of at most 65, so the key depends on the lock. Shuffling the cube's place changed pulls from 87.5% to 88.0%, a change of 0.5 points against an allowed 2, so the key ignores the irrelevant variable. Pulls rose from 76.7% with recall's fixed weights to 87.5% with the learned key, against 96.0% for a key with the true values. The weight is an estimate. Numbers are in [results.json](../../experiments/133-key-learned-from-recall-outcome/results.json); see [card 133](../../experiments/133-key-learned-from-recall-outcome/card.md).
