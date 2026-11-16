---
id: result/135-passive-memory-cannot-recover-unseen-change
type: result
title: passive memory cannot recover a change it never saw, even knowing how old the memory is
card: '135'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock touched earlier, a quarter of touches hidden, recall with the plain record
value: 79.5
against: 96.2
n: 1032
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/belief-keeps-last-value
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
When a quarter of the touches are made by another hand and not seen, the remembered lock is wrong in 13.9% of the 1,032 pulls with the lock touched earlier. Recall with card 134.1's record scores 79.5 balanced accuracy against 96.2 for the true lock (95.3 and 96.1 with nothing hidden). Adding the time since the object was last seen to the record moves the log loss only from 0.214 to 0.192, against 0.055 for the true lock, so it closes about a seventh of the gap; a small trained network on the same record does no better (0.169). With half of the touches hidden the age adds nothing (log loss 0.282 to 0.271). A change that happened out of sight is not in the store, so the agent has to look again. The weight is an estimate. Numbers are in [results.json](../../experiments/135-state-that-changes-out-of-sight/results.json); see [card 135](../../experiments/135-state-that-changes-out-of-sight/card.md).
