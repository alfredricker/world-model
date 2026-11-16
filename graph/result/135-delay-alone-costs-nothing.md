---
id: result/135-delay-alone-costs-nothing
type: result
title: a longer delay alone costs this memory nothing when nothing changes out of sight
card: '135'
model: model/og4
env: environment/ogbench-visual-scene
measure: share of pulls (%) where the remembered lock equals the true lock, nothing hidden
value: 100
against: 100
n: 1408
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/belief-keeps-last-value
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
With nothing hidden, the remembered lock equals the true lock in all 1,408 pulls checked, whatever the delay since it was seen. Accuracy does fall with the number of touches since the lock was seen, from 98.4 at 1 touch to 89.3 at 8 or more, but the true-lock key falls almost the same way (98.2 to 88.4), so the fall comes from pulls that fail with the lock open and not from forgetting. Nothing in this scene changes unless the agent touches it, so a delay of any length tests nothing; the test bed had to hide touches made by another hand. The weight is an estimate. Numbers are in [results.json](../../experiments/135-state-that-changes-out-of-sight/results.json); see [card 135](../../experiments/135-state-that-changes-out-of-sight/card.md).
