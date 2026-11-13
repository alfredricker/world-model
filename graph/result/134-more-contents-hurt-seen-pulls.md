---
id: result/134-more-contents-hurt-seen-pulls
type: result
title: adding the last effect per object lowers accuracy where the lock was already seen
card: '134'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock touched earlier, 400 episodes
value: 86.3
against: 87.5
n: 1534
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
Criterion 2 fails as declared. On the 1,534 pulls with the lock touched earlier (143 do not move), adding the items drops balanced accuracy from 88.5 to 86.3, against an allowed drop of 1 point (a floor of 87.5). Reopenings are unchanged at 93.4. Extra items at equal weight dilute the key where it already worked. The weight is an estimate. Numbers are in [results.json](../../experiments/134-unseen-state-from-effects/results.json); see [card 134](../../experiments/134-unseen-state-from-effects/card.md).
