---
id: result/172-moments-loss-reads-every-variable
type: result
title: a loss over moments reads every scene variable, the cube first, with no decoder, touch or label
card: '172'
model: model/og5
env: environment/ogbench-visual-scene
measure: AUROC telling pairs that differ in the cube's place from pairs that differ in nothing, validation episodes, 5-minute label-free fit
value: 0.954
against: 0.49
n: 37323
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
Trained for five minutes to pick the near moment of an episode, a 32-number state tells the cube's moves at 0.954, the drawer at 0.902, the window at 0.826 and the locks at 0.899 and 0.918 (trivial baselines 0.56-0.67; pairs chosen by truth 0.96-0.999). A straight-line read-out recovers drawer, window and locks at R² 0.90-0.95. Card 145's state, trained an hour to redraw every patch, scored 0.49 on the cube. The arm's pose explains 0.7% of the state. Card 168's rule-built slots still read better (0.976-1.0) ([card 172](../../experiments/172-factors-from-sparse-change-over-time/card.md)).
