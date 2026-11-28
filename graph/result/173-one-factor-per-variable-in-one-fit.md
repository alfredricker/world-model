---
id: result/173-one-factor-per-variable-in-one-fit
type: result
title: a fixed temperature and a small box give one factor per scene variable in one label-free fit
card: '173'
model: model/og5
env: environment/ogbench-visual-scene
measure: variables (of five) whose change lies at least 80% in their own factor when only they differ over 5 steps, one seed, K = 8
value: 4
against: 4
n: 1
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
With the temperature fixed at 0.5, eight coordinates, a cost on change between consecutive steps and factors read from 5-step changes, one 5-minute fit emptied one coordinate and gave five factors, one per variable: shares 73, 96, 94, 92 and 83% (card 172: 8-19%). Read through its own factor a variable is told at AUROC 0.970-0.990 on validation (card 168's slots 0.976-1.0) and 0.937-0.987 on the frames before a touch (slots 0.862-0.994). One seed; the ladder of sizes and the other seeds were not run because the gate failed ([card 173](../../experiments/173-tight-box-sparse-change-factors/card.md)).
