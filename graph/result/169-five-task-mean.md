---
id: result/169-five-task-mean
type: result
title: five-task mean with chains searched over the scene model
card: '169'
model: model/og4
env: environment/ogbench-visual-scene
measure: mean success over the five visual-scene tasks, percent, 750 episodes
value: 90.5
against: 53.6
n: 750
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: tested
  to: principle/conditions-from-own-failed-try
- rel: tested
  to: principle/own-tries-first
- rel: measures
  to: ability/scene-five-task-mean
---
90.5% (90.4, 90.4, 90.8 over three evaluation seeds) against og4's 53.6%; published GAS 54.4% and HIQL 49.0% learn from 64-pixel frames alone and are not like for like. Search costs 2.9 ms per decision. Without the condition search, tasks 2 and 4 score 0% ([card 169](../../experiments/169-zero-heuristic-event-bfs-on-ogbench/card.md)).
