---
id: result/171-needless-touches-lose-put-away
type: result
title: a search with effects from the model spends a seventh of its touches on parts already at the goal
card: '171'
model: model/og5
env: environment/ogbench-visual-scene
measure: locked put-away self-made problems solved, percent of 100
value: 64.0
against: 75.0
n: 100
outcome: fail
lost_points: 11
links:
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: instance
  to: problem/relations-between-variables
  weight: 1.0
  weight_basis: counted
---
With chain training and the range, card 170's search solves 64% of locked put-away problems against 57% with card 168's weights and 75% for og5 (separate samples of 100; a difference is good to about 7 points). In put-away episodes 11-15% of the model-driven search's touches go to the window or its lock while both are at the goal, against 1.8% for og5; 20 of 36 failures have two or more. Replacing untouched slots by recalled twins flips 0.4% of predictions on recorded touches, so the cause is not that; it is not yet known ([card 171](../../experiments/171-transition-head-trained-on-its-own-chains/card.md)).
