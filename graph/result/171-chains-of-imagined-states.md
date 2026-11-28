---
id: result/171-chains-of-imagined-states
type: result
title: training on chains of imagined states helps the put-away route, not recorded three-touch chains
card: '171'
model: model/og5
env: environment/ogbench-visual-scene
measure: whole state after three imagined touches matches the reading, percent, 1,384 held-out chains
value: 69.9
against: 73.2
n: 1384
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: counted
---
Predicting from an imagined state costs 3.0 points on recorded three-touch chains (62.6% against 65.6% with every touch predicted from the real reading); one-touch errors adding up are the larger loss. Chain training brings no gain there beyond the range (69.9% against 69.0%). On the route every put-away needs it does: imagined from the start of 60 problems, the route reaches the goal for 38 with chain training against 19 without ([card 171](../../experiments/171-transition-head-trained-on-its-own-chains/card.md)).
