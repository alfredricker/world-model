---
id: result/171-pull-end-as-a-range
type: result
title: a pull's end predicted as a range matches the reading, and agrees with the benchmark's tolerance
card: '171'
model: model/og5
env: environment/ogbench-visual-scene
measure: a pull's read end inside the predicted 10th-90th percentile range, percent, 472 held-out pulls
value: 83.5
against: 85.0
n: 472
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: instance
  to: problem/reading-resolution
  weight: 1.0
  weight_basis: counted
---
With two more outputs trained by the quantile loss, the read end of a pull falls in the predicted range for 83.5% of pulls with this card's model (bar 85%) and 89.1% with the range alone, against 69.0% for one value and a threshold. Live, 642 of 644 slides the agent took to be at the goal were truly within the benchmark's 4 cm, which the agent is not told. The range alone did not change live success (55% against 57% on locked put-away) ([card 171](../../experiments/171-transition-head-trained-on-its-own-chains/card.md)).
