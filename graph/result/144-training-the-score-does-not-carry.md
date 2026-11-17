---
id: result/144-training-the-score-does-not-carry
type: result
title: training the score until its centre is exact inside the range does not carry outside it
card: '144'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube in a third of the table withheld from training, cm
value: 1.94
against: 1.84
n: 1650
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/find-then-place
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
The declared arm of [card 144](../../experiments/144-a-centre-that-means-the-middle/card.md). With the smooth map fixed and the cube group's score trained on, the table is read to 0.67 cm (the map alone 1.09; og4's form 0.45), but the withheld third to 1.94 cm (the map alone 1.84; bar 1.5) and the open drawer to 1.76 cm (og4's form 1.70; bar 1.0). The open drawer's error is one seed of three in which the first stage finds the cube weakly (2.4 to 3.2 cm in every arm; the other seeds 1.1 to 1.3).
