---
id: result/142-map-trained-with-the-score-is-unsteady
type: result
title: a map trained together with the score is unsteady
card: '142'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube on the table, map and score trained together on all labels, cm
value: 4.19
against: 0.45
n: 4793
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
Trained together with the score, the own map reads the table to 4.19 cm (og4's form 0.45) and the withheld third to 8.11, 1.49 and 1.49 cm over three seeds; the shared map to 8.22 cm. The same form fitted after the score reads 1.08 and 1.84 cm ([card 142](../../experiments/142-find-then-place/card.md)).
