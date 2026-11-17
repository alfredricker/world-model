---
id: result/144-bounded-shift-keeps-precision-and-carries
type: result
title: a shift bounded to one patch keeps the smooth map's reach and most of the offset's precision
card: '144'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube in a third of the table withheld from training, cm
value: 1.21
against: 3.32
n: 1650
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/find-then-place
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
A side arm of [card 144](../../experiments/144-a-centre-that-means-the-middle/card.md), not its criterion. With og4's weights frozen, a second-order map from their centre, and a shift of each patch's location computed from its features and bounded to one patch, the cube is read to 1.21 cm in a withheld third of the table (og4's form 3.32; the map alone 1.84) and 1.89 cm more than 5 cm past its edge (5.86; 2.66). Inside the range it reads 0.61 cm (og4's form 0.45; the map alone 1.09). Three seeds: 1.13, 1.33, 1.17.
