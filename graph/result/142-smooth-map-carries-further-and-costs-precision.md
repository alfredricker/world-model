---
id: result/142-smooth-map-carries-further-and-costs-precision
type: result
title: a smooth map carries further than the offset and costs precision inside
card: '142'
model: model/og4
env: environment/ogbench-visual-scene
measure: median error of the cube in a third of the table withheld from training, cm
value: 1.84
against: 3.32
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
A second-order map from the centre of the read-out's weights, fitted by least squares on the agent's own labels, reads the cube to 1.84 cm in a withheld third of the table and 2.66 cm more than 5 cm past its edge (og4's form: 3.32 and 5.86; bars 1.0 and 1.5). Inside the range it reads 1.05 cm against 0.45. One map shared by every object is 0.04 to 0.44 cm behind a map per object ([card 142](../../experiments/142-find-then-place/card.md)).
