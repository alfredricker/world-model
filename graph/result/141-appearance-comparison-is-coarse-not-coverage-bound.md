---
id: result/141-appearance-comparison-is-coarse-not-coverage-bound
type: result
title: comparing by appearance barely depends on coverage but is coarse for small differences
card: '141'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy loss when remembered pairs in the withheld range are dropped, drawer 4 to 10 cm (points)
value: 2.9
against: 0
n: 6494
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/reading-resolution
  weight: 1.0
  weight_basis: counted
---
Dropping the remembered pairs with the slide 4 to 10 cm open costs the appearance reading only 1.5 to 2.9 points on pairs differing by 4 to 10 cm (drawer 81.8 to 78.9, window 76.6 to 75.1). It is coarse everywhere instead, scoring only 60 to 82% on differences below 10 cm, against 98.7 to 99.2 for the probe at 4 to 10 cm. The n is the number of drawer pairs in that cell (4,267 same and 2,227 different) ([card 141](../../experiments/141-representation-test-bed/card.md)).
