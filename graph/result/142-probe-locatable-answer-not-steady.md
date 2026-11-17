---
id: result/142-probe-locatable-answer-not-steady
type: result
title: the labelled probe's "can the cube be located" answer is not steady
card: '142'
model: null
env: environment/ogbench-visual-scene
measure: hidden-cube pair cell of the labelled probe, lowest of 5 seeds
value: 82.4
against: 94.6
n: 1394
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/absence-not-a-value
  weight: 1.0
  weight_basis: counted
---
Over 5 seeds the labelled probe's hidden-cube pair cell runs from 82.4 to 94.6, and with the drawer shut it calls the cube locatable in 3 to 29% of frames. Its value cells are steady (cube 0.16 to 0.17 cm on the table, 0.30 to 0.40 in the open drawer), and with a gap its smallest slide cells score 97.9 and 95.3 ([card 142](../../experiments/142-find-then-place/card.md)).
