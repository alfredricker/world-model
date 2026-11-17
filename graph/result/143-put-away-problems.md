---
id: result/143-put-away-problems
type: result
title: put-away problems
card: '143'
model: model/og4
env: environment/ogbench-visual-scene
measure: share of 50 put-away problems in which the cube ends inside the drawer
value: 0.68
against: 0.0
n: 50
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/release-point-labels-the-object
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
With the cube on the table, the drawer open and a goal frame showing the cube in the drawer, og4 puts it there in 0 of 50 problems and with the release-label read-out in 34 of 50 (bar 35). When a remembered move that the hand never follows up is also allowed to end where the hand let go, 45 of 50; that addition with og4's read-out gives 0 ([card 143](../../experiments/143-found-where-it-was-let-go/card.md)).
