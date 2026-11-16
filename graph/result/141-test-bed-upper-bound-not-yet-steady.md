---
id: result/141-test-bed-upper-bound-not-yet-steady
type: result
title: the test bed's upper bound is not yet steady, so it cannot yet serve as a bar
card: '141'
model: model/og4
env: environment/ogbench-visual-scene
measure: same-or-different cells where the labelled probe scores at least 95 (of eleven)
value: 7
against: 11
n: 11
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 0.2
  weight_basis: estimated
---
The labelled probe is under 95 balanced accuracy in four of eleven cells: both slides at 1 to 4 cm (87.4, 82.3), the hidden cube (82.4) and the window's lock (92.3). A check run afterwards, not a declared arm, shows the 1 to 4 cm cells rise to 97.8 (drawer) and 95.1 (window) when "same" and "different" are separated by a gap. The probe's answer to whether the cube can be located also differs between two fits: 29% against 2% of 168 shut-drawer frames called locatable. The weight is an estimate ([card 141](../../experiments/141-representation-test-bed/card.md)).
