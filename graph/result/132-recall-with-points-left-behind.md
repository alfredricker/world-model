---
id: result/132-recall-with-points-left-behind
type: result
title: recall keyed on touch plus points left behind
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: AUC, predicted next frame shows the cube inside vs an empty drawer, on drawer reopenings (closed under 3 cm, then over 10 cm)
value: 0.67
against: 0.998
n: 198
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/object-permanence
  weight: 1.0
  weight_basis: estimated
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
- rel: tested
  to: principle/episode-history-in-the-recall-key
---
Recall keyed on the touch plus episode history scored AUC 0.67 on 198 drawer reopenings (40 with the cube inside, 158 without), against 0.40 for the touch alone and 0.998 for real frames. The history is the points left behind: where the hand released after carrying something 5 cm or more, kept until touched again. Error on changed patches was 1.41 against 2.33 for copying, and the gap between its two predictions was 19% of the real gap. It falls short because the hand-written key is noisy. A left-behind point in the drawer region is present for 88% of with-cube reopenings but also 51% of empty ones (drawer-handle releases and stale points). So only 32% of recalled neighbours for a with-cube reopening are with-cube reopenings, against 6% when the drawer is truly empty. This is a diagnostic from the uncarded gate runs (`tools/card132/recall.py`, `runs/132/recall_hist*.json`); no card is written yet, and the weights are estimates.
