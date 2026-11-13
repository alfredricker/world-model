---
id: result/132-recall-touch-alone
type: result
title: recall keyed on the touch alone
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: AUC, predicted next frame shows the cube inside vs an empty drawer, on drawer reopenings (closed under 3 cm, then over 10 cm)
value: 0.40
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
---
Recall of the 5 nearest remembered touches, keyed on the touch alone, scored AUC 0.40 on 198 drawer reopenings (40 with the cube inside, 158 without), below chance 0.5; real frames reach 0.998. Error on changed patches was 1.32 against 2.33 for copying the last frame. The touch alone does not say whether the cube is inside. This is a diagnostic from the uncarded gate runs (`tools/card132/recall.py`, `runs/132/recall_hist*.json`); no card is written yet, and the weights are estimates.
