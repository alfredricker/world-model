---
id: result/132-latent-predictor-empty-drawer
type: result
title: latent predictor always predicts an empty drawer
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: AUC, predicted next frame shows the cube inside vs an empty drawer, on drawer reopenings (closed under 3 cm, then over 10 cm)
value: 0.66
against: 0.998
n: 198
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/object-permanence
  weight: 1.0
  weight_basis: estimated
- rel: tested
  to: principle/event-level-latent-state-prediction
---
A transformer over DINOv2 patch features (reduced to 64 dims) and a touch token, trained to predict the frame before the next touch, scored AUC 0.66 at best (weighted loss, no memory state) on 198 drawer reopenings (40 with the cube inside, 158 without). Chance is 0.5 and real frames reach 0.998. The other variants scored 0.49 to 0.64. A recurrent memory state added nothing (0.64 with, 0.66 without). Error on changed patches was 0.69 against 2.33 for copying the last frame, so it learns ordinary consequences and predicts the drawer opening. But it predicts an empty drawer every time: the gap between its two predictions is 2% of the real gap. There were only about 50 with-cube examples in training. This is a diagnostic from the uncarded gate runs (`tools/card132/gate.py`, `runs/132/gate_*.json`); no card is written yet, and the weight is an estimate.
