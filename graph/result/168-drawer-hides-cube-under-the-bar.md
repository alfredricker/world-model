---
id: result/168-drawer-hides-cube-under-the-bar
type: result
title: closing the drawer on the cube is predicted below the bar
card: '168'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy of predicting that a drawer touch hides the cube lying in the open drawer, percent
value: 82.2
against: 85.0
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: instance
  to: problem/object-permanence
  weight: 1.0
  weight_basis: estimated
---
On drawer touches made while the cube lies in the open drawer, SceneWNet predicts 'hidden afterwards' at 82.2% (83.5, 81.0, 82.1) against a bar of 85.0%. Over every touch made in that situation (101 that hide, 814 that do not) it scores 87.6%. og4 has no such prediction ([card 168](../../experiments/168-consequence-trained-scene-world-model/card.md)).
