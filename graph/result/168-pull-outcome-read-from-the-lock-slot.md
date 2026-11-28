---
id: result/168-pull-outcome-read-from-the-lock-slot
type: result
title: whether a pull moves the slide is predicted with the lock read from the scene
card: '168'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy of predicting that a drawer or window pull moves the slide, percent, 1,678 held-out pulls
value: 91.5
against: 82.7
n: 3
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: tested
  to: principle/conditions-asked-of-the-effect-model
- rel: instance
  to: problem/relations-between-variables
  weight: 1.0
  weight_basis: estimated
---
SceneWNet predicts pull outcomes at 91.5% (drawer 88.0, window 96.5; bar 88.0; with true locks 94.8). Against: a network given only the pre-touch action scores 82.7%, so the scene adds 8.8 points, not the 41.5 a 50% floor would suggest. Swapping the lock slot flips 97.1% of correct 'moves' predictions; an imagined button press flips 80.2% ([card 168](../../experiments/168-consequence-trained-scene-world-model/card.md)).
