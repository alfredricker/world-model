---
id: result/168-one-network-reads-what-og4-misread
type: result
title: one consequence-trained network reads the cube in and behind the drawer and half-open slides
card: '168'
model: model/og4
env: environment/ogbench-visual-scene
measure: mean balanced accuracy over the 9 pair cells of card 141's test bed outside 1-4 cm, percent, 300 test episodes
value: 94.9
against: 84.4
n: 3
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/absence-as-a-value
- rel: tested
  to: principle/release-point-labels-the-object
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: estimated
---
SceneWNet, one network over frozen DINOv2 patches trained on touch consequences without truth labels or a decoder, scores 94.9% against og4's 84.4% (mean of 3 training seeds). Cube in the open drawer 91.9% against 70.6%, read to 1.04 cm against 27.7 cm; cube hidden 92.4% against 61.0%; drawer and window open 4-10 cm 95.9% and 94.5% against 81.8% and 76.6%. Its training targets come from hand-set rules over touches, not from discovery ([card 168](../../experiments/168-consequence-trained-scene-world-model/card.md)).
