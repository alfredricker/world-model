---
id: result/151-attractors-learned-at-three-levels
type: result
title: attractors at three levels are learned in one shot, locally and without labels, and are the parts the world shows
card: '151'
model: model/og4
env: environment/world-of-parts
measure: percent of output tokens right at noise 4.0 with learned patterns, mean of 5 worlds
value: 92.0
against: 90.0
n: 20000
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/layers-of-attractor-modules
---
Criterion 2 of [card 151](../../experiments/energy-modules/151-layers-of-attractor-modules-in-a-world-of-parts/card.md), failed as worded: it asked for exactly 16, 24 and 32 patterns in every world, and world 5 shows only 15 of its patches, which is what was stored there. In worlds 1 to 4 the layers hold 16, 24 and 32 patterns, one per true patch, object and scene, from 2,000 unlabelled scenes; the stack with learned patterns scores 92.0% against 92.0% with the true ones. A pattern is stored when nothing fits, kept when met twice, and merged with one it comes to fit; without the merge the first version left 17, 37 and 139.
