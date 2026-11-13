---
id: result/122-cube-found-at-256
type: result
title: cube found at 256
card: '122'
model: null
env: environment/ogbench-visual-scene
measure: cube found by the touch signal at 256 pixels
value: 0.61
against: 0.6
n: null
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/frozen-pretrained-patch-features
---
At 256-pixel frames the cube is the exact top patch in 61% of touches, against 36% at 64 pixels and a bar of 60% ([card 122](../../experiments/122-sharper-frames/card.md)).
