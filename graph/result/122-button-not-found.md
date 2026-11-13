---
id: result/122-button-not-found
type: result
title: button not found
card: '122'
model: null
env: environment/ogbench-visual-scene
measure: button found by the touch signal at 256 pixels
value: 0.27
against: 0.6
n: null
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/frozen-pretrained-patch-features
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
The button itself was found in 27% of lock flips at 256 pixels (bar 60%, chance 8%); the handle outranks it ([card 122](../../experiments/122-sharper-frames/card.md)).
