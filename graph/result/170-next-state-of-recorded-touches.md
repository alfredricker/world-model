---
id: result/170-next-state-of-recorded-touches
type: result
title: the scene model's next state matches the reading after presses and moves, less often after pulls
card: '170'
model: model/og5
env: environment/ogbench-visual-scene
measure: pulls that move a slide whose predicted value is within the threshold of the reading after, percent, 467 held-out pulls
value: 69.0
against: 79.9
n: 467
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/effect-model-over-variables
- rel: instance
  to: problem/reading-resolution
  weight: 1.0
  weight_basis: counted
---
On 1,544 held-out recorded transitions the model's predicted next state matches the reading after 98.6% of lock flips, 94.2% of table moves and 99.3% of unchanged slots, and 69.0% of moving pulls; a written 'pull goes to the end' scores 79.9%. Where a pull ends varies by up to 0.5 over the middle 80% of pulls in one direction, against a threshold of 0.18, and the pre-touch action does not say where ([card 170](../../experiments/170-effects-imagined-by-the-scene-model/card.md)).
