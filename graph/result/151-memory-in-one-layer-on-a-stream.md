---
id: result/151-memory-in-one-layer-on-a-stream
type: result
title: on a changing stream, memory in one layer helps; memory in every layer and asynchronous updating cost
card: '151'
model: model/og4
env: environment/world-of-parts
measure: percent of output tokens right over ticks 6 to 15 after a change of scene, noise 4.0, memory in the scene layer, mean of 5 worlds
value: 100.0
against: 97.1
n: 5
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/layers-of-attractor-modules
---
Criterion 3 of [card 151](../../experiments/energy-modules/151-layers-of-attractor-modules-in-a-world-of-parts/card.md): with a low-pass memory in the scene layer 100.0% against 92.1% for a single look (bar: 5.0 above), about four ticks behind a change; memory in the patch layer does the same. Memory in every layer adds the lags (82.6%). Updating a random half of the modules per tick gives 78.4%, where in card 150 it cost nothing: here the input changes every 16 ticks.
