---
id: result/150-three-layers-on-sprite-pixels
type: result
title: three feed-forward layers of energy modules read each object's motion from pixels, however they are clocked
card: '150'
model: model/og4
env: environment/sprite-world
measure: percent of moving steps for which the object module's displacement is right, 40 updates per frame, 200 test episodes
value: 99.5
against: 97.0
n: 200
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/layers-of-attractor-modules
- rel: tested
  to: principle/binding-from-interaction
---
[Card 150](../../experiments/150-three-feed-forward-layers-of-energy-modules/card.md): motion modules per pixel, binding modules per point and object modules give the right displacement in 99.5% of moving steps (true masks 99.8%, one module for the whole frame 0.0%); 5 updates per frame 99.5%, one update 98.8%, a random half of the points per tick 99.5%. The three ways of reading the layer below (state, state gated by scalar energy, whole belief) agree to 0.1 points: this world's noise is too low to tell them apart.
