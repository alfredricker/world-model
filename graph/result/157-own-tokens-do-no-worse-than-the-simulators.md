---
id: result/157-own-tokens-do-no-worse-than-the-simulators
type: result
title: transitions read from the binding layer do no worse on the mean than the simulator's tokens on the same experience
card: '157'
model: model/og4
env: environment/sprite-world
measure: percent of never-met objects for which the action that brings the object along is right, own tokens, mean of 10 worlds
value: 86.4
against: 85.0
n: 10
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/transitions-from-own-states
---
From [card 157](../../experiments/energy-modules/157-transitions-read-from-the-binding-layers-own-states/card.md): bound 100.0 (true classes given, own tokens), one class for all 36.7. Per world the difference from the simulator's tokens runs from -29.8 to +19.7 points; own tokens are better on fixed objects (98.6 against 72.1) and worse on takeable ones (80.2 against 98.8). One run of the agent per world.
