---
id: result/155-imperfect-masks-blur-the-appearance-prior
type: result
title: appearance read from masks stored by binding blurs the shape feature, and a class with few objects loses its prior
card: '155'
model: model/og4
env: environment/sprite-world
measure: percent of never-met sticky objects for which the take action is right, mask at test only, world 1 by shape
value: 20.0
against: 90.0
n: 65
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/feedback-as-stored-couplings
- rel: tested
  to: principle/kinds-by-one-energy
---
From [card 155](../../experiments/155-object-modules-fed-by-binding/card.md): checked by rerunning the world. 13 sticky objects with transitions; the shape attractor of the squares (the sticky shape) holds 7 sticky and 3 takeable objects (mean pattern overlap 0.81), 6 sticky objects fall on other attractors, and the prior gives a square 0.66 for the sticky class. The other nine worlds are at 100.0.
