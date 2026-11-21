---
id: result/155-bound-objects-behave-as-with-masks
type: result
title: an object the binding has found is given its dynamics class and handled as with the simulator's masks
card: '155'
model: model/og4
env: environment/sprite-world
measure: percent of known objects bound at first sight for which the take action is right, lowest of 10 worlds
value: 100.0
against: 95.0
n: 10
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/feedback-as-stored-couplings
- rel: tested
  to: principle/kinds-by-one-energy
---
From [card 155](../../experiments/155-object-modules-fed-by-binding/card.md): learning reads no mask. Effects are at most 0.4 points under the bound on the same transitions; two dynamics classes (takeable, sticky) in every world, purity 98.4 to 100%. Over all known objects the take action is right for 94.5% against 98.8 with masks; the loss is the 8 to 10% of known movable objects that never moved in training.
