---
id: result/154-objects-that-never-moved-are-never-bound
type: result
title: an object that has never moved has no stored pattern and is not bound by feedback
card: '154'
model: model/og4
env: environment/sprite-world
measure: percent found at first sight of pool objects that moved in fewer than 3 steps of training, 19 of 20 worlds
value: 0.0
against: 90.0
n: 2365
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/feedback-as-stored-couplings
---
From [card 154](../../experiments/154-feedback-from-object-modules-to-binding/card.md): 10 to 24 of the 40 pool objects per world, the fixed ones among them; in the worlds by shape the fixed shapes are never seen moving at all. Feedback from stored patterns cannot reach them; they need binding couplings learned as a function of appearance, or another signal from the agent's action.
