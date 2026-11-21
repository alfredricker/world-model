---
id: result/154-no-harm-criterion-failed-on-a-scoring-cut-off
type: result
title: one world of twenty breaks the bar on recognising never-stored things, by a correct recognition counted as wrong
card: '154'
model: model/og4
env: environment/sprite-world
measure: percent of things counted as never stored that are recognised as a stored pattern, world 4 by shape
value: 5.4
against: 2.0
n: 129
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/feedback-as-stored-couplings
---
From [card 154](../../experiments/154-feedback-from-object-modules-to-binding/card.md): checked by rerunning the world. All 7 cases are one pool object that moved in 2 steps by the simulator's record (the scoring cut-off for known was 3) while the binding layer counted 3 and stored a pattern of it with mask overlap 1.0; it was recognised as itself. Recognition by a pattern stored from another thing: 0 of 129 at first sight, 1 of 1,097 in covered scenes. The other 19 worlds are at most 1.6%; no label lies on the ground in any world.
