---
id: result/153-one-run-of-twenty-fails-at-30-episodes
type: result
title: with 30 episodes one sprite world of five by shape leaves fixed and takeable objects in one kind
card: '153'
model: model/og4
env: environment/sprite-world
measure: percent of never-met objects for which the take action is right, world 3 by shape, 30 episodes
value: 55.0
against: 90.0
n: 1
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/kinds-by-one-energy
---
From [card 153](../../experiments/energy-modules/153-kinds-found-by-one-energy-in-two-domains/card.md): 19 of 20 runs at 30 and 100 episodes reach 90%. Checked by replaying the run: the stored seeds were two sticky and two takeable objects; no fixed object's own difference lowered the energy, because a proposed kind is credited with what it explains and not with what its removal lets the old kind explain. Card 152's learner fails a different world at 30 episodes (80.8).
