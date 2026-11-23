---
id: result/160-acting-fails-where-a-fact-enters-the-energy-twice
type: result
title: the sparse energy splits credit between a door's open code and the memory of opening it, and the agent then finds no plan; a perceptron in the same agent does better
card: '160'
model: model/og4
env: environment/minigrid
measure: percent of 100 new layouts solved, mean of five test tasks and of 3 experiences
value: 44.5
against: 72.5
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/events-as-latent-state-changes
---
Against: a multilayer perceptron on the same inputs in the same agent. UnlockPickup, BlockedUnlockPickup and KeyCorridorS3R3: 8.7, 9.7 and 5.3% (the perceptron 41, 45.7, 75.7; told the rules 100). The learned move effect lets the agent through an open door of five colours of six only when memory holds the act of opening it; the planner imagined the door open without it. With that one inconsistency removed after the test: 42, 40 and 8%. Also missing: "hands must be empty" for 5 to 8 of 24 objects, learned per code from 0 to 2 tries.
