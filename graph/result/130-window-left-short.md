---
id: result/130-window-left-short
type: result
title: window left short
card: '130'
model: model/og4
env: environment/ogbench-visual-scene
measure: failures in tasks 2 and 3 that ended with the window 4 to 10 cm open
value: 0.75
against: null
n: 16
outcome: fail
lost_points: 24
links:
- rel: tested
  to: principle/own-tries-first
- rel: measures
  to: ability/scene-task2
- rel: measures
  to: ability/scene-task3
- rel: instance
  to: problem/reading-resolution
  weight: 0.5
  weight_basis: estimated
- rel: instance
  to: problem/open-loop-skills
  weight: 0.3
  weight_basis: estimated
- rel: instance
  to: problem/success-signal-from-own-reading
  weight: 0.2
  weight_basis: estimated
---
12 of 16 failures in tasks 2 and 3 ended with the window 4 to 10 cm open and read as closed; 12 of 100 episodes is 24 points ([card 130](../../experiments/130-own-tries-first/card.md)). The reading poisons the success counts too. Weights are estimates.
