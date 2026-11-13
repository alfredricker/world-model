---
id: result/130-drawer-never-opened
type: result
title: drawer never opened
card: '130'
model: model/og4
env: environment/ogbench-visual-scene
measure: task 3 episodes that never opened the drawer in 750 steps
value: 0.08
against: null
n: 50
outcome: fail
lost_points: 8
links:
- rel: tested
  to: principle/own-tries-first
- rel: measures
  to: ability/scene-task3
- rel: instance
  to: problem/open-loop-skills
  weight: 1.0
  weight_basis: counted
---
4 episodes of task 3 hit the 750-step limit without opening the drawer ([card 130](../../experiments/130-own-tries-first/card.md)); a replay does not react to the drawer while it runs.
