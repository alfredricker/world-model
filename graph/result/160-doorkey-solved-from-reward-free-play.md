---
id: result/160-doorkey-solved-from-reward-free-play
type: result
title: from reward-free random play the chaining agent solves DoorKey in sizes it never experienced, where PPO with the reward does not
card: '160'
model: model/og4
env: environment/minigrid
measure: percent of 100 new layouts solved, mean of DoorKey-8x8 and DoorKey-16x16 and of 3 experiences
value: 99.3
against: 3.2
n: 3
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/acting-by-chaining-learned-statements
---
700,000 steps of uniform random actions with no reward in seven small worlds, then 20 example success steps as the goal. Against: PPO trained with the reward on the task itself for the same number of steps (better of two views, 3 seeds: 6.0% and 0.3%). With half the experience or less the energy model's agent solves 5 to 16%.
