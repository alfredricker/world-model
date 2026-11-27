---
id: result/164-system-two-and-system-one-solve-minigrid
type: result
title: System 2 condition selection with held-event surprise recovery paired with fixed single-event System 1 q solves 98.3% of fresh MiniGrid layouts from pure random play
card: '164'
model: model/og4
env: environment/minigrid
measure: percent of 100 fresh layouts solved across 5 tasks, mean of 3 reward-free random-play experiences
value: 98.3
against: 93.2
n: 3
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: tested
  to: principle/conditions-from-own-failed-try
---
Against: card 162's agent on the exact same 100 fresh layouts per task (seeds 8000–8099, 3 experiences: card 162 91.4, 93.8, 94.4; Arm A 99.2, 98.4, 97.4). UnlockPickup rises from 85.3% to 98.7%, BlockedUnlockPickup from 85.7% to 98.7%, destroyed boxes drop from 87 to 8, and closest-action agreement of System 1 q is 99.6% on DoorKey-8x8, 91.2% on UnlockPickup and 82.2% on BlockedUnlockPickup with zero Dijkstra walking or random-action fallback. Pure System 1 swap without surprise recovery (Arm A0): 88.5% mean (-4.7 pts).
