---
id: result/167-partial-observability-belief-and-frontier-exploration
type: result
title: True 7x7 occluded partial-view agent with persistent belief and epistemic frontier BFS solves 93.1% of fresh MiniGrid layouts zero-shot (-4.1 pts vs full view)
card: '167'
model: model/og4
env: environment/minigrid
measure: percent of 100 fresh layouts solved across 5 tasks under 7x7 occluded egocentric partial observability, mean of 3 reward-free random-play experiences
value: 93.1
against: 25.5
n: 3
outcome: pass
lost_points: 4.1
links:
- rel: tested
  to: principle/belief-keeps-last-value
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: tested
  to: principle/conditions-from-own-failed-try
---
Against: no-frontier partial-view baseline (25.5% mean: DK8 75.3%, DK16 24.0%, KeyCorridor 8.0%, UnlockPickup 20.0%, BlockedUnlockPickup 0.0%) and full-observability Card 166 upper bound (97.2% mean: 99.0, 98.4, 94.2) on the exact same 100 fresh layouts per task (seeds 11000-11099, 1,500 episodes per agent across 3 experiences). Card 167 achieves 93.1% mean (95.2, 94.4, 89.6; +67.6 pts over baseline, only -4.13 pts retention gap vs omniscient full view) with zero extra training of WNet or q_reach: DoorKey-8x8 100% (20.8 steps), DoorKey-16x16 100% (95.7 steps), KeyCorridorS3R3 86.0%, UnlockPickup 86.0%, BlockedUnlockPickup 93.3% (88.4% across the 3 100%-occluded-goal tasks vs baseline's 9.3%). Zero observation leaks verified across 50 episodes (4,192 steps); 581 episodes solved after the goal left the 7x7 cone; mean compute 5.64 ms/step (3.11x faster than full-view Card 164).
