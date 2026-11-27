---
id: result/164-zero-dijkstra-event-search-fails-gate
type: result
title: event-level search using one discounted single-event q as a reachability gate solves DoorKey-8x8 (40/40) but fails multi-room pickup tasks
card: '164'
model: model/og4
env: environment/minigrid
measure: percent of 40 gate layouts solved on UnlockPickup, experience 1 (Arm B, zero Dijkstra)
value: 25.0
against: 75.0
n: 1
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/plan-backward-over-conditions
---
A gate result, one experience, not a test. Against: card 162 on the same 40 gate layouts. Arm B solves DoorKey-8x8 at 40/40 in 16.2 steps (7.3 abstract expansions, zero Dijkstra, zero move model) where reachable (0.86/0.60) and blocked (0.005) separate by 100x, but fails UnlockPickup (25%), BlockedUnlockPickup (17.5%) and KeyCorridor (10%) because one threshold on discounted V conflates walking distance with binary reachability on unseen multi-room layouts.
