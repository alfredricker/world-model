---
id: result/165-zero-dijkstra-system-two-with-neural-reachability
type: result
title: Zero-Dijkstra System 2 with learned neural reachability propagation (RNet) solves 97.1% of fresh MiniGrid layouts at 2.71x the speed of Card 164
card: '165'
model: model/og4
env: environment/minigrid
measure: percent of 100 fresh layouts solved across 5 tasks, mean of 3 reward-free random-play experiences
value: 97.1
against: 98.1
n: 3
outcome: pass
lost_points: 1.1
links:
- rel: tested
  to: principle/a-token-is-what-it-is-and-where-it-lies
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: tested
  to: principle/conditions-from-own-failed-try
---
Against: Card 164 Arm A (Dijkstra WxHx4 pose graph + Card 162 move model + hand-written want_in/want_out/free_reach/forward_ok) on the exact same 100 fresh layouts per task (seeds 9000–9099, 3 experiences: Card 165 98.4, 96.4, 96.4 = 97.1%; Card 164 Arm A 98.4, 97.0, 99.0 = 98.1%; -1.06 pts, well within the 5-pt general-form margin). RNet (~150 params, 15.5s/exp on pure Stage R random play, zero headings or cardinal priors) recovers 39/39 passable codes and exact 4 cardinal offsets in all 3 experiences and scores 100.00% on both r_arr and r_act across 224,448 fresh queries (0 fp, 0 fn, seeds 9000–9019). ZeroDijkstraAgent solves DoorKey-8x8 100%, DoorKey-16x16 100%, KeyCorridorS3R3 92.0%, UnlockPickup 96.3%, BlockedUnlockPickup 97.0% with 6 destroyed goal tokens (vs 8 for Card 164), running 2.71x faster overall (6.48 vs 17.55 ms/step; 4.53x faster on KeyCorridorS3R3).
