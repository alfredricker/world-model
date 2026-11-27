---
id: result/166-unified-factored-relational-world-model
type: result
title: Unified shared-representation WNet beats Card 165's two-model System 2 across all 3 experiences (97.1% vs 95.7%, KeyCorridor 98.3% vs 92.0%)
card: '166'
model: model/og4
env: environment/minigrid
measure: percent of 100 fresh layouts solved across 5 tasks, mean of 3 reward-free random-play experiences
value: 97.1
against: 95.7
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/a-token-is-what-it-is-and-where-it-lies
- rel: tested
  to: principle/one-shared-output-for-no-change
- rel: tested
  to: principle/conditions-from-own-failed-try
---
Against: frozen Card 165 (separate RNet + Card 160 Energy MLP / Table / gid / statement()) on the exact same 100 fresh layouts per task (seeds 10000-10099, 3 experiences: Card 166 98.2, 97.8, 95.2 = 97.1%; Card 165 97.8, 95.4, 94.0 = 95.7%; +1.34 pts, +20 net layout wins). KeyCorridorS3R3 jumps +6.3 pts (92.0% -> 98.3%, Exp 2 86% -> 98%, Exp 3 94% -> 100%), DoorKey-8x8 100%, DoorKey-16x16 100%, BlockedUnlockPickup 95.3%, UnlockPickup 91.7% vs Card 165's 91.3% (missed only the fixed pre-registered 94.0% UnlockPickup sub-bar due to a harder Exp 3 draw for frozen System 1 q_reach_3.pt, hence outcome: fail 2 of 3, Decision: Keep). WNet achieves 100.00% r_arr/r_act across 223,296 fresh queries (0 fp, 0 fn) and 99.98% live + counterfactual 1-step cell-effect accuracy across 44,406 fresh queries (99.95% change recall, 99.98% specificity) at 7.33 ms/step (1.15x Card 165, 2.39x faster than Card 164).
