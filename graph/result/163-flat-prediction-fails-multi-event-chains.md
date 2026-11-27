---
id: result/163-flat-prediction-fails-multi-event-chains
type: result
title: one flat learned prediction walks to near events from random play but fails multi-event chains even with demonstrations
card: '163'
model: model/og4
env: environment/minigrid
measure: UnlockPickup gate layouts solved of 40, experience 1 (stage D upper bound with 2,500 demonstrations)
value: 0.0
against: 30.0
n: 1
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/plan-backward-over-conditions
---
A gate result, one experience, not a test. Against: card 162's agent on the same 40 layouts (30 of 40). Flat q solves DoorKey-5x5 to 8x8 and Unlock (38–40 of 40) from random play, and handed the next event solves DoorKey-8x8 (40/40), DoorKey-16x16 (40/40) and KeyCorridor (39–40/40). Over multi-event chains the action gap shrinks to 0.011 at 16–19 steps out, under approximation error, and defining an event without its action conflates picking up a box with destroying it by toggle.
