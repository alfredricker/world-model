---
id: result/137-old-values-in-the-key-cost-warm-accuracy
type: result
title: old values left in the key cost accuracy within the episode on the drifting and timer children
card: '137'
model: model/og4
env: environment/memory-world
measure: share of the gap closed (%), warm touches of the fast-drift child, mean of three worlds
value: 62
against: 73
n: 1540
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/episode-history-in-the-recall-key
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
With old records kept, warm touches lose accuracy where the state changes by itself: fast drift goes from 73% to 62% of the gap closed (11 points) and the timer from 85% to 78% (7 points), against an allowed loss of 5. Slow drift is unchanged (82 to 83%), and the other children lose 3 to 5 points (random 92 to 89, rewired 64 to 59). Values from earlier episodes sit in the key beside the current ones and recall cannot tell which to trust. The weight is an estimate. Numbers are in [results.json](../../experiments/137-records-that-outlast-the-episode/results.json); see [card 137](../../experiments/137-records-that-outlast-the-episode/card.md).
