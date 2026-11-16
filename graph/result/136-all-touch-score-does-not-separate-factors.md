---
id: result/136-all-touch-score-does-not-separate-factors
type: result
title: scoring every touch does not separate the factors, best and worst differing by 36 points
card: '136'
model: model/og4
env: environment/memory-world
measure: points between best and worst factor in share of gap closed
value: 36
against: 40
n: 9
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/object-codes-with-per-object-memory
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
The best factor closes 100% of the gap from base rate to the best possible answer and the worst closes 64%, a spread of 36 points against a bar of 40. The cause is that a child is touched about 12 times in an episode, and its own earlier outcomes reveal its parent, so most touches are easy whatever kind of memory the factor asks for. The measure needs the touches where the child's own history is empty (see the cold-touch breakdown in [136-nothing-kept-from-the-last-episode](136-nothing-kept-from-the-last-episode.md)). The weight is an estimate. Numbers are in [results.json](../../experiments/136-memory-world/results.json); see [card 136](../../experiments/136-memory-world/card.md).
