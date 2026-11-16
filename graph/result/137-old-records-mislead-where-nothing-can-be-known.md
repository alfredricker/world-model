---
id: result/137-old-records-mislead-where-nothing-can-be-known
type: result
title: old records mislead recall where nothing about the unseen state can be known
card: '137'
model: model/og4
env: environment/memory-world
measure: log loss, cold touches of the random child, mean of three worlds
value: 0.722
against: 0.683
n: 171
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/episode-history-in-the-recall-key
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
Cold touches of the random child, where nothing can be known before the episode: log loss is 0.722 with old records kept, against 0.683 before and 0.694 for the base rate, with at most 0.02 more allowed. The slow-drift child is 0.722 against 0.727 before and the fast-drift child 0.708 against 0.711, so those hold. Recall does not fall back to the base rate when the stored cases do not apply, and it is now worse than knowing nothing. The rewired child, cold, goes from -20% to -40% of the gap closed, since the old records carry the old wiring. The weight is an estimate. Numbers are in [results.json](../../experiments/137-records-that-outlast-the-episode/results.json); see [card 137](../../experiments/137-records-that-outlast-the-episode/card.md).
