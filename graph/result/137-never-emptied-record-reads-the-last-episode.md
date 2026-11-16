---
id: result/137-never-emptied-record-reads-the-last-episode
type: result
title: a record never emptied between episodes lets recall read the state the last episode left
card: '137'
model: model/og4
env: environment/memory-world
measure: share of the gap closed (%), cold touches of the carry child, mean of three worlds
value: 99
against: 70
n: 530
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/episode-history-in-the-recall-key
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
With the record kept across episodes instead of emptied at each start, cold touches of the carry child go from -4% to 99% of the gap closed (mean of three worlds), against a bar of 70%. Log loss is 0.030, against 0.020 for the best possible answer, 0.718 before and 0.694 for the base rate. No rule says which objects carry over: the same per-object key reads the last episode wherever it is relevant. This answers [the card 136 finding](136-nothing-kept-from-the-last-episode.md) that recall kept nothing from the last episode. The weight is an estimate. Numbers are in [results.json](../../experiments/137-records-that-outlast-the-episode/results.json); see [card 137](../../experiments/137-records-that-outlast-the-episode/card.md).
