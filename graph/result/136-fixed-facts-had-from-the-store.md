---
id: result/136-fixed-facts-had-from-the-store
type: result
title: fixed facts are had from the store alone
card: '136'
model: model/og4
env: environment/memory-world
measure: log loss on cold touches of the default and timer children
value: 0.020
against: 0.020
n: 594
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/store-everything-read-by-similarity
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
This is from a breakdown run after the card, not a declared measure. On cold touches of the children whose parent has a fixed start value or a timer, recall reaches the floor of 0.020 log loss, the same as the best possible answer. The base rates are 0.315 and 0.124. What is always so is repeated in every stored episode, so reading by similarity finds it without any memory within the episode. The weight is an estimate. Numbers are in [results.json](../../experiments/136-memory-world/results.json); see [card 136](../../experiments/136-memory-world/card.md).
