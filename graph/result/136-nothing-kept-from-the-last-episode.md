---
id: result/136-nothing-kept-from-the-last-episode
type: result
title: recall keeps nothing from the last episode, so a parent that carries over is unread on first touches
card: '136'
model: model/og4
env: environment/memory-world
measure: log loss on cold touches of the carry child
value: 0.718
against: 0.020
n: 276
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/episode-history-in-the-recall-key
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
This is from a breakdown run after the card, not a declared measure. On cold touches (neither the child nor its parents seen yet in the episode) of the child whose parent carries its state over between episodes, recall's log loss is 0.718. The best possible answer scores 0.020 and the base rate 0.700, so recall is slightly worse than knowing nothing. The record holds nothing from one episode to the next, as in the model now, so the state left at the end of the last episode is not available at the start of the next. The weight is an estimate. Numbers are in [results.json](../../experiments/136-memory-world/results.json); see [card 136](../../experiments/136-memory-world/card.md).
