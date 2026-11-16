---
id: result/135-looking-by-memory-age-recovers-most
type: result
title: looking when a memory of that age is often wrong recovers most of the loss
card: '135'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), all pulls, a quarter of touches hidden, look when the age of the memory is too often wrong
value: 91.7
against: 95.5
n: 1634
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/look-again-when-memory-is-stale
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
The agent looks at the condition object before a touch when the share of stored re-sightings of a memory of that age that were wrong is over 0.2, and always looks if the object was never seen. On all 1,634 test pulls this scores 91.7 balanced accuracy with 0.38 looks per pull, against 80.7 for never looking, 83.8 for random looks at the same rate and 95.5 for always looking (1 look per pull); the bar was 91 with at most 0.45 looks. That is 74% of the way from never to always for 38% of the looks, and 7.9 points above random looks (bar 3). The log loss falls from 0.223 to 0.108 (random 0.194, always 0.065). Much of the gain is that every pull whose lock was never seen gets a look (96.6 against 58.3 for never). The weight is an estimate. Numbers are in [results.json](../../experiments/135-state-that-changes-out-of-sight/results.json); see [card 135](../../experiments/135-state-that-changes-out-of-sight/card.md).
