---
id: result/135-age-is-a-weak-sign-among-seen-locks
type: result
title: among locks already seen, the age of a memory is a weak sign that it is wrong
card: '135'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock touched earlier, a quarter of touches hidden, look by memory age (0.25 looks per pull)
value: 89.9
against: 91
n: 1032
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/look-again-when-memory-is-stale
- rel: instance
  to: problem/state-across-time
  weight: 1.0
  weight_basis: estimated
---
In the 1,032 pulls with the lock touched earlier the age rule scores 89.9 balanced accuracy at 0.25 looks per pull, against a bar of 91 at no more than 0.35 looks (never 85.2, always 95.9). An oracle that looks only when the memory is wrong reaches 95.3 with 0.07 looks, so age picks out the wrong memories poorly. A steadier estimate from the 100 nearest re-sightings instead of 25 (7% wrong at age 1 rising to 30% at age 12; checked after the run, not a declared arm) traces the same curve, 90.2 at 0.27 looks, so the limit is what age can tell and not how it is estimated. The weight is an estimate. Numbers are in [results.json](../../experiments/135-state-that-changes-out-of-sight/results.json); see [card 135](../../experiments/135-state-that-changes-out-of-sight/card.md).
