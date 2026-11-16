---
id: result/139-more-neighbours-do-not-fix-nothing-known
type: result
title: reading 200 nearest stored touches instead of 25 does not fix the error where nothing is known
card: '139'
model: model/og4
env: environment/memory-world
measure: log loss, cold touches of the random child, mean of three worlds (200 neighbours)
value: 0.733
against: 0.712
n: 171
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
This is the feasibility gate of [card 139](../../experiments/139-an-answer-that-knows-when-not-to-trust-itself/card.md), which tested whether the cold error of [card 138](138-key-cannot-say-nothing-is-known.md) is the noise of reading only 25 stored touches. With the 200 nearest stored touches, cold touches of the random child do not improve: log loss 0.733 against 0.712 with 25, and 0.694 for the base rate (lower is better). Warm touches collapse, from 100% to 67% of the gap closed. The card concludes that the error is a bias and not the noise of 25 samples. The weight is an estimate. Numbers are in [results.json](../../experiments/139-an-answer-that-knows-when-not-to-trust-itself/results.json); see [card 139](../../experiments/139-an-answer-that-knows-when-not-to-trust-itself/card.md).
