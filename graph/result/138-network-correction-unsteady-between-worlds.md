---
id: result/138-network-correction-unsteady-between-worlds
type: result
title: the network correction to the key is unsteady between worlds and falls under the prior on fast drift
card: '138'
model: model/og4
env: environment/memory-world
measure: share of the gap closed (%), warm touches of the fast-drift child, mean of three worlds
value: 56
against: 68
n: 1540
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/key-prior-corrected-by-evidence
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
The declared arm, weights plus a small network over the key's items, gets warm fast drift to 56% of the gap closed (mean of three worlds), against a bar of 68% and 62% for the prior of card 137. In the three worlds it is 46%, 36% and 88%, so the mean hides a large spread. Chance is also 3 points under the prior (70 against 73), where 2 are allowed. The timer reaches 94% (bar 80), so the network can help; it is not steady. On stored episodes held out from training, its error is above that of the plain weights at every cost tried, and still falling at the largest cost tried (100), so the range of costs stopped too early and this failure is not conclusive for the network. As run, it fits the stored touches of a world more closely than they support. For now [the weights arm](138-item-weights-win-back-the-cost-of-old-records.md) is the one kept. The weight is an estimate. Numbers are in [results.json](../../experiments/138-key-correction-on-the-memory-world/results.json); see [card 138](../../experiments/138-key-correction-on-the-memory-world/card.md).
