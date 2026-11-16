---
id: result/138-item-weights-win-back-the-cost-of-old-records
type: result
title: one learned weight per key item wins back the cost of old records and beats a key that knows the truth
card: '138'
model: model/og4
env: environment/memory-world
measure: share of the gap closed (%), warm touches of the fast-drift child, mean of three worlds
value: 82
against: 62
n: 1540
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/key-prior-corrected-by-evidence
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
This is the arm that was reported, not the declared arm; the declared arm is the network in [the next result](138-network-correction-unsteady-between-worlds.md). One learned weight per item of the key, trained only on how well stored touches recall one another, with records never emptied, lifts warm touches on every factor against the prior of card 137. Fast drift goes from 62% to 82% of the gap closed (mean of three worlds), the timer from 78% to 93% and the random child from 89% to 100%. It beats the reference key that knows which old values count (76%, 85% and 92% on those three factors), and it keeps cold touches of the carry child at 100%, against 99% for the prior. This answers [the card 137 finding](137-old-values-in-the-key-cost-warm-accuracy.md) that old values left in the key cost accuracy within the episode. The weight on the item "seen in this episode" already pulls fresh and stale records apart, so no combination of items is needed. On the declared bars this arm would pass criteria 1 and 3 and fail 2 (cold random child 0.712, bar 0.70), see [the cold result](138-key-cannot-say-nothing-is-known.md). The weight is an estimate. Numbers are in [results.json](../../experiments/138-key-correction-on-the-memory-world/results.json); see [card 138](../../experiments/138-key-correction-on-the-memory-world/card.md).
