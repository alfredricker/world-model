---
id: result/138-key-cannot-say-nothing-is-known
type: result
title: the corrected key still cannot say that nothing is known on cold touches
card: '138'
model: model/og4
env: environment/memory-world
measure: log loss, cold touches of the random child, mean of three worlds (weights arm)
value: 0.712
against: 0.694
n: 171
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/key-prior-corrected-by-evidence
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
On cold touches where nothing or little can be known before the episode, recall with the learned weights stays at or above the base rate in log loss (lower is better). Random child 0.712 against 0.694 for the base rate (0.722 for card 137's prior); chance 0.663 against 0.656; two 0.651 against 0.591; rewired 0.729 against 0.695. The weights improve a little on the prior on the random child (0.722 to 0.712) but do not reach the base rate, and the bar was 0.70. The declared network arm is similar (0.716). A likely reason, not tested: the key decides which stored cases are near, not how sure the answer should be when no stored case is like the present one. The reference key that knows the truth gets 0.669 on the random child, so a better answer exists. The weight is an estimate. Numbers are in [results.json](../../experiments/138-key-correction-on-the-memory-world/results.json); see [card 138](../../experiments/138-key-correction-on-the-memory-world/card.md).
