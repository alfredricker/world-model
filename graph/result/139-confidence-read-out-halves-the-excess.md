---
id: result/139-confidence-read-out-halves-the-excess
type: result
title: a confidence read-out fitted on recall's own errors moves cold log loss toward the base rate but misses three of five bars
card: '139'
model: model/og4
env: environment/memory-world
measure: log loss, cold touches of the random child, mean of three worlds (read-out arm)
value: 0.704
against: 0.694
n: 171
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/recall-confidence-learned-from-own-errors
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
A read-out fitted on recall's own held-out errors, from its answer and how fresh the records in the key are, moves cold log loss toward the base rate on every child (lower is better). Random child 0.704 with the read-out, 0.712 under [card 138's answer](138-key-cannot-say-nothing-is-known.md) and 0.694 for the base rate. Two 0.620, 0.651 and 0.591. Drift slow 0.709, 0.721 and 0.695. Three of the five bars are missed, each by 0.005 or 0.01 or more above the base rate: random (+0.010), drift slow (+0.014) and two (+0.029); drift fast (+0.003) and chance (+0.003) are within their bars. The read-out also costs warm fast drift 1.4 points of gap closed (81.0% against 82.4% under card 138's answer, 1 point allowed). The card calls the removal of about half of the excess on every child too little, and decides to stop. The weight is an estimate. Numbers are in [results.json](../../experiments/139-an-answer-that-knows-when-not-to-trust-itself/results.json); see [card 139](../../experiments/139-an-answer-that-knows-when-not-to-trust-itself/card.md).
