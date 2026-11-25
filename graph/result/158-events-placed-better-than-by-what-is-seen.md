---
id: result/158-events-placed-better-than-by-what-is-seen
type: result
title: a latent state inferred from own tokens places taken and released events better than the state read from visible motion
card: '158'
model: model/og4
env: environment/sprite-world
measure: F1 of the model's events against the simulator's taken and released steps, percent, mean of 10 worlds
value: 71.3
against: 53.5
n: 10
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/events-as-latent-state-changes
---
From [card 158](../../experiments/energy-modules/158-events-as-changes-of-an-objects-latent-state/card.md): above the baseline in 10 worlds of 10; 99.7 with the simulator's tokens, 70.0 with the true behaviours given, so the gap is the tokens. Releases without a move are mostly missed; where an event shows only later its place is decided by the sparsity prior.
