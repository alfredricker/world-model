---
id: result/174-four-variables-factor-on-fresh-episodes
type: result
title: drawer, window and both locks keep their own factor over three seeds on fresh episodes; held-out loss picks six coordinates
card: '174'
model: model/og5
env: environment/ogbench-visual-scene
measure: variables (of five) with a mean of at least 80% of their change in their own factor, three seeds, 100 fresh episodes
value: 4
against: 4
n: 3
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
Mean shares 96, 93, 86 and 83% for drawer, window and the two locks (cube 73%); each read through its own factor at 0.976-0.991, against 0.999-1.000 for card 168's rule-built slots and 0.53-0.68 for an untrained encoder. The held-out loss over sizes 4 to 16 (1.075, 0.931, 0.958, 0.934, 0.926 nats) picks 6 by the declared rule, within run-to-run noise. The lock factors move 10-14% of a flip while the hand is on the button ([card 174](../../experiments/174-do-the-five-factors-hold/card.md)).
