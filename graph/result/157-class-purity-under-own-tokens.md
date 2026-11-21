---
id: result/157-class-purity-under-own-tokens
type: result
title: dynamics classes fitted on own tokens are less pure and more numerous than on the simulator's
card: '157'
model: model/og4
env: environment/sprite-world
measure: purity of the dynamics classes against the true behaviours, own tokens, percent, mean of 10 worlds
value: 88.9
against: 90.0
n: 10
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/transitions-from-own-states
---
From [card 157](../../experiments/157-transitions-read-from-the-binding-layers-own-states/card.md): 97.8 with the simulator's tokens; 4 to 6 classes against 3 or 4. The own contact token catches 69.4% of the simulator's contacts (the box around a pattern's points often reaches outside the object). Leaving out the patterns on the ground does not cure it (one world 75.1 to 86.0, another 61.1 to 64.8).
