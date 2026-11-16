---
id: result/141-labels-are-not-the-cure-for-a-narrow-reading
type: result
title: a probe trained on the true position fails outside its training range just as the agent's read-out does
card: '141'
model: model/og4
env: environment/ogbench-visual-scene
measure: median cube error in the withheld third of the table, aim read-out (cm)
value: 2.18
against: 0.64
n: 1650
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
With a third of the table withheld from training, the aim read-out's median error is 2.18 cm in the withheld part against 0.64 cm in the kept part, 3.4 times larger (the card's bar was 5 times). A probe trained on the true position fails the same way, 1.91 against 0.19 cm, and its slide error grows 6 to 10 times in a withheld range of openings. A position read by a trained map from patches is therefore right only where the map was trained, whatever signal trains it ([card 141](../../experiments/141-representation-test-bed/card.md)). The n is one fit per arm.
