---
id: result/134-neighbour-share-is-an-honest-probability
type: result
title: the share of 25 nearest events is an honest probability where nothing is known
card: '134'
model: model/og4
env: environment/ogbench-visual-scene
measure: log loss, pulls with nothing known, old contents (card 133.1), 400 episodes
value: 0.492
against: 0.538
n: 228
outcome: pass
lost_points: 0
links:
- rel: instance
  to: problem/prior-over-unseen-state
  weight: 1.0
  weight_basis: estimated
---
This holds for the old contents, not for this card's change. On the 228 pulls with nothing known, the share of the 25 nearest stored events in which the slide moved gives a log loss of 0.492 against 0.538 for always saying the base rate, and over all pulls the stated probabilities lie within 1.7 points of the observed shares (bar 5). With the new contents the log loss is 0.580 against a bar of 0.56, so the card's own criterion fails on that half. The weight is an estimate. Numbers are in [results.json](../../experiments/134-unseen-state-from-effects/results.json); see [card 134](../../experiments/134-unseen-state-from-effects/card.md).
