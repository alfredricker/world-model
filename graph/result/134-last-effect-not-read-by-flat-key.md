---
id: result/134-last-effect-not-read-by-flat-key
type: result
title: the last effect per object is in the memory but a flat key does not read it
card: '134'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock never touched but the slide pulled earlier, 400 episodes
value: 57.9
against: 88
n: 464
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/episode-history-in-the-recall-key
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
Criterion 1 fails as declared. With the last effect of each object added to the memory, the 464 pulls whose lock was never touched but whose slide was pulled earlier (27 do not move) score 57.9% balanced accuracy against a bar of 88, the true lock at 91.9% and a rule reading the one item ("the same as the last pull of this slide") at 90.3%. The content is present and a single-item rule uses it, but one of 33 equal-weight items is not read by nearest-neighbour recall. The weight is an estimate. Numbers are in [results.json](../../experiments/134-unseen-state-from-effects/results.json); see [card 134](../../experiments/134-unseen-state-from-effects/card.md).
