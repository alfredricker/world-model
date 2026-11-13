---
id: result/134-learned-correction-misses-rare-cases
type: result
title: the learned correction does not find the deciding pulls over the new contents
card: '134'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), pulls with the lock never touched but the slide pulled earlier, card 133.1 network correction, 3 seeds
value: 55.0
against: 91.9
n: 464
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
Post hoc (a check run after the result, not a declared arm): card 133.1's learned network correction over the new contents scores 55.0 in the middle group (mean of 3 seeds, 47.5 to 61.6) against 91.9 for the true lock. The pulls that decide are about 35 of 7,400 stored events, and carry almost no weight in a mean error. The weight is an estimate. Numbers are in [results.json](../../experiments/134-unseen-state-from-effects/results.json); see [card 134](../../experiments/134-unseen-state-from-effects/card.md).
