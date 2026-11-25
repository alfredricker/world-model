---
id: result/147-couplings-and-object-files-agree
type: result
title: evidence kept between pairs of points and evidence kept per object file bind equally well; they differ in cost and in what a misread motion does
card: '147'
model: model/og4
env: environment/sprite-world
measure: percent of frames in which a moved thing is found, pairwise couplings read by settling, motion read from pixels
value: 97.9
against: 98.4
n: 200
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/binding-from-interaction
- rel: tested
  to: principle/grouping-as-settling
- rel: instance
  to: problem/object-discovery
  weight: 0.3
  weight_basis: estimated
---
[Card 147](../../experiments/energy-modules/147-binding-from-interaction-in-a-sprite-world/card.md): couplings accumulated from co-motion and read by settling find 97.9% against 98.4% for object files, inside the card's 1.5-point margin, and both are unchanged with a ceiling of 16 objects. The couplings keep 37.7 million numbers per episode and take 78 s for 200 episodes; the files keep 49 thousand and take 13 s. Before the motion front end was tightened (30 validation episodes) the couplings found 96.4% and the files 90.1%: a few background pixels misread as moving alike open a file, the files run out, and nothing closes one. The weight is an estimate.
