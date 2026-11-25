---
id: result/147-binding-from-interaction-finds-moved-objects
type: result
title: from motion under the agent's interaction alone, moved objects are found, held and released
card: '147'
model: model/og4
env: environment/sprite-world
measure: percent of frames in which a moved thing is found (IoU of at least 0.5), motion read from pixels, object files with accumulated prediction error
value: 98.4
against: 93.0
n: 200
outcome: pass
lost_points: 0
links:
- rel: tested
  to: principle/binding-from-interaction
- rel: instance
  to: problem/object-discovery
  weight: 0.5
  weight_basis: estimated
---
Criteria 1 to 3 of [card 147](../../experiments/energy-modules/147-binding-from-interaction-in-a-sprite-world/card.md), 200 test episodes: 98.4% found (bar 93.0; 98.9 with the simulator's true motion; 39.9 for grouping by the current step's motion with no memory). Held objects share a group with the hand in 2.3% of frames and released objects are read as background in 1.8% (bars 5.0); objects 3 or 4 pixels wide are found in 98.9% (bar 85.0). An object is bound in the first frame after it first moves in 99.5%. The world grants rigid flat motion, texture and a fixed camera. The weight is an estimate.
