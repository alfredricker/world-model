---
id: result/173-state-moves-when-nothing-differs
type: result
title: the lock factors move when nothing in the scene does
card: '173'
model: model/og5
env: environment/ogbench-visual-scene
measure: state's change over 5-step windows where nothing differs, as percent of its change where one variable does
value: 36.6
against: 20.0
n: 5741
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/perception-from-narrow-signal
  weight: 1.0
  weight_basis: counted
---
The gate's bar was 20% and the fit gave 36.6% (single steps 57.2%; card 172 74.9%). Two thirds of the movement is in the two lock factors, 0.08-0.09 a window against 0.85-0.92 for a real flip; it is as large when the cube and slides are still to the millimetre, and barely follows the arm's movement (correlation 0.11). Cause not found. It also drags the whole-state reading of the cube and drawer below card 172's (0.910, 0.866) ([card 173](../../experiments/173-tight-box-sparse-change-factors/card.md)).
