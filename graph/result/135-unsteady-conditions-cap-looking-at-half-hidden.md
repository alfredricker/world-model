---
id: result/135-unsteady-conditions-cap-looking-at-half-hidden
type: result
title: with half of the touches hidden the choice by contrast finds no lock for the window, so the agent never looks before a window pull
card: '135'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), all pulls, half of touches hidden, look by memory age (0.29 looks per pull); no bar
value: 83.7
against: 84.5
n: 1073
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/conditions-by-contrast-over-touches
- rel: instance
  to: problem/recall-key
  weight: 1.0
  weight_basis: estimated
---
With half of the touches hidden (1,073 pulls; no bar was set) the choice by contrast found the drawer's lock but no lock for the window, so the agent never looked before a window pull. The age rule scores 83.7 with 0.29 looks per pull, against 72.6 for never looking, 84.5 for always looking at the objects it knows of (0.51 looks) and 94.5 for the true lock. This is card 134.1's limit: conditions need enough clean stored cases, and with half hidden there are 3,741 stored events against 5,592 at a quarter. It caps what any looking rule can do here. The weight is an estimate. Numbers are in [results.json](../../experiments/135-state-that-changes-out-of-sight/results.json); see [card 135](../../experiments/135-state-that-changes-out-of-sight/card.md).
