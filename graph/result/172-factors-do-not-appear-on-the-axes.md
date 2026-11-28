---
id: result/172-factors-do-not-appear-on-the-axes
type: result
title: variables do not line up on the coordinates of a contrastive state larger than the scene
card: '172'
model: model/og5
env: environment/ogbench-visual-scene
measure: share of the state's change lying in a variable's best coordinate when only that variable changes, percent, best of five variables
value: 19.4
against: 80.0
n: 6684
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/state-from-nearness-in-time
- rel: instance
  to: problem/object-discovery
  weight: 1.0
  weight_basis: counted
---
All 32 coordinates were in use and none moved together; a variable's change was 8-19% in its best coordinate against a bar of 80%, and 9-27% even with pairs chosen by the true state, so the gate's upper bound did not reach the bar and the main run was not made. Unmixing step changes into independent directions gave 13-46%. The learned temperature fell to 0.20 and undid the penalty meant to empty spare coordinates; single steps are mostly jitter (a lock's flip moves the state 1.4-1.6 times an idle step) ([card 172](../../experiments/172-factors-from-sparse-change-over-time/card.md)).
