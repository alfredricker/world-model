---
id: result/169-gives-up-at-the-end-of-long-chains
type: result
title: the agent gives up near the end of long chains
card: '169'
model: model/og4
env: environment/ogbench-visual-scene
measure: share of failed episodes that end with no chain found, of 71 failures in 750 episodes
value: 0.56
against: null
n: 71
outcome: fail
lost_points: 5
links:
- rel: tested
  to: principle/plan-backward-over-conditions
- rel: instance
  to: problem/success-signal-from-own-reading
  weight: 1.0
  weight_basis: counted
---
40 of 71 failures are the agent stopping because the search finds no chain from the state it reads: 26 of task 5's 33 failures, mostly within the last two presses, and 13 of task 4's 19. A pull repeated to the step limit accounts for 19 more ([card 169](../../experiments/169-zero-heuristic-event-bfs-on-ogbench/card.md)).
