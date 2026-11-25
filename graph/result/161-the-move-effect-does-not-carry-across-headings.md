---
id: result/161-the-move-effect-does-not-carry-across-headings
type: result
title: the learned effect of the agent's own move does not carry from one heading to another, and KeyCorridor fails where a door must be passed heading west
card: '161'
model: model/og4
env: environment/minigrid
measure: KeyCorridorS3R3 layouts solved of 40 outside every test, experience 3; against the same agent with MiniGrid's rule for the agent's move only
value: 28
against: 38
n: 2
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/acting-by-chaining-learned-statements
---
A diagnostic after card 161's test. Each heading has its own weights and each neighbour enters as one input per (position, code), so an open door of a colour never walked through heading west is not passable in imagination (43 of 233 door approaches heading west in experience 1, 103 of 233 in experience 3; none in the other headings). Experience 1: 36 against 40. The role, the cell ahead, is not separated from what fills it (GOAL P3).
