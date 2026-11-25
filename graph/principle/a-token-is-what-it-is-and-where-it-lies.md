---
id: principle/a-token-is-what-it-is-and-where-it-lies
type: principle
title: In the effect network every token is two parts, what it is and where it lies relative to the token asked about, combined by attention
status: tested
links:
- rel: addresses
  to: problem/relations-between-variables
  basis: claimed
---
What a token is has one embedding, used wherever the token appears. Where it lies (an offset, "far", or "out of view by action a") has another. Attention's keys come from where a token lies, with the token asked about and the action as the query; what the token is, plus the bare fact that something lies there, is what it contributes. No rotation is built in: which offset matters for which heading is learned. This separates a role (the cell ahead) from what fills it (GOAL P3). Tested for the agent's own move in [card 162](../../experiments/162-a-neighbour-is-what-it-is-and-where-it-lies/card.md): kept. One network for the agent and every cell alike failed that card's gate and is open.
