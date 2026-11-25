---
id: principle/binding-from-interaction
type: principle
title: Points are bound into objects by how they move while the agent acts, with the evidence accumulated over time
status: tested
links:
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
The user's direction of 2026-11-19: the signal for what is one object comes from the agent's interaction (tracked motion, effects, events), not from how patches look. Points followed from frame to frame each report the displacement they made; one step of moving apart outweighs several of moving together, and the evidence is kept over time, either between pairs of points (couplings read by [settling](grouping-as-settling.md)) or between each point and each of at most K object files (the file that has predicted the point's motion best owns it). First tested in [card 147](../../experiments/energy-modules/147-binding-from-interaction-in-a-sprite-world/card.md), where both forms pass in a world built so that motion says everything. Not yet tested with plain surfaces, motion in depth, a moving camera or jointed objects; effects other than motion are not yet used.
