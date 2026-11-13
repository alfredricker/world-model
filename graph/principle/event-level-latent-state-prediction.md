---
id: principle/event-level-latent-state-prediction
type: principle
title: Event-level latent state prediction
status: tested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
---
A network predicts the latent state (patch features) at the next touch from the current patches and the touch, optionally carrying a recurrent memory state. Tested in the uncarded gate runs under working number 132 (no card is written yet). It learns ordinary consequences but not what is hidden in the drawer.
