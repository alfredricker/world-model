---
id: principle/deterministic-memory-path-in-latent-state
type: principle
title: Deterministic memory path in the latent state
status: untested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
---
A latent state model keeps a deterministic recurrent part beside a stochastic part, so information can pass unchanged across many steps while the stochastic part models uncertainty. PlaNet (Hafner et al. 2018) found both parts necessary, the deterministic one for remembering what is out of view; DreamerV3 uses the same split. The carried state is a single opaque vector with no lookup of a given past event.
