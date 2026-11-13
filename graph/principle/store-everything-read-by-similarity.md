---
id: principle/store-everything-read-by-similarity
type: principle
title: Store everything, read by similarity
status: untested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
---
Write every experience without learning when to write, and learn only the read: return a similarity-weighted average of the nearest stored entries. Model-free episodic control and Neural Episodic Control do this with k nearest neighbours; modern Hopfield networks (Ramsauer et al. 2020) show the update is transformer attention, with a temperature that moves from one matching entry to an average of similar ones.
