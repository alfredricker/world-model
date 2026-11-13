---
id: principle/memory-in-event-sized-chunks
type: principle
title: Memory stored in event-sized chunks
status: untested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: addresses
  to: problem/recall-key
  basis: claimed
---
Store the past as chunks, each kept in full with one summary key learned from its contents, and recall coarse-then-fine: pick the relevant chunks by their keys, then read only those in detail. Proposed in HCAM (Lampinen et al. 2021) with fixed-length chunks; here the natural chunk would be one event, bracketed by a touch.
