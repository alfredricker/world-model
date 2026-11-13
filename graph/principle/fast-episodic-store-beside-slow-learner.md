---
id: principle/fast-episodic-store-beside-slow-learner
type: principle
title: Fast episodic store beside a slow learner
status: untested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
---
Keep a slow network that learns general structure, and next to it a fast store that writes each specific experience once and returns it from a partial cue, so one episode can change behaviour at once without disturbing what the network knows. Proposed by complementary learning systems theory (Kumaran et al. 2016) and built as a stored working state reinstated by a gate in Ritter et al. 2018.
