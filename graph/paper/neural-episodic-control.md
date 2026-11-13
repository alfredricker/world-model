---
id: paper/neural-episodic-control
type: paper
title: Neural Episodic Control
cite: Pritzel et al. 2017
papi: '1703_01988'
links:
- rel: addresses
  to: problem/recall-key
  basis: claimed
- rel: proposes
  to: principle/recall-key-learned-from-recall-outcome
- rel: proposes
  to: principle/fast-episodic-store-beside-slow-learner
- rel: proposes
  to: principle/store-everything-read-by-similarity
- rel: proposes
  to: principle/keys-separate-from-contents
---
**Problem.** Deep reinforcement-learning agents need orders of magnitude more experience than people. The paper names three causes: small gradient-descent learning rates (to avoid interference), sparse rewards that a network under-fits, and reward information that propagates one step at a time through bootstrapped targets.

**Method.** Neural Episodic Control (NEC) keeps one memory per action, called a differentiable neural dictionary: an append-only list of keys and values. A convolutional network (the DQN architecture) turns the pixel state into a key vector h. A lookup finds the p nearest stored keys (p = 50, using kd-trees) and returns the average of their values weighted by a kernel, 1 / (squared distance + 0.001), which has heavy tails so that distant neighbours do not collapse to one. The stored values are Q estimates: N-step returns (N = 100) with an off-policy bootstrap from the memory itself. Every state the agent visits is written; nothing is learned about when to write. If the exact key exists, its value moves toward the new estimate with a high learning rate (fast, tabular-style update). The network and the stored keys and values are also trained by gradient descent on the squared error between the lookup and the N-step estimate, using a small replay buffer, so the key embedding is shaped by value prediction.

**Evidence.** On Atari (57 games), median human-normalised score at 1, 2, 4, 10, 20, 40 million frames is 16.7, 27.8, 36.0, 54.6, 72.0, 83.3 percent for NEC against 12.8 to 61.9 for MFEC and, at 10 million frames, 15.7 (Nature DQN), 22.4 (prioritised replay) and 3.6 (A3C). Only at 40 million frames does prioritised replay (89.0) pass it. It does not need reward clipping.

**Assumptions and limits.** It needs a scalar reward and value targets; the memory serves value estimation, not recalling what happened. Gains are in the early, data-poor phase; other methods catch up later. Memory is capped at 5x10^5 entries per action, with least-recently-used overwrite.

**Bearing on this project.** It speaks to recall-key: the key (the embedding) is shaped by how well the retrieved values predict the target, so the notion of "similar" is learned from the outcome of recall, unlike the reward-blind pixel or autoencoder keys of its predecessor. It also writes everything and learns only the read.
