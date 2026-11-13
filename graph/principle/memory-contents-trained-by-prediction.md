---
id: principle/memory-contents-trained-by-prediction
type: principle
title: Memory contents trained by prediction
status: untested
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
---
What is written to memory is a compressed state trained to predict the episode (reconstruct observations, predict the next state), not a vector shaped only by the downstream task reward. MERLIN (Wayne et al. 2018) shows this against reward-trained memory on a card game and 3D navigation; Dreamer and PlaNet train their latent state the same way. Here it would mean training short-term memory contents to predict later observations, such as why a drawer is full.
