---
id: principle/recall-key-learned-from-recall-outcome
type: principle
title: Recall key learned from the outcome of recall
status: tested
links:
- rel: addresses
  to: problem/recall-key
  basis: claimed
---
The embedding used as the recall key is trained by how well what it retrieves predicts the target, so the notion of a similar case is learned, not fixed. Neural Episodic Control (Pritzel et al. 2017) trains the key network through the value read back from memory; retrieval-augmented RL (Goyal et al. 2022) trains the retrieval query from the agent's loss. The contrast is a fixed pixel or autoencoder key.
Card 133 tested it: a key learned from the outcome of recall lifted pulls from 76.7% to 87.5% and depends on the lock, but missed its accuracy bars ([card 133](../../experiments/133-key-learned-from-recall-outcome/card.md)).
