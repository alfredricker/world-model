---
id: paper/merlin
type: paper
title: Unsupervised Predictive Memory in a Goal-Directed Agent (MERLIN)
cite: Wayne et al. 2018
papi: '1803_10760'
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: addresses
  to: problem/object-permanence
  basis: claimed
- rel: proposes
  to: principle/memory-contents-trained-by-prediction
---
**Problem.** A reinforcement-learning agent fails when most of what it needs is out of view (partial observability), especially with a long delay between seeing something and needing it. An external memory trained only by the reward signal (end-to-end) also fails on simple tasks: reward is a thin, delayed teacher for deciding what to store.

**Method.** MERLIN has two parts. The memory-based predictor (MBP) compresses each observation (about 10^4 numbers) into a state variable z_t (about 10^2 numbers), a Gaussian sample. A recurrent network predicts a prior over z_t from past states and actions; the new observation corrects it to a posterior. Every z_t is written to a new memory row; several read heads each make a key and average the most similar rows (normalised dot product). The MBP is trained by a variational lower bound: decoders reconstruct image, self-motion, text, previous action and previous reward from z_t, plus a penalty on the gap between prior and posterior. A further decoder predicts the return from z_t, so z_t keeps reward-relevant information. The policy reads z_t and memory but only the policy is trained by policy gradient; none reaches the MBP. "Retroactive updating" also stores, beside each z_t, a discounted sum of the states that followed.

**Evidence.** On the children's Memory card game, a recurrent agent (RL-LSTM) and one with a reward-trained external memory (RL-MEM) did not solve the task; MERLIN found the optimal strategy. In four one-shot 3D navigation tasks, MERLIN revisited the goal more than twice as often per episode as the comparison agents in the large environment, and beat human testers (extended table). The goal position could be decoded from its state to 1.67 grid units mean Manhattan error on a 15 by 15 grid; baseline states did not match this.

**Assumptions and limits.** It needs a reward for the return-prediction term, and performance fell when that term's weight was too low or too high. It was tested in simulated 3D worlds with a policy-gradient learner; memory grows by one row per step.

**Bearing on this project.** It speaks to state-across-time and recall-key: what is written into memory is trained by predicting the episode, not by the downstream task. Here, short-term memory contents could be trained to predict later observations (a cube shut in a drawer explains a later full drawer), with an outcome term favouring what matters.
