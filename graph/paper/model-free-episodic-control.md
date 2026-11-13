---
id: paper/model-free-episodic-control
type: paper
title: Model-Free Episodic Control
cite: Blundell et al. 2016
papi: '1606_04460'
links:
- rel: proposes
  to: principle/fast-episodic-store-beside-slow-learner
- rel: proposes
  to: principle/store-everything-read-by-similarity
---
**Problem.** Deep reinforcement-learning methods take millions of interactions to reach human-level play, because gradient updates absorb a rare good experience slowly. Humans, and the hippocampus in the brain, can exploit a rewarding episode after seeing it once. The paper asks whether a simple model of hippocampal episodic control can do this in hard sequential tasks, including 3D worlds where the same state is almost never seen twice.

**Method.** The agent keeps a table Q_EC(s, a) holding the highest discounted return ever obtained after taking action a in state s. At the end of each episode the table is updated by a backward pass: a new entry stores the return, an existing one keeps the maximum of old and new. The policy takes the action with the highest stored value, with a small random-action rate (0.005). For a state not in the table, the value is the mean of the k nearest stored states (k = 11 on Atari, 50 on Labyrinth), found in one buffer per action; old entries are dropped when the buffer is full. The state is a fixed embedding of the pixels, either a random projection to 64 numbers or the 64 parameters of a variational autoencoder (VAE) trained on one million random-policy frames. Nothing is trained by gradient descent about control.

**Evidence.** On five Atari games and three 3D Labyrinth levels it beat DQN, DQN with prioritised replay and A3C early in training. On "Forage and Avoid" A3C needed over 40 million steps to match what it reached in under 3 million frames. On the sparse-reward Double-T-Maze, A3C never reached positive reward while episodic control did. On Q*bert and River Raid it was overtaken later by parametric methods.

**Assumptions and limits.** It assumes near-deterministic transitions and rewards (the max update is risk-seeking in stochastic worlds). It needs reward, and embeddings are fixed and reward-blind, so small relevant details can be lost (a later paper, Neural Episodic Control, learns it). It does not generalise as parametric networks do.

**Bearing on this project.** It shows recall by nearest-neighbour lookup over stored cases can serve behaviour with almost no learning, which supports recalling remembered touches, but it also shows the key choice matters: a fixed generic embedding is the weakness that the recall-key problem faces.
