---
id: paper/been-there-done-that
type: paper
title: 'Been There, Done That: Meta-Learning with Episodic Recall'
cite: Ritter et al. 2018
papi: 1805_09692
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: addresses
  to: problem/recall-key
  basis: claimed
- rel: proposes
  to: principle/fast-episodic-store-beside-slow-learner
---
**Problem.** Meta-learning agents learn new tasks quickly but reset their memory at the end of each task. When a task reappears, as in natural environments, they must explore again instead of reusing what they found.

**Method.** Tasks (a Markov decision process plus a context cue) recur in sequence. The agent, "epL2RL", is an LSTM trained to learn to reinforcement-learn (reward is an input), plus an episodic memory, a differentiable neural dictionary. At the end of each task the agent stores its LSTM cell state (the result of its exploration) as the value, with the context cue embedding as the key. On every step it looks up the nearest key (k = 1, cosine distance) and gets back a stored cell state. That state is added to the current cell state through a new learned "reinstatement gate" next to the LSTM's input and forget gates. Training is by ordinary reinforcement learning. Feeding the retrieved vector to the LSTM as an ordinary input failed (chance performance), so gated reinstatement is the key design choice.

**Evidence.** On barcode bandits, Omniglot bandits (cues are different drawings of a character class), compositional bandits and a 4x4 contextual water maze, epL2RL earns more reward than an LSTM agent without memory and than one given the context as input. Regret on the Omniglot bandit falls with each reappearance of a context with weights frozen, so the gain is from memory, not gradient learning; the memoryless agent shows no such fall. In the water maze the agent reaches the goal almost directly after one earlier exposure. In an episodic two-step task, a fitted choice model shows it learned an episodic model-based strategy, as humans do.

**Assumptions and limits.** Context cues are clean and visible in one step (barcodes, pretrained embeddings), episode boundaries are given so the agent writes at the end, and one stored memory is read per step. The authors name as next steps learning the query key, building the key from information gathered over time, and deciding when to write.

**Bearing on this project.** It addresses recall-key directly: its limits section says the key will have to be learned and aggregated over time, possibly from the LSTM state, which is our open problem. Borrowable ideas are storing the whole working state under a cue and reinstating it through a gate rather than as an input.
