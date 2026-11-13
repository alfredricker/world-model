---
id: paper/retrieval-augmented-rl
type: paper
title: Retrieval-augmented reinforcement learning
cite: Goyal et al. 2022, arXiv:2202.08417
papi: '2202_08417'
links:
- rel: addresses
  to: problem/recall-key
  basis: claimed
- rel: proposes
  to: principle/recall-key-learned-from-recall-outcome
---
- Authors: Anirudh Goyal and colleagues (full author list in the primary source).
- Publication: ICML 2022, PMLR 162; inspected arXiv:2202.08417v4.
- Primary source: https://arxiv.org/html/2202.08417v4
- Method: sections 2.1–2.3, algorithm 1. R2A combines a recurrent retrieval process with an agent. Sampled historical trajectories are re-encoded, summarized bidirectionally, and selected through attention/top-k retrieval; retrieved context influences values or policy.
- Evidence: offline multi-task DQN and online Atari R2D2 experiments report improved learning over their baselines. These are policy/value results, not learned relational world-model planning.
- Limits: sampled retrieval bounds access cost but can miss rare evidence. Memory uses recurrent slots as read state, not discovered visual objects. Raw retrieval is not a guarantee of transfer or reliable belief revision.
- Quote: “re-encode the raw experiences” (paper: Retrieval Augmented Reinforcement Learning; arXiv:2202.08417; source: paper PDF; ref: section 2.2).

**Proposed GRL adaptation — original integration.** Store completed experience segments with observation provenance, time, executed controls, and observed outcomes. Retrieve evidence for candidate procedures, compare its relations with the current situation, and train the read for prediction and execution. PHWM's unbottlenecked retrieval failure motivates routing relevant relational comparisons rather than directly copying retrieved outcomes. This relational read is our integration with local evidence, not R2A's published algorithm.

**Implementation questions.** Version or re-encode memory under representation updates. Completed past trajectories can contain their own future outcomes; the current query cannot see future observations. Maintain exceptions alongside earlier evidence and distinguish historical facts from current beliefs. Test retrieval removal, shuffled memories, conflicting episodes, retention, and held participant bindings. Learned forgetting/revision remains a hypothesis requiring a mechanism and a behavioral test.
