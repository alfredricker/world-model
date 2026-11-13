---
id: paper/dreamerv3
type: paper
title: Mastering Diverse Domains through World Models (DreamerV3)
cite: Hafner, Pasukonis, Ba and Lillicrap 2023 (arXiv:2301.04104)
papi: dreamerv3
links:
- rel: proposes
  to: principle/memory-contents-trained-by-prediction
- rel: proposes
  to: principle/deterministic-memory-path-in-latent-state
---
**Problem.** Reinforcement-learning algorithms usually need expert tuning for each new domain, because reward and observation scales differ widely. The paper asks for one algorithm and one set of hyperparameters that works across many domains, and tests it on Minecraft diamond collection from scratch.

**Method.** DreamerV3 has three networks. The world model is a recurrent state-space model: a recurrent sequence model gives h_t from the previous h, latent z and action; an encoder gives a categorical latent z_t (discrete, straight-through gradients) from h_t and the input x_t; a predictor gives the prior of z_t from h_t alone; decoders predict the input, the reward and an episode-continuation flag. The loss has a prediction term, a dynamics term (trains the prior toward the posterior) and a representation term (the reverse, with stop-gradients; weights 1, 1 and 0.1), with free bits so already-predictable parts stop being penalised. The actor and critic are trained entirely on 16-step trajectories imagined in the latent space of the world model, with lambda-returns. Robustness comes from symlog transforms of inputs, a symexp two-hot (binned) loss for rewards and values, returns normalised by their 5th to 95th percentile range, and a fixed entropy weight.

**Evidence.** With fixed settings it outperformed tuned specialist methods on 150-plus tasks in 8 domains (Atari, ProcGen, DMLab, Atari100k, proprioceptive and visual control, BSuite, Minecraft). In Minecraft, all trained agents found diamonds within 100 million steps using one GPU for nine days, with no human data or curriculum; the compared methods reached the iron pickaxe but found none. Ablations show every robustness technique matters, most of all the KL objective. Larger models (12M to 400M parameters) learn faster and score higher. Dreamer relies mostly on the world model's unsupervised signal rather than on reward.

**Assumptions and limits.** It needs reward, large data budgets (hundreds of thousands to hundreds of millions of steps) and a fixed observation format; its memory is the recurrent state.

**Bearing on this project.** It confirms that a latent state trained to reconstruct and predict, plus a recurrent carry, is a strong general memory of the episode so far, and its finding that the unsupervised world-model signal carries most of the learning supports training memory contents by prediction. It gives no way to recall a specific past event on demand.
