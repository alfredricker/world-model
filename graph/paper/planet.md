---
id: paper/planet
type: paper
title: Learning Latent Dynamics for Planning from Pixels (PlaNet)
cite: Hafner et al. 2018
papi: '1811_04551'
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: proposes
  to: principle/deterministic-memory-path-in-latent-state
- rel: proposes
  to: principle/memory-contents-trained-by-prediction
---
**Problem.** Planning works well when the dynamics are known, but learning dynamics accurate enough for planning from images is hard. A single image does not show the full state, so the agent needs a belief over a hidden state built from history.

**Method.** PlaNet learns a latent state-space model from the agent's own image-action-reward sequences. The recurrent state-space model (RSSM) splits the state into a deterministic part h_t, updated by a recurrent network from the previous h, stochastic state and action, and a stochastic Gaussian part s_t drawn from h_t. Observation (a decoder) and reward are predicted from both. An encoder infers s_t from h_t and the image (a filtering posterior over past observations). It is trained by a variational bound: reconstruction minus the divergence between posterior and the transition prior. "Latent overshooting" adds the same divergence for multi-step predictions, computed only in latent space, to improve long-range prediction. Actions are chosen with no policy or value network: at each step the cross-entropy method samples 1000 action sequences over a short horizon, scores them by summed predicted rewards in latent space, refits to the top 100, and repeats, replanning every step. New episodes collected this way are added to the data.

**Evidence.** On six DeepMind-control tasks from 64 by 64 images, with one setting for all, PlaNet at 1000 episodes scored, for example, 821 on cartpole swing-up, 662 on cheetah run and 951 on walker walk, against D4PG's (pixels, 100,000 episodes) 862, 524 and 968, and A3C's 558, 214 and 311. It used about 200 times fewer episodes, and beat D4PG on cheetah by 26 percent. Both the deterministic and stochastic paths mattered on all tasks, and without the stochastic path the agent did not learn. Online data collection by planning helped on all tasks.

**Assumptions and limits.** It needs reward to plan, reconstructs pixels (its authors suggest learning without reconstruction), plans over a fixed short horizon with fixed action repeat, and was tested on visually plain simulated tasks.

**Bearing on this project.** The paper notes cartpole requires memorising the cart when it is out of view and credits the deterministic path with carrying information over many steps, a direct answer pattern for state-across-time. The weakness is that the carried state is an opaque vector with no way to look up a specific past event.
