---
id: paper/modern-hopfield-networks
type: paper
title: Hopfield Networks is All You Need
cite: Ramsauer et al. 2020 (arXiv:2008.02217)
papi: hopfield-networks-is-all-you-need
links:
- rel: proposes
  to: principle/store-everything-read-by-similarity
---
**Problem.** Classical Hopfield networks store few patterns (about d patterns in d dimensions, or about 0.14 d with small errors) and work on binary states, so they have not fitted into deep networks. The paper asks for an associative memory with continuous states that is differentiable, stores many patterns and retrieves in one step.

**Method.** Store N patterns as columns of X (each of dimension d). For a query state xi, the new energy is minus a scaled log-sum-exp of the similarities X^T xi, plus half the squared norm of xi and constants. The update rule is xi_new = X softmax(beta X^T xi): a similarity-weighted average of the stored patterns. The inverse temperature beta decides the result: low beta gives the average of all patterns; intermediate beta gives metastable states, averages over a group of similar patterns; high beta gives a single stored pattern. Updates converge globally to a stationary point of the energy, retrieval takes one step with exponentially small error, and capacity grows exponentially with d (proportional to c^((d-1)/4) under stated conditions). The update is the same as transformer attention, which explains heads as global averaging or partial averaging. The authors propose three layers: Hopfield (associate two sets), HopfieldPooling (a learned query pools a set) and HopfieldLayer (a learned stored set acts like prototypes).

**Evidence.** On multiple-instance learning, Hopfield pooling reached AUC 91.3 on Tiger, 94.9 on Elephant and 89.5 on UCSB breast cancer, above the best prior numbers listed (91.0, 94.4, 88.0), though below on Fox (64.05 versus 73.8). Immune-repertoire classification with about 300,000 instances per sample: AUC 0.832 versus 0.825 for the best baseline. On 75 small UCI datasets it had the best average rank. It also set the state of the art on two drug-design datasets.

**Assumptions and limits.** Capacity results assume patterns separated enough to be distinguishable; with similar patterns retrieval returns averages (metastable states); beta and the pattern projections are learned or tuned.

**Bearing on this project.** Recall by soft similarity over stored vectors is the same operation as attention, so a recall-key can be a learned query into a bank of stored episode vectors, and beta lets recall move between "the single matching case" and "the average of similar cases".
