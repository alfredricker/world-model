---
id: paper/slot-attention
type: paper
title: Object-Centric Learning with Slot Attention
cite: Locatello et al. 2020
papi: object-centric-learning-with-slot-attention
links:
- rel: proposes
  to: principle/object-slots
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
**Problem.** Learning object-centric representations from raw images (one vector per object) usually needs supervision or task-specific architectures, and most networks give distributed codes that do not separate objects, so reasoning over compositions generalises poorly.

**Method.** Slot Attention sits between a convolutional encoder and a task head. The input is N feature vectors (with position embeddings); the module holds K slot vectors, initialised by sampling from a shared learned Gaussian, so the number of slots can change at test time and no slot is tied to a class. For T iterations (3 in training), each slot forms a query, inputs form keys and values, and the attention softmax is normalised over the slots, so slots compete to explain each input feature. Each slot takes the attention-weighted mean of the values and updates itself with a shared GRU and a residual MLP. The module is permutation-invariant in the inputs and permutation-equivariant in the slots. For unsupervised discovery, a spatial-broadcast decoder renders each slot as an image plus a mask, and the masked sum is trained against the input by mean squared error. For set prediction, each slot goes through an MLP and the outputs are matched to labelled objects with the Hungarian algorithm.

**Evidence.** Object discovery, Adjusted Rand Index (excluding background), five seeds: CLEVR6 98.8 (IODINE 98.8, MONet 96.2); Multi-dSprites 91.3 (IODINE 76.7, MONet 90.4); Tetrominoes 99.5 (one outlier seed excluded; IODINE 99.2); a plain MLP baseline 60.4, 60.3 and 25.1. Batch size on one 16 GB GPU was 64 against 4 for IODINE, and training took about 24 hours on 8 GPUs against about 7 days. Set prediction on CLEVR matched or beat DSPN; more iterations or slots at test time helped, and accuracy fell gracefully with more objects than trained on.

**Assumptions and limits.** Images are synthetic and static; the paper does not handle video. The slot count K must be set; one Tetrominoes seed (of five) was an outlier and was excluded; the objects must be separable by appearance.

**Bearing on this project.** It speaks to object-discovery: a competitive, unsupervised grouping of patch features into a set of object vectors. The same slot vectors, if carried across frames, would be the units that a memory can write and later find.
