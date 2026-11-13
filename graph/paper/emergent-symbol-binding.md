---
id: paper/emergent-symbol-binding
type: paper
title: Emergent Symbols through Binding in External Memory
cite: Webb, Sinha and Cohen 2020 (arXiv:2012.14601)
papi: '2012_14601'
links:
- rel: proposes
  to: principle/keys-separate-from-contents
---
**Problem.** Networks learn abstract rules (same versus different, ABA versus ABB patterns) from a few examples poorly, and fail to apply them to entities not seen in training. Some argue symbol-like variable binding is needed. The paper asks whether a network can learn such binding itself from data.

**Method.** The Emergent Symbol Binding Network (ESBN) pairs an LSTM controller with an external memory split into two parts: values and keys. Each input image is encoded by a separate encoder into an embedding z_t, and z_t itself is stored, unchanged by the LSTM, as a value. The LSTM outputs a separate key vector at each step and writes it beside that value. To read, the current embedding is compared by dot product with all stored values; a softmax gives weights and a sigmoid gives per-entry confidences. The weighted sum of the keys (with the confidence) is fed to the LSTM at the next step, gated. So the controller never sees the content of the images, only the keys it wrote, which become learned variable names, and the content-addressed read tells it which earlier slot matches the current item. Training is end-to-end by the task loss, with temporal context normalisation of the embeddings.

**Evidence.** Four tasks on 100 Unicode-character images: same/different, relational match-to-sample, distribution-of-three and identity rules. With 10^4 training problems (a few hundred in the extreme regimes) and test images withheld from training (up to 98 of 100 held out for same/different), ESBN generalised at 95 percent or better on all four tasks. LSTM, Neural Turing Machine, Metalearned Neural Memory, Relation Net, Transformer and PrediNet did not in the most extreme regimes. ESBN reached near-perfect training accuracy in 100 to 200 updates; the others needed thousands or tens of thousands.

**Assumptions and limits.** Sequences are short (up to 9 images), the memory is reset for every problem, supervision is a task label, and the task rule is purely relational; it was not tested on control or long episodes.

**Bearing on this project.** It supports keeping keys separate from contents: the key the agent writes can be an arbitrary learned name, while contents stay as raw perceptual vectors. For recall-key this suggests the key for an episode could be written by a learned controller, not computed from the content.
