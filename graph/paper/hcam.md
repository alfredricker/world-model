---
id: paper/hcam
type: paper
title: 'Towards mental time travel: a hierarchical memory for reinforcement learning agents'
cite: Lampinen, Chan, Banino and Hill 2021
papi: 2105_14039
links:
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: addresses
  to: problem/object-permanence
  basis: claimed
- rel: addresses
  to: problem/recall-key
  basis: claimed
- rel: proposes
  to: principle/memory-in-event-sized-chunks
---
**Problem.** Reinforcement-learning agents forget details of the past, especially after delays or distractor tasks. A recurrent network (an LSTM) compresses everything into one state, and a transformer that attends over all past steps becomes costly and, in practice, learns poorly when the relevant event is far back and buried in unrelated steps.

**Method.** The Hierarchical Chunk Attention Memory (HCAM) divides the past into fixed-length chunks. Each chunk is stored in full, as a sequence of the agent's layer inputs, together with one summary key made by average-pooling over the chunk. Recall is two-stage. First, a learned query is compared with all summary keys by softmax attention, giving a relevance score per chunk. Second, the top-k chunks (k can be 1) are read in detail with ordinary multi-head attention, and their outputs are summed, weighted by relevance. The block is added to each layer of a Gated TransformerXL agent. Gradients are not passed into stored memory, so the encoder must learn short-term codes. The authors show an auxiliary loss that reconstructs the input image and language is necessary for the memory to be useful.

**Evidence.** In a "ballet" task (recall which dancer performed a distinctive 16-step dance, among 2 to 8 dances with delays of 16 or 48 steps), HCAM stays high with 8 dances while the TransformerXL baseline degrades even though the whole task fits in its window, and the LSTM stays near chance. In a 3D object-permanence task (which object was hidden in which box, with 30-second delays and the agent looking away), HCAM learns the long version well while the baseline stays near chance. After rapid word learning with up to 10 distractor phases, HCAM extrapolates better and recalls words several episodes later, never trained to. It matches specialised memories on Passive Visual Match, Paired Associative Inference (MEMO) and One-Shot StreetLearn (EPN, though slower to learn).

**Assumptions and limits.** Chunk size is fixed, not tied to events (the authors report robustness to it), and training needs the reconstruction loss. Tasks are synthetic.

**Bearing on this project.** It speaks to the state-across-time and recall-key problems: the key it uses is a learned summary of a stretch of experience, which is what our recall key must be, and the cube-in-the-drawer case resembles its hidden-object task. A borrowable idea is coarse-then-fine recall over event-sized chunks, with a summary key learned from the chunk's contents instead of written by hand.
