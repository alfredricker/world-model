---
id: paper/complementary-learning-systems
type: paper
title: 'What learning systems do intelligent agents need? Complementary learning systems theory updated'
cite: Kumaran, Hassabis and McClelland 2016
papi: complementary-learning-systems
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
**Problem.** What learning systems must an intelligent agent have? A single network that learns by small weight changes generalises well but is slow, cannot act on one experience, and forgets old knowledge when new facts are forced in quickly (catastrophic interference).

**Method.** This is a theory paper, not a model; it updates the original complementary learning systems (CLS) theory. The theory says there are two systems. The neocortex is a slow, parametric learner: a multi-layer network whose weights absorb the statistics of the environment through repeated, interleaved examples, with overlapping, similarity-based codes. The hippocampus is a fast, instance-based store. The dentate gyrus and CA3 give each experience its own sparse code (pattern separation), and strengthened connections let a partial cue bring back the whole event (pattern completion). Replay of stored events to the neocortex, interleaved with other experience, teaches it gradually. The update adds three things. Replay is reweighted toward surprising, novel or rewarded events, so rare significant events are not swamped. Recurrent reactivation of several traces together supports inference across episodes (the REMERGE model). Neocortical learning is fast when the new information fits an existing schema.

**Evidence.** The paper reviews rather than reports new results: replay of place sequences in rodents, including shortcuts and reward-biased replay; patients and tasks such as paired-associate inference; rats learning new flavor-place pairs in one exposure only inside a familiar arena (Tse and colleagues); and a simulation in which a network learned schema-consistent facts rapidly without interference. It links the theory to experience replay in the Deep Q-Network, to continual learning, and to external memories such as Neural Turing Machines and memory networks.

**Assumptions and limits.** It rests on mammalian data, mostly rodent; the authors list open questions, including how replay is selected and whether hippocampal codes differ across regions.

**Bearing on this project.** It supports a fast episodic store beside a slow learned model as the answer to state across time and a hidden cube. It also speaks to the recall-key problem: stored events are retrieved by partial cues, and the recurrent step, where a retrieved memory is fed back as a new query, could let a touch find a related earlier event.
