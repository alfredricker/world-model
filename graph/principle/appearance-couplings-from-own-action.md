---
id: principle/appearance-couplings-from-own-action
type: principle
title: Binding couplings are predicted from appearance by a network taught only by the couplings the agent's own action produced
status: tested
links:
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
The binding layer couples two points by the log-odds that they are parts of one object, accumulated from whether they moved together ([card 147](../../experiments/147-binding-from-interaction-in-a-sprite-world/card.md)). A small network reads the frame around two points and their offset and predicts one of three states for the pair: together, apart, no evidence. Its labels are the binding layer's own couplings at the end of each episode (evidence from three steps on); no mask, no similarity loss, no grouping prior. At first sight the expected coupling is added to the binding energy, with the pairs expected to show nothing centred, and the layer settles as always; a group bound this way that does not move is stored as a pattern too, so that its transitions reach the dynamics classes. Tested in [card 156](../../experiments/156-appearance-couplings-taught-by-the-agents-own-action/card.md) with the experience gathered by a script and by an agent that goes to the groups its own binding sees and tries both grips. It extends [feedback as stored couplings](feedback-as-stored-couplings.md) from one object's stored couplings to couplings predicted for any appearance. Limits: the read-out was chosen in development, some labels fall on the ground, and the reach of the embedding is 7 by 7 pixels.
