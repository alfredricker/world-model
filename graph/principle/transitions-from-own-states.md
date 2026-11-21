---
id: principle/transitions-from-own-states
type: principle
title: An object module's transitions are read from the binding layer's own states, with no simulator token and no mask
status: tested
links:
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
An object is a stored pattern and its points in an episode. At each step its transition is built from what the binding layer saw: contact (the box around its points lies side by side with the hand's box), the displacement most of its points made, and whether that displacement is the hand's own. "Held", a state of the simulator that no frame shows, is dropped, and the test asks for an effect that can be seen (the object comes along). The dynamics classes of [card 153.1](../../experiments/153.1-dynamics-classes-from-appearance-and-transitions/card.md) are fitted on these tokens. Tested in [card 157](../../experiments/157-transitions-read-from-the-binding-layers-own-states/card.md) on the experience of the agent of [appearance couplings from own action](appearance-couplings-from-own-action.md). The agent's own pose, extent and action stay given. Limits: contact as boxes in the image, whole-pixel displacements, effects within the step.
