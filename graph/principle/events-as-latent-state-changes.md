---
id: principle/events-as-latent-state-changes
type: principle
title: An object module keeps a latent state from step to step; an event is a change of that state, and its conditions are read off the learned transition
status: tested
links:
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
Each object module keeps a discrete latent state that persists (a hidden Markov model with inputs, in coupling form): what is seen of the object depends on the state (it comes along when the hand moves, or not), and the state's change depends on the tokens of the step (contact, grip, whether the hand moved) through sparse couplings, a shared set plus each dynamics class's difference. Fitted by expectation-maximisation on the module's own tokens ([transitions from own states](transitions-from-own-states.md)). An event is a change in the most probable sequence of states; its conditions are the tokens whose change alone makes it improbable, starting from the situation in which it most often happened and only among situations the experience produced. Tested in [card 158](../../experiments/energy-modules/158-events-as-changes-of-an-objects-latent-state/card.md). It restores "held", which no frame shows, as something inferred, and gives GOAL P12's statements (an effect and its conditions). Declared: two state values, conditions within the step, object and hand only.
