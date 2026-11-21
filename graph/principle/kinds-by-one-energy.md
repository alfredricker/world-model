---
id: principle/kinds-by-one-energy
type: principle
title: Kinds of behaviour are found by lowering one energy - how badly each item's kind predicts its records, plus a latent-cause prior on the grouping
status: tested
links:
- rel: addresses
  to: problem/prior-over-unseen-state
  basis: claimed
---
An item is an individual (an object, a word) with records of what it did. A kind is a set of couplings (a shared set plus a sparse difference). The energy is the sum over items of minus the log probability of their effect tokens under their kind, minus the log of a Chinese-restaurant prior on the grouping. Every item learns its own sparse difference; a difference is stored as a kind when that lowers the energy counting only what it explains of other items' records; kinds are refitted, merged when that lowers the energy, and given up when left with one member. One hand-set number, the prior's concentration. Looks are coupled to kinds by every item, in proportion to what its records say, then by its belief from looks and records together. Tested in [card 153](../../experiments/153-kinds-found-by-one-energy-in-two-domains/card.md) in sprites and in a stream of words. It replaces the thresholds of [the object module](object-module-what-where-does.md)'s first learner. Weakness: a kind is proposed by one item and credited only with what it explains, so it fails when no item has enough records of its own. [Card 153.1](../../experiments/153.1-dynamics-classes-from-appearance-and-transitions/card.md) answers that weakness: the class of object i is inferred from the appearance prior times the likelihood of its transitions during learning as at test, a class is proposed from the pooled transitions of all objects sharing an appearance attractor (or from one object), and it is kept only if it lowers the negative log likelihood of held-out transitions; no prior on the number of classes. It passes at 30 episodes and at 100 sentences, at the cost of extra small classes and 4 to 34 times the run time.
