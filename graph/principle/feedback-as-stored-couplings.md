---
id: principle/feedback-as-stored-couplings
type: principle
title: An object module feeds back by writing the binding couplings it stored into the binding layer where its pattern is recognised
status: tested
links:
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
Binding modules (one per followed point, a belief over objects) are coupled pairwise by the log-odds that two points are parts of one object, accumulated from whether they moved together ([card 147](../../experiments/147-binding-from-interaction-in-a-sprite-world/card.md)). An object module stores, for a group that motion bound, where its points lie in its own frame, what they look like, and the mean couplings it had inside and to everything else. At the first sight of a scene every stored pattern is tried at every place; a pattern recognised at a place writes those couplings between the points it explains and the rest, and its module's belief that the points are its own; the binding layer then settles as it always does. Recognition is a likelihood ratio: each pattern pixel's chance rate is read from the frame, a hidden pixel costs log 2, pixels explained by a better pattern stop counting, and the bar is the prior against one hypothesis among patterns times places. Tested in [card 154](../../experiments/154-feedback-from-object-modules-to-binding/card.md) on sprite pixels with no masks. Part of [the layers of attractor modules](layers-of-attractor-modules.md). Limits: a pattern is one object's pixels (no change of size, turn or light), an object that never moved has no pattern, and the memory grows by a pattern per object.
