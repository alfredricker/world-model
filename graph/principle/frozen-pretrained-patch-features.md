---
id: principle/frozen-pretrained-patch-features
type: principle
title: Frozen pretrained patch features
status: adopted
links:
- rel: addresses
  to: problem/perception-from-narrow-signal
  basis: claimed
---
A pretrained DINOv2 encoder gives per-patch features that are not trained on this world; early blocks find when things change, late blocks group by appearance ([card 114.1](../../experiments/114.1-counterfactual-lock/card.md), [card 119](../../experiments/119-variables-from-lasting-change/card.md), [card 122](../../experiments/122-sharper-frames/card.md)).
