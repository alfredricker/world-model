---
id: principle/state-from-nearness-in-time
type: principle
title: A scene state trained to tell near moments apart, the arm's pose given
status: tested
links:
- rel: addresses
  to: problem/perception-from-narrow-signal
  basis: shown
  evidence: result/172-moments-loss-reads-every-variable
- rel: addresses
  to: problem/object-discovery
  basis: claimed
---
A frame is mapped to a few bounded numbers, trained only to pick the moment near in time out of other moments, with the distance a sum of absolute differences and a frozen term that sees only the two poses. No decoder, no contact signal, no label. The loss is over moments, so a variable counts by how often it tells two moments apart and not by its area. It read every variable of OGBench's scene in five minutes; it did not put them on separate coordinates ([card 172](../../experiments/172-factors-from-sparse-change-over-time/card.md)). With the temperature fixed and eight coordinates, one fit gave a factor per variable; one seed, and the state still moves when nothing in the scene does ([card 173](../../experiments/173-tight-box-sparse-change-factors/card.md)). Over three seeds at a size picked by the held-out loss, two factor every variable and one loses the cube ([card 174](../../experiments/174-do-the-five-factors-hold/card.md)).
