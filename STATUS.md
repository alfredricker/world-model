# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-08, `master`. **Rung:** 1, not passed.
  **Architecture:** version 18 (cards 074–074.2). **Direction:** CHARTER's
  "Current direction", rule 8; the standing evaluation is its three
  MiniGrid tiers.
- **Best version:** 18. Tier 1 DoorKey-8x8 **100%** in 24.4 steps
  (threshold 99%); tier 2 BlockedUnlockPickup **59%**; tier 3 ObstructedMaze-Full-v1 **5/30** within an
  hour per episode (1.7 s per step; 23 of 30 stopped by the hour).
- **Tier 3's memory** ([085.3](experiments/085.3-sampled-queries-constant-rate/card.md)):
  fits stored per memory (`WM_STORED_FITS=1`); tier 3's built once (85 minutes).
- **Version 19** ([086](experiments/086-version-19-general-conditions/card.md)): approved, gate passed; runs wait.
- **Properties** ([087](experiments/087-properties-as-learned-action-effects/card.md), revise):
  a network on the encoder's vector reads walkable, pick up and toggle
  for every tile, new hues included; a hue never in memory goes from
  0–16% to 96–100% in five folds, green 75% (untraced). Tiers 1–2 no worse.
- **Relation from ignored directions** (088–090, revise/stop/revise): what
  properties ignore is hue, a quantity within every kind (R² 0.99), but
  keys' and doors' hue directions never line up (31–88°), not with a
  made-of path (089) nor fresh hues (090); MiniGrid draws a locked door's
  panel darker than its key. Card 070's relation, learned from whether the
  door opens, does line them up (66 of 66). Colour transfer paused; focus
  is tiers 2 and 3 (the user, 2026-10-08).
- **Router** ([091](experiments/091-router-trained-on-diverse-worlds/card.md), running):
  a network over recall's keys as tokens, trained on tiers 1–2 and the
  decoy world, votes over stored keys as recall's prior. Tier 1 100%,
  tier 2 **79%** (version 18: 59%; 21 seeds won, 1 lost). Tier 3:
  `runs/091/tier3.sh`, for the user to run (about 2 hours).
- **Next** ([092](experiments/092-view-effects-in-recall/card.md), draft): view effects in recall, queried by goal condition.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
