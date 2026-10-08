# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-08, `master`. **Rung:** 1, not passed.
  **Architecture:** version 18 (cards 074–074.2). **Direction:** CHARTER's
  "Current direction", rule 8; the standing evaluation is its three
  MiniGrid tiers.
- **Best version:** 18. Tier 1 DoorKey-8x8 **100%** in 24.4 steps
  (threshold 99%; version 17: 20.8 steps); tier 2 BlockedUnlockPickup
  **59%** (version 17: 25%); tier 3 ObstructedMaze-Full-v1: memory cannot
  be built yet.
- **Version 18** ([074.2](experiments/074.2-orders-within-time/card.md)): orders read from plans.
- **Conditions over roles and relations** (084–084.4, stop): admitted
  before identity ([084.3](experiments/084.3-conditions-without-constants-first/card.md)) they get every table right.
- **Tier 3's memory** ([085.3](experiments/085.3-sampled-queries-constant-rate/card.md)):
  linear, about 68 minutes in full; fits now stored per memory
  (`WM_STORED_FITS=1`, bit-identical); tier 3's built and stored (about 85 minutes once).
- **Version 19** ([086](experiments/086-version-19-general-conditions/card.md),
  approved 2026-10-08): 084.3's conditions in the agent's recall; judged
  by the tiers and "right key within 2 tries" in every colour fold.
  Gate passed. Main runs wait (now after 087–089).
- **Properties** ([087](experiments/087-properties-as-learned-action-effects/card.md), revise):
  a network on the encoder's vector reads walkable, pick up and toggle
  for every tile, new hues included; a hue never in memory goes from
  0–16% to 96–100% in five folds, green 75% (untraced). Tiers 1–2 no worse.
- **Relation from ignored directions** ([088](experiments/088-relation-from-ignored-directions/card.md),
  revise): the leftover is hue, but coded per kind (70–88° apart); next,
  [089](experiments/089-encoder-with-a-made-of-part/card.md) (stop): a "made of" path lost colour in training;
  nine fixed colours code colour as a table. Drafted: [090](experiments/090-encoder-on-fresh-hues/card.md), fresh hues in the encoder's stream.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
