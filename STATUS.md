# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-09, `master`. **Rung:** 1, not passed.
  **Architecture:** version 20 (cards 091.1 and 087's forward prior, kept
  overnight; the user may revert). **Direction:** CHARTER's "Current
  direction", rule 8; the standing evaluation is its three MiniGrid tiers.
- **Best version:** 20. Tier 1 DoorKey-8x8 **100%** in 24.4 steps; tier 2
  BlockedUnlockPickup **95%** (19: 79%, 18: 59%); tier 3
  ObstructedMaze-Full-v1 **14/30** within an hour per episode (18: 5/30;
  10 seeds won, 1 lost), 1.45 s per step.
- **Versions 19–20** ([091](experiments/091-router-trained-on-diverse-worlds/card.md),
  [091.1](experiments/091.1-router-within-identity/card.md)): a router over recall's keys as
  tokens is recall's prior (19 alone looped on tier 3, 0/30); 20 votes within
  the same codes first, and forward's prior is card 087's property network.
- **View effects** ([092](experiments/092-view-effects-in-recall/card.md), revise): goal into
  view first; tier 2 half the steps, same successes; tier 3 too slow (0/20).
- **Object files** ([094](experiments/094-object-files-and-layout/card.md), revise): planner unchanged.
  [094.1](experiments/094.1-relative-position-recall/card.md) **stop**. [094.2](experiments/094.2-router-on-admitted-conditions/card.md): tiers 1–2 met
  (200, 95); tier 3 not run, superseded by 095 (decision with the user).
- **State change** ([095](experiments/095-state-change-prediction/card.md), a series): recall predicts
  every token's change by one rule. [095.1](experiments/095.1-which-tokens-change/card.md) (offline):
  arm C (exemplar vote, network prior) learns the front and hand roles,
  fixes the red box in every colour (100%), held-out within 0.01% of
  version 20; **keep**. [095.2](experiments/095.2-recall-at-scale/card.md): index and forgetting keep every
  prediction, vote 31× faster, 18× less RAM; gate dropped. [095.3](experiments/095.3-state-change-in-the-agent/card.md) draft.
- **Then:** [096](experiments/096-structure-memory-by-consolidation/card.md) structure, [097](experiments/097-procedural-memory/card.md) procedural memory.
- **Waiting:** card 086 (approved, gate passed); colour transfer (088–090) paused.
- **Overnight 2026-10-08** ([overnight-10-08.md](overnight-10-08.md)): 091's
  tier 3, 092, 091.1; read it, then delete it.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
