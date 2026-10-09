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
- **View effects** ([092](experiments/092-view-effects-in-recall/card.md), revise):
  when the goal's tile is unseen, bring it into view first (explore, or
  open a door ranked by recall's "what came into view"). Gate passed (17
  of 091's 21 tier 2 failures with the true effect). Tier 2 no more
  successes than version 20 but half the steps (64 against 119), nearly
  no random actions; tier 3 too slow (searches the map every step;
  0/20). Proposed revision: keep the target between steps, search
  incrementally, gate on speed.
- **Tier 2's remaining failures** (5 of 100, the same with and without
  092): one traced (1002017) is a 6-step loop holding a key before the
  box; recall's prediction is right, the cause is elsewhere in planning.
- **Card 086** (general conditions): approved, gate passed; runs wait.
- **Colour transfer** paused (cards 088–090); focus is tiers 2 and 3.
- **Overnight 2026-10-08** ([overnight-10-08.md](overnight-10-08.md)): 091's
  tier 3, 092, 091.1; read it, then delete it.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
