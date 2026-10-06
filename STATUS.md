# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-06, `master`. **Rung:** 1, not passed.
  **Architecture:** version 18 (cards 074–074.2). **Direction:** CHARTER's
  "Current direction", rule 8; the standing evaluation is its three
  MiniGrid tiers.
- **Best version:** 18. Tier 1 DoorKey-8x8 **100%** in 24.4 steps
  (threshold 99%; version 17: 20.8 steps); tier 2 BlockedUnlockPickup
  **59%** (version 17: 25%); tier 3 ObstructedMaze-Full-v1: memory cannot
  be built yet.
- **Cards of 2026-10-06:** 073 keep (version 17); 074 revise; 074.1
  revise for cost (its step counts were wrong, a counter clash; marked);
  [074.2](experiments/074.2-orders-within-time/card.md) keep, version 18:
  orders read from plans, found orders followed, ties kept, two-part
  needs split, caches with the same actions. Tier 2 out of time 4 (6).
- **Known cost of version 18:** ties ranked by acts before steps make
  tier 1 routes 17% longer (walks to the door before the key in front).
- [076](experiments/076-router-for-remembered-orders/card.md) stop: a
  router over the situation's tokens; tier 2 97.7% against "no order"
  96.2%, but worse than "no order" in the decoy world.
- [077](experiments/077-relational-router/card.md) stop: the router
  reading what tokens do and card 070's relations, not appearance.
  Inputs keep orders (99.8%), tier 2 unchanged, but tier 2 to decoy
  world transfer 16% caught at 3% precision (appearance: 17%, 3%).
- **Next, for the user:** a route relation from walking ("this tile
  stands on the way to that target") as the router's input? CHARTER
  wording on networks inside components (proposed in chat). Later: tie
  cost by steps; 075; 068's revision; trying's retry; tier 3's memory.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
