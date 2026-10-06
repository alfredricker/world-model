# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-06, `master`. **Rung:** 1, not passed.
  **Architecture:** version 17 (card 073). **Direction:** CHARTER's
  "Current direction", rule 8; the standing evaluation is its three
  MiniGrid tiers.
- **Best version:** 17. Tier 1 DoorKey-8x8 **100%** (threshold 99%);
  tier 2 BlockedUnlockPickup **25%**; tier 3 ObstructedMaze-Full-v1:
  memory cannot be built yet.
- **Cards of 2026-10-05/06:** 066–067 keep (the tiers; walking by
  propagation); 068 revise; 069, 071 stop; 070 revise; 072 keep
  (version 16, trying); 073 keep (version 17, orders); 074 revise (a
  kept order overrode found ones).
  - [074.1](experiments/074.1-orders-with-ties-and-split-needs/card.md)
    revise, for cost: found orders followed, ties kept, two-part needs
    split. Decoy with the key known 100%; all 47 tier 2 loops gone; tier
    1 100%; tier 2 **51%** (better than 25%, p = 0.0003); but 15 tier 2
    episodes out of time (limit 6): order tests rebuild card 043's
    conditions in every imagined situation.
- **Next, for the user:** approve
  [076](experiments/076-remembered-orders-with-where/card.md)
  (remembered orders that record where things are; report only; to
  square with the note below). Then 074.2 (to draft): cache conditions
  so 074.1's arm B decides the same within time, as version 18. Later:
  [075](experiments/075-motion-as-learned-system-1/card.md); 068's
  revision; trying's retry; tier 3's memory; 058, 065, 055 (draft).

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
