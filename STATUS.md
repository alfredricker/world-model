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
- **Known cost of version 18:** tier 1 routes 17% longer (ties by acts).
- **Routers over the situation's tokens** (076–078, stop): orders from
  tokens learned in tier 2 (97.7% against "no order" 96.2%) but did not
  carry to the decoy world; "on the way" (078) is what orders turn on
  (93% of decoy orders) and gave the first transfer (41% caught, 18%).
- [079](experiments/079-relational-effect-recall/card.md) /
  [080](experiments/080-relation-as-ordered-quantity/card.md) stop:
  toggle recall reading roles and card 070's relation, no identity. A
  readout linear in the relation opens a key never seen opening its door
  in 6 of 6 colour folds (version 18: 0 of 6), but unstably over seeds.
- **CHARTER** (2026-10-06): recall and networks, each where it predicts
  better. Reasoning over recalled tries
  ([082.1](experiments/082.1-reasoning-with-both-queries/card.md), [083](experiments/083-reasoning-with-a-learned-relation/card.md)) stop:
  gates fail (≤ 49; ≤ 54 of 56 with card 070's relation).

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
