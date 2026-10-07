# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-07, `master`. **Rung:** 1, not passed.
  **Architecture:** version 18 (cards 074–074.2). **Direction:** CHARTER's
  "Current direction", rule 8; the standing evaluation is its three
  MiniGrid tiers.
- **Best version:** 18. Tier 1 DoorKey-8x8 **100%** in 24.4 steps
  (threshold 99%; version 17: 20.8 steps); tier 2 BlockedUnlockPickup
  **59%** (version 17: 25%); tier 3 ObstructedMaze-Full-v1: memory cannot
  be built yet.
- **Version 18** ([074.2](experiments/074.2-orders-within-time/card.md),
  2026-10-06): orders read from plans, ties kept, two-part needs split,
  caches. Tier 2 out of time 4 (6); tier 1 routes 17% longer.
- **Learned routing (076–083, stop):** orders from tokens learned in tier
  2, carried to the decoy world only with "on the way" (078: 41% at 18%);
  a readout linear in card 070's relation opened a never-seen pair in 6
  of 6 folds, unstably (080); reasoners over recalled tries failed gates.
- **Conditions over roles and relations**, all stop, left-out pair 0 of
  6. [084](experiments/084-conditions-over-roles-and-relations/card.md)–[084.2](experiments/084.2-back-off-most-specific-first/card.md):
  identity admitted first made every general condition redundant.
  [084.3](experiments/084.3-conditions-without-constants-first/card.md),
  [084.4](experiments/084.4-gamma-by-combinations-left-out/card.md)
  stop: roles and cuts first; the pair's cell holds only openings in
  every fold, tier 2's hand table right (9/9, 3/3); diagnosis 6 of 6,
  declared 0 of 6: γ at its grid's top, fitted on outcome classes that
  name the result tile, which no other combination has. Next: γ on
  the category, combinations left out.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
