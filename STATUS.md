# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-06, on `master` (not pushed). **Rung:** 1, not
  passed. **Architecture:** version 17 (card 073: needs ordered by the
  states they conflict over, on version 16).
- **Direction:** CHARTER.md's "Current direction", with rule 8. User,
  2026-10-05: the standing evaluation is CHARTER's three MiniGrid tiers;
  walking knows only its next step and whether it brings it closer.
- **Best version and its scores** (CHARTER's tiers, card 066's seeds):
  version 17. Tier 1 DoorKey-8x8 **100%** (200; threshold 99%); tier 2
  BlockedUnlockPickup **25%** (100; version 16 2%); tier 3
  ObstructedMaze-Full-v1: memory cannot be built yet.
- **Cards of 2026-10-05/06** (066 keep: the tiers; 067 keep, version 15:
  walking by propagation):
  - [068](experiments/068-play-starts/card.md) revise: blue's door is not
    openable through recall's predicted tile for a toggled door.
  - [069](experiments/069-relations-as-learned-weights/card.md) stop: a
    learned relation beside one-tile terms, on nine fixed colours, learned
    a threshold per door colour (45 of 66 for a left-out colour).
  - [070](experiments/070-relation-only-fresh-hues/card.md) revise: the
    encoder learns "this key fits this door" (66 of 66; 9 of 9 for unseen
    hues), but recall weighs the door's identity 40 times the relation.
  - [071](experiments/071-recall-fit-by-combination/card.md) stop.
  - [072](experiments/072-try-the-likeliest-way/card.md) keep, version
    16: trying the likeliest untried way; decoy folds 7% to 95–98% (a
    known key: 94%); refits after a try dropped (the user).
  - [073](experiments/073-order-needs-by-conflicting-states/card.md)
    keep, version 17: reasonable goal orderings; tier 2 2% to 25%.
  - [074](experiments/074-conflicts-read-from-plans/card.md) revise:
    orders read from plans, no list of parts; seed 1001002 and 28 of 47
    tier 2 loops fixed, but a kept order overrode found ones (decoy 94%).
- **Next, for the user:** approve
  [074.1](experiments/074.1-orders-with-ties-and-split-needs/card.md)
  (found orders followed; ties kept; card 043's two-part needs split);
  prioritise [075](experiments/075-motion-as-learned-system-1/card.md)
  (walking as a learned System 1). Later: card 068's revision; trying's
  retry after a refusal; tier 3's memory; cards 058, 065, 055 (draft).

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
