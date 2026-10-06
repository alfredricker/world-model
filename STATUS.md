# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05, on `master` (not pushed). **Rung:** 1, not
  passed. **Architecture:** version 16 (card 072: card 070's encoder,
  `rel:P`, trying the likeliest untried way).
- **Direction:** CHARTER.md's "Current direction", with rule 8. User,
  2026-10-05: the standing evaluation is CHARTER's three MiniGrid tiers;
  walking knows only its next step and whether it brings it closer.
- **Best version and its scores** (CHARTER's tiers, card 066's seeds):
  version 16. Tier 1 DoorKey-8x8 **100%** (200; threshold 99%); tier 2
  BlockedUnlockPickup **2%** (100; no different from version 15's 0%);
  tier 3 ObstructedMaze-Full-v1: memory cannot be built yet.
- **Today's cards:**
  - [066](experiments/066-minigrid-tiers/card.md) keep: the harness and
    why tier 2 fails.
  - [067](experiments/067-closeness-by-propagation/card.md) keep, version
    15: walking by propagation; the same decisions, setup 21 s against 47.
  - [068](experiments/068-play-starts/card.md) revise: blue's door is not
    openable through recall's predicted tile for a toggled door.
  - [069](experiments/069-relations-as-learned-weights/card.md) stop: a
    learned relation beside one-tile terms, on nine fixed colours, learned
    a threshold per door colour (45 of 66 for a left-out colour).
  - [070](experiments/070-relation-only-fresh-hues/card.md) revise: with
    the relation as the only path and fresh hues, the encoder learns
    "this key fits this door" (66 of 66; 9 of 9 for unseen hues; tier 1
    still 100%). But recall weighs the door's identity 40 times the
    relation, so a key never seen opening its door is predicted not to:
    the decoy folds 7%.
  - [071](experiments/071-recall-fit-by-combination/card.md) stop: each
    quarter of the vector mixes a tile's kind and colour.
  - [072](experiments/072-try-the-likeliest-way/card.md) keep, version
    16: trying the likeliest untried way; decoy folds 7% to 95–98% (a
    known key: 94%); refits after a try dropped (the user).
- **Next, for the user:** the hand's conditions (tier 2, the decoy
  failures; ARCHITECTURE's known limits); learned projections for the
  tiles' conditions (card 071); card 068's revision; tier 3's memory.
- **The user's demo** is on 2026-10-06: version 15 on tier 1. **Later:**
  card 058 (rung 1's clearing commitment); 065; 055 (draft).
