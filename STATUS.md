# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05, on `master` (not pushed). **Rung:** 1, not
  passed. **Architecture:** version 15 (card 067's walking by
  propagation), card 054's encoder.
- **Direction:** CHARTER.md's "Current direction", with rule 8. User,
  2026-10-05: the standing evaluation is CHARTER's three MiniGrid tiers;
  walking knows only its next step and whether it brings it closer.
- **Best version and its scores** (CHARTER's tiers, card 066's seeds):
  version 15. Tier 1 DoorKey-8x8 **100%** (200; threshold 99%); tier 2
  BlockedUnlockPickup **0%** (100); tier 3 ObstructedMaze-Full-v1: memory
  cannot be built yet (counted as no different).
- **Today's cards:**
  - [066](experiments/066-minigrid-tiers/card.md) keep: the harness and
    why tier 2 fails.
  - [067](experiments/067-closeness-by-propagation/card.md) keep, version
    15: walking by propagation; the same decisions, setup 21 s against 47.
  - [068](experiments/068-play-starts/card.md) revise: play starts give
    every colour its open door walked through; 5 of 6 locked doors
    openable (4 before); tiers unchanged. Blue fails in recall's
    predicted tile for a toggled door (card 038's ways), not in memory.
    Recall's view-set fit no longer builds its 7 GB tensor (same numbers).
  - [069](experiments/069-relations-as-learned-weights/card.md) stop: a
    learned relation over whole vectors, trained into the encoder, failed
    its gate (45 of 66 for a left-out colour, against 47 before); it
    learned a threshold per door colour.
- **Next, for the user to choose:** card 068's revision (the tile a door
  becomes when toggled); the hand's conditions (put down before picking
  up), which tier 2 needs; a second relation attempt (the relation as the
  only path for what depends on both tiles, fresh colours per update);
  later, recall's view-set fit at tier 3's scale.
- **The user's demo** is on 2026-10-06: version 15 on tier 1.
- **Later:** rung 1's clearing commitment (card 058); the both world's
  random steps (065); card 055 (draft).
