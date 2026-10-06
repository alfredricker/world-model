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
  - [068](experiments/068-play-starts/card.md) revise: play starts make
    5 of 6 locked doors openable (4 before); tiers unchanged; blue fails
    in recall's predicted tile for a toggled door (card 038's ways).
  - [069](experiments/069-relations-as-learned-weights/card.md) stop: a
    learned relation beside one-tile terms, on nine fixed colours, learned
    a threshold per door colour (45 of 66 for a left-out colour).
  - [070](experiments/070-relation-only-fresh-hues/card.md) revise: with
    the relation as the only path and fresh hues, the encoder learns
    "this key fits this door" (66 of 66; 9 of 9 for unseen hues; tier 1
    still 100%). But recall weighs the door's identity 40 times the
    relation, so a key never seen opening its door is predicted not to:
    the decoy folds 7%.
  - [071](experiments/071-recall-fit-by-combination/card.md) stop:
    fitting recall by leaving whole combinations out cut the door's weight
    but not enough; each quarter of the vector mixes kind and colour.
- **Next, for the user to choose:** the tiles' own conditions read
  through learned projections, as the relation is (card 071's pointer);
  the hand's conditions (tier 2); card 068's revision; later, recall's
  view-set fit at tier 3's scale.
- **The user's demo** is on 2026-10-06: version 15 on tier 1.
- **Later:** card 058 (rung 1's clearing commitment); 065; 055 (draft).
