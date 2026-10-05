# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05 (overnight), on `master` (not pushed). **Rung:**
  rung 1 begun (cards 056, 059), not passed. **Architecture:** version
  14 with card 054's encoder (confirmed by the user 2026-10-05).
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-10-05: no learned codebook (identity up to
  noise); perception keeps every visible difference; gates on behaviour.
- **Overnight objective (the user):** a principled, scalable encoder
  that keeps the goal/condition hierarchy, then goals from example
  frames, then a view smaller than the map. Card by card in
  [overnight-10-05.md](overnight-10-05.md); all 10 cards used.
- **Where it stands:**
  - Encoder: [054](experiments/054-identity-up-to-noise/card.md) revise
    (planner and conditions kept on 4 seeds; recall's "other key" weak);
    [063](experiments/063-replay-what-the-transition-model-misses/card.md)
    stop (prioritized replay did not help; the per-part comparison is
    the limit).
  - Goals: [056](experiments/056-goals-from-example-frames/card.md) revise
    (frames reached as well as written-in goals);
    [059](experiments/059-how-soon-from-the-chain/card.md) revise (how
    soon, rank 0.97–0.99); [065](experiments/065-goals-from-small-frames/card.md)
    keep (goals from small frames, 98–100%).
  - Small view, all keep: [057](experiments/057-view-smaller-than-the-map/card.md)
    v11, [060](experiments/060-walls-hide-what-is-behind/card.md) v12
    (occlusion), [061](experiments/061-moves-learned-from-the-small-view/card.md)
    v13 and [062](experiments/062-tries-stored-as-believed/card.md) v14
    (memory learned from the small view),
    [064](experiments/064-small-view-on-the-harder-worlds/card.md) (two
    doors 30/30, cluttered 100/100).
  - [058](experiments/058-route-or-clear-the-way/card.md) stop (dithering).
- **Next (the user, 2026-10-05):** CHARTER's three MiniGrid tiers
  (DoorKey-8x8 ≥ 99%; BlockedUnlockPickup, ObstructedMaze-Full against
  the best version). Card 066: the adapter and version 14's baselines;
  card 067: closeness by local propagation (value iteration on the
  believed map) in place of route and chain tables.
- **Later:** the key–door comparison (063); rung 1's clearing commitment;
  the both world's random steps under the small view (065); card 055.
