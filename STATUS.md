# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05 (overnight), on `master` (not pushed). **Rung:**
  rung 1 begun (cards 056, 059), not passed. **Architecture:** version
  14 (cards 057, 060, 061, 062, kept overnight: for the user to confirm).
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-10-05: no learned codebook (identity up to
  noise); perception keeps every visible difference; gates on behaviour.
- **Overnight objective (the user):** a principled, scalable encoder
  that keeps the goal/condition hierarchy, then goals from example
  frames, then a view smaller than the map. Card by card in
  [overnight-10-05.md](overnight-10-05.md).
- **Where it stands:**
  - [054](experiments/054-identity-up-to-noise/card.md) revise: the
    encoder without a codebook keeps the planner at 100% and the
    conditions on 4 seeds; recall's colour cases on noisy tries fall short.
  - [056](experiments/056-goals-from-example-frames/card.md) revise:
    goals inferred from frames are reached as well as written-in ones.
  - [057](experiments/057-view-smaller-than-the-map/card.md) keep (v11):
    a 7 × 7 view, 100% at 1.10–1.17× the full view's steps.
  - [058](experiments/058-route-or-clear-the-way/card.md) stop: pricing
    clearing every step made the agent dither.
  - [059](experiments/059-how-soon-from-the-chain/card.md) revise: how
    soon from the chain, rank 0.97–0.99 against true steps.
  - [060](experiments/060-walls-hide-what-is-behind/card.md) keep (v12):
    occlusion; look at the frontier, open a door to see past it.
  - [061](experiments/061-moves-learned-from-the-small-view/card.md) keep
    (v13): the move model learned from the 7 × 7 view.
  - [062](experiments/062-tries-stored-as-believed/card.md) keep (v14):
    stored tries record the believed view; conditions still discovered.
  - [063](experiments/063-replay-what-the-transition-model-misses/card.md)
    stop: prioritized replay did not teach "other key"; the comparison is.
  - [064](experiments/064-small-view-on-the-harder-worlds/card.md) keep:
    v14 on two doors 30/30 and the cluttered world 100/100.
- **Next decisions for the user:** confirm versions 11–14; whether card
  054's encoder replaces version 10's, and the next encoder step (card
  063: compare key and door by a learned comparison, not per part);
  rung 1: walking that keeps its choice of clearing the way, and goal
  examples where random play is too thin (the both world's door).
- **Open:** maps much larger than the view (routes are precomputed for a
  fixed lattice); card 055 (the network as recall's prior), draft.
- **papi:** needs OpenRouter credit for summaries. 9 of 10 cards used.
