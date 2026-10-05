# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05 (overnight), on `master` (not pushed). **Rung:**
  rung 1 begun (cards 056, 059), not passed. **Architecture:** version
  11 (card 057, kept overnight: for the user to confirm).
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-10-05: no learned codebook (identity up to
  noise); perception keeps every visible difference; gates on behaviour.
- **Overnight objective (the user):** a principled, scalable encoder
  that keeps the goal/condition hierarchy, then goals from example
  frames, then a view smaller than the map. Card by card in
  [overnight-10-05.md](overnight-10-05.md).
- **Where it stands:**
  - [054](experiments/054-identity-up-to-noise/card.md) revise: the
    transition-trained encoder with identity up to noise keeps the
    planner at 100% (familiar, chained, cluttered; 4 seeds) and the
    conditions; recall's colour cases on noisy tries fall short.
  - [056](experiments/056-goals-from-example-frames/card.md) revise:
    goals inferred from frames are reached as well as written-in ones.
  - [057](experiments/057-view-smaller-than-the-map/card.md) keep: a
    7 × 7 view, 100% at 1.10–1.17× the full view's steps.
  - [058](experiments/058-route-or-clear-the-way/card.md) stop: pricing
    clearing every step made the agent dither.
  - [059](experiments/059-how-soon-from-the-chain/card.md) revise: how
    soon from the chain, rank 0.97–0.99 against true steps.
  - [060](experiments/060-walls-hide-what-is-behind/card.md): occlusion,
    running.
- **Next decisions for the user:** confirm version 11; whether card
  054's encoder replaces version 10's; rung 1's open pieces: walking
  that keeps its choice of clearing the way, and goal examples where
  random play is too thin (the both world's door).
- **Open:** recall's colour separation on noisy tries (054); the
  starting memory learned from full views (057's exception); card 055
  (the network as recall's prior), draft.
- **papi:** GLM-5.3 via OpenRouter; needs credit for summaries.
- **Pinned:** demonstrations once random play is too thin; GOAL.md
  hypotheses draft.
