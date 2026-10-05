# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-05, on `master` (not pushed). **Rung:** before
  rung 1. **Architecture:** version 10 (card 051).
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions (P21).
  User, 2026-10-04: change the encoder by replacing pieces, not adding
  terms; tint set aside. User, 2026-10-05: no learned codebook (identity
  up to noise); perception keeps every visible difference, relevance is
  recall's and the planner's; gates on the agent's behaviour.
- **Overnight objective (the user, 2026-10-05):** an encoder that is
  well principled and scalable and keeps the goal/condition hierarchy;
  the agent still discovers conditions. Then goals from example frames
  (rung 1), then an egocentric view smaller than the map. Rules in
  AGENTS.md ("Overnight sessions"); record in `overnight-10-05.md`.
- **Active card:** [054](experiments/054-identity-up-to-noise/card.md),
  approved: identity up to noise, and an encoder that keeps every
  visible difference. Then
  [055](experiments/055-network-as-recall-prior/card.md) (the network
  as recall's prior), draft.
- **Recent cards:**
  - [053](experiments/053-recall-on-transition-encoder/card.md): fail,
    revise. The visibility margin made the encoder see what actions
    change (planner 100% in the familiar worlds); its codebook merges
    key colours (recall 45% / 85%);
  - [052](experiments/052-staged-encoder/card.md): fail, revise. Kept
    the generator with play starts. Corrected 2026-10-05: step T's
    encoder erased what actions change ("40 of 40" was scored by its own
    codes; 0 of 40 against the simulator);
  - [051](experiments/051-planner-chains-and-rules/card.md): pass, keep
    (version 10).
- **Open:** colour spread over parts; codes drifting once the encoder
  learns while acting (R4); cluttered seed 403 layout 93; "unknown" read
  as "fails"; conjunctions spliced.
- **papi:** GLM-5.3 via OpenRouter; needs credit for summaries.
- **Pinned:** card 029's arm 3 in the both world; demonstrations once
  random play is too thin; GOAL.md hypotheses draft.
