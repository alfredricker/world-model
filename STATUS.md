# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-04, on `master` (the overnight branch merged, not
  pushed). **Rung:** before rung 1. **Architecture:** version 10 (card
  051): memory indexed by situation, threats and commitment in the
  planner, routes through two tokens, the hand kept.
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions (GOAL.md
  P21). User, 2026-10-01: codes for identity, vectors for similarity.
  User, 2026-10-04: change the encoder by replacing pieces that do not
  work, not by adding terms; tint set aside (the agent never acts on
  it); the encoder learns latent transitions together with conditions.
- **Active card:** [053](experiments/053-recall-on-transition-encoder/card.md),
  draft: version 10's recall and planner on an encoder trained by
  transitions. Step 1 (the R1 checks): gates and code restarts
  calibrated, the encoder under noise. Step 2 (R2): does recall's
  front–held relation read the key–door match, and does the planner
  still work.
- **Next decision (the user):** approve card 053.
- **Recent cards:**
  - [052](experiments/052-staged-encoder/card.md): done, fail, revise.
    Kept: the generator with play starts, recall's predictions as the
    drift measure, the transition objective (its model predicts both
    colour cases 40 of 40; the stand-in recall cannot read the match);
  - [051](experiments/051-planner-chains-and-rules/card.md): pass, keep
    (version 10);
  - [050](experiments/050-own-tries-first/card.md): pass, keep (version 9).
- **Open:** colour spread over parts (version 10's relation is per
  part); code relabelling once the encoder learns while acting (R4);
  memory keyed by exact pixel repeats; cluttered seed 403 layout 93;
  "unknown" read as "fails"; conjunctions spliced.
- **papi:** GLM-5.3 via OpenRouter; needs credit for summaries.
- **Pinned:** card 029's arm 3 in the both world; demonstrations once
  random play is too thin; GOAL.md hypotheses draft.
