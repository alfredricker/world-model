# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-04 (branch `overnight-051-052`, not merged).
  **Rung:** before rung 1. **Architecture:** version 10 (card 051,
  kept by the user today): memory indexed by situation, threats and
  commitment in the planner, routes through two tokens, the hand kept.
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions (GOAL.md
  P21). User, 2026-10-01: codes for identity, vectors for similarity;
  fixes must be general principles. User, 2026-10-03: near-linear cost
  before BabyAI or Crafter. User, 2026-10-04: change the encoder by
  replacing pieces that do not work, not by adding terms.
- **Active card:** [052](experiments/052-staged-encoder/card.md), the
  staged encoder, infant stage (gate 2), extended step by step with the
  user's approval (steps 2b–2h; log.md is the record):
  - drift is judged by recall's predictions, not raw codes (2c);
  - play starts in the stream; their held key was lost before (2d);
  - the old recipe ignores colour even on three clean colours (2f);
  - uniformity with a fixed codebook replaces the pixel anchor (VISReg
    tried and dropped: its shape term killed the network); the relation
    term was removed; either the effect term or error-driven
    differentiation then separates key colours (2g: 97.5% / 100%);
  - running: 2h, both on nine colours with tint and noise.
- **Next decision (the user):** which interaction term; then bring the
  planner and acting into card 052 (the agent's recall and conditions
  shaping the encoder, step 4; the new encoder driving the agent).
- **Recent cards:**
  - [052](experiments/052-staged-encoder/card.md): draft, gate 1 pass,
    gate 2 in progress;
  - [051](experiments/051-planner-chains-and-rules/card.md): pass, keep
    (version 10); compiled rules, cost pass, "unknown" left for later;
  - [050](experiments/050-own-tries-first/card.md): pass, keep (version 9).
- **Open:** cluttered seed 403 layout 93; "unknown" read as "fails";
  parts mix kind and colour; conjunctions spliced; re-admission on large
  memory slow (only on surprise); new appearances with a fixed codebook.
- **papi:** GLM-5.3 via OpenRouter; needs credit for summaries.
- **Pinned:** card 029's arm 3 in the both world; demonstrations once
  random play is too thin; GOAL.md hypotheses draft.
