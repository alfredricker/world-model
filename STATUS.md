# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-01. **Rung:** before rung 1. **Architecture:** version 7
  (cards 038–044, kept with card 044): planning on the encoder's vectors,
  recall in two levels, conditions checked only in situations the learned
  effects produce, and the state as tokens (a what and a where each; moves
  as fitted transformations; no pose table). Seeds 400–404: all four
  familiar worlds 100%, effects exact, card 029's steps.
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions, not
  imagined action sequences (GOAL.md P21). User, 2026-10-01: codes for
  identity, vectors for similarity; fixes must be general principles;
  tokenize everything (tiles, the held thing), optimize recall later.
- **Next decision:** approve [card 045](experiments/045-movement-through-conditions/card.md)
  (draft, asked for by the user): movement through conditions. System 1
  approach (a learned how-soon over placements); System 2 over its
  conditions (the tiles on its route walkable; doors as conditions,
  waypoints otherwise); no imagined step checked by recall; tested on
  unseen rooms. If approved, it replaces card 041 and the obstacles card.
- **Recent cards:**
  - [044](experiments/044-state-as-tokens/card.md): pass, keep;
  - [043](experiments/043-consistent-hypotheticals/card.md): pass, keep;
  - [042](experiments/042-codes-in-recall/card.md): fail, revise (the
    either world; the cause was in the planner);
  - [041](experiments/041-where-and-how-soon/card.md): approved, revised
    to how-soon only; replaced by 045 if 045 is approved;
  - [040](experiments/040-relations-between-things/card.md): stopped;
    relations wait until colour transfer resumes (paused by the user);
  - [039](experiments/039-view-as-a-set/card.md): approved, superseded in
    use by 042–043; 038: decision pending.
- **Open in version 7:** the same-thing level reads all four codebooks,
  not only those each rule needs; conjunctions are still checked in
  spliced situations; no relations; walking imagines steps (card 045);
  0.31–1.09 s per layout (version 5: 0.015–0.024).
- **papi:** GLM-5.3 via OpenRouter; needs credit to regenerate summaries.
- **Pinned:** card 029's arm 3 in the both world; demonstrations once
  random play is too thin; GOAL.md hypotheses draft.
