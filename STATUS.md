# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-30. **Rung:** before rung 1. **Architecture:** version 5
  (card 029): a counted model of every action's effects (028), with
  subgoals worked backward through its learned rules at every step.
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-29: piled-up fixes signal a weak component;
  planning by recall is right.
- **[Card 036](experiments/036-fresh-codes/card.md) (pass; keep):** a new
  piece gets a fresh code. All 28 tiles were named apart in 8 of 10 seeds,
  against 1 of 10 with one "new" marker.
- **[Card 037](experiments/037-recall-in-the-planner/card.md) (fail;
  stop):** the planner's entries built from recall-weighted tries.
  - Nothing lost: 9 of 10 seeds (counting on the same codes: 6).
  - New keys and open doors predicted: 2 of 10 (8 needed; counting: 0).
  - Acted: 8 of 10 (counting: 2).
  - Picking up and dropping a new key now transfer (8 of 9 seeds per
    colour). Forward onto the new open door does not (3 of 9), though
    recall's vote was right in 9 of 9. The agent in a new doorway usually
    has a code of its own where it differs from the door, and an outcome
    stated from familiar codes cannot produce one. The same holds for closing a new
    door (never predicted) and for world (b) (5.6% of layouts).
  - The fitted metric is needed. With λ at its start, no finished seed's
    model fitted in the planner's 254 tuples, so the arm was stopped at
    25 minutes. The command to finish it is in the card.
- **[Card 038](experiments/038-planning-on-vectors/card.md) (approved
  2026-09-30; built, shakedown failed, awaiting the user):** the planner
  holds encoder vectors; recall predicts each effect. With oracle vectors
  (seed 399, 40 layouts per world), held-out effects are right in
  98–100%, but the goal is reached in 0–5%. The "like" test (k ≥ 1/2 under
  the action's λ) cannot tell holding nothing from holding the green key
  (k = 0.998), or switch on from off (0.98). λ is fitted only on which
  places change. Arm A shows the same (0.96–1.00). Decision needed before
  Appendix A changes.
- **papi:** now set to GLM-5.3 through OpenRouter; the account needs
  credit before summaries can be regenerated.
- **Pinned:** card 029's arm 3 in the both world (command in the card);
  demonstrations once random play is too thin; GOAL.md hypotheses draft.
