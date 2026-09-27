# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-26
- **Rung:** before rung 1. **Direction (agreed with the user):** theory
  first, focused on condition discovery until it is solid. Theory in
  [card 003](experiments/003-conditions-theory-check/card.md) section 2:
  a condition is what a goal's achieving action needs, found by
  contrasting successes with failures; each condition becomes a subgoal.
- **Cards 003–004 (keep):** exact checks in an 8×8 key-door room;
  walking to X as an action with conditions gives key → door → goal.
- **Card 005 (keep):** achievement and walk values learned from frames
  pass on new layouts; needs ~300 unlocks (exact counting: 10).
- **Card 006 (fail, revise):** conditions can't be read off the internal
  state afterwards; a separable state isn't needed for discovery.
- **Card 007 (pass, keep):** from pixels and one supplied goal (goal
  square), each condition defined as "the states where this goal's
  achieving action works are within walking reach". Discovered door open
  (0.996) → holding the matching key (1.0) → empty hands (1.0; nothing
  achieves it, so it is kept). Each behaves as a condition (100% / 0%);
  acting on them solves 99.2% of new layouts vs 0% for the goal alone.
  Caveat: card 005's encoder was also trained with other supplied goals.
- **Card 008 (fail, revise):** encoder trained on the goal-square signal
  only. Frozen: the key is barely in its state (read-out 0.73), discovery
  stops at door open; acting 0%. Continual (the encoder keeps learning
  from its own discovered goals): key read-out rises to 0.98, chain door
  open → matching key → empty hands found and behaves as conditions; but
  far-from-target walk values flicker around the 0.1 threshold, making a
  spurious level 4 and acting 75.8% (< 90%).
- **Card 009:** shelved (undoability is the wrong definition of a
  condition). **Next:** card 010 (draft, for the user): threshold-free
  condition definition (or of ands, admitted by Bayesian evidence) tested
  on five worlds (key, switch, either, both, no drop; irrelevant vase),
  exact then learned from pixels. Card 011: detectors without vocabulary.
- **Housekeeping:** LESSONS.md is over two pages (~180 lines), needs a
  merge pass. Uncommitted: cards 003–007 code (`envs/keydoor*.py`,
  `conditions.py`, `reach.py`, `learn_keydoor.py`, `latent_conditions.py`,
  `discover.py`, `tests/test_keydoor.py`) and today's document edits.
  Card 001 abandoned, 002 on hold. `bin/prun` skips 32-bit libraries.
