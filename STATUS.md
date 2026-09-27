# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-26
- **Rung:** before rung 1. **Direction (agreed with the user):** theory
  first, focused on condition discovery until it is solid. Theory in
  [card 003](experiments/003-conditions-theory-check/card.md) section 2:
  a condition is what a goal's achieving action needs, found by
  contrasting successes with failures; each condition becomes a subgoal.
- **Card 003 (keep):** exact check in an 8×8 key-door room; conditions
  found at the achieving step from 10 successes.
- **Card 004 (keep):** walking to X is an action with conditions; the
  chain key → door → goal appears without naming places; 500/500 layouts.
- **Card 005 (keep):** achievement and walk values learned from full-view
  frames pass on new layouts; needs ~300 unlocks (exact counting: 10).
- **Card 006 (fail, revise):** conditions cannot be read off the network's
  internal state afterwards (held redundantly); separable internal state
  is not needed for discovery.
- **Card 007 (pass, keep):** from pixels and one supplied goal (goal
  square), each condition defined as "the states where this goal's
  achieving action works are within walking reach". Discovered door open
  (0.996) → holding the matching key (1.0) → empty hands (1.0; nothing
  achieves it, so it is kept). Each behaves as a condition (100% / 0%);
  acting on them solves 99.2% of new layouts vs 0% for the goal alone.
  Caveat: card 005's encoder was also trained with other supplied goals.
- **Next:** card 008, the same discovery on an encoder trained with only
  the goal-square signal (or none), to remove that caveat. Then a world
  where two conditions are needed at once (chained rooms).
- **Earlier:** card 001 (chained rooms screen) abandoned, code kept; card
  002 (rare-event sampling) on hold, a candidate fix for the ~300-unlock
  need.
- **Housekeeping:** LESSONS.md is over two pages (~180 lines), needs a
  merge pass. Uncommitted: cards 003–007 code (`envs/keydoor*.py`,
  `conditions.py`, `reach.py`, `learn_keydoor.py`, `latent_conditions.py`,
  `discover.py`, `tests/test_keydoor.py`) and today's document edits.
  **Tooling:** `bin/prun` skips 32-bit libraries.
