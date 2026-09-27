# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-26
- **Rung:** before rung 1. **Direction (agreed with the user today):**
  theory first. The theory of conditions is written up in
  [card 003](experiments/003-conditions-theory-check/card.md), section 2: a
  condition is what the action that achieves a goal needs (the door opens
  on a toggle only when facing it *and* holding the matching key), found by
  contrasting successes with failed attempts; each condition becomes a
  subgoal, down to movement. CHARTER has a "Theory first" paragraph.
- **Card 003 (done, keep):** conditions theory check in an 8×8 key-door
  room, random play, exact computation with simulator variables. Finding
  conditions at the achieving step recovered every goal's exact conditions
  (unlock: facing the door + matching key; key: facing it + empty hands;
  doorway: facing the door + door open) from 10 successes; the distractor
  key never works. Long-range jumps are dominated by the final approach.
  Amendment: the chain goal square → door runs through positional
  conditions (in the right room, in the doorway) that we named by hand.
- **Card 001** (architecture selection, chained rooms): abandoned,
  "revise"; its code and evaluator data stay for the harder test later.
- **Card 002:** on hold; depends on card 001's winner.
- **Card 004 (done, keep):** walking to X as an action with conditions.
  Found: walk to goal square needs "door open" (0.996); walking to door or
  keys needs nothing. Chaining the found rules gave key → pickup → door →
  toggle → goal in 500/500 new layouts, 1.006 × the shortest solution, no
  place named by hand and the distractor never picked.
- **Next (user to choose):** learn achievement and walk-reach probabilities
  from frames (compare with the exact values), or first a harder exact world
  (key behind another door). CHARTER's rung 1 and current world are then
  redefined from cards 003–004.
- **Housekeeping:** LESSONS.md is over two pages (157 lines) and needs a
  merge pass. Uncommitted: cards 003–004 (`envs/keydoor.py`, `conditions.py`,
  `reach.py`, `tests/test_keydoor.py`; 41 tests pass) and today's document
  edits.
- **Tooling:** `bin/prun` skips 32-bit libraries.
