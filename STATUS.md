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
- **Card 001** (chained rooms screen): abandoned, "revise"; code kept.
- **Card 002:** on hold. **Tooling:** `bin/prun` skips 32-bit libraries.
- **Card 004 (done, keep):** walking to X as an action with conditions.
  Found: walk to goal square needs "door open" (0.996); walking to door or
  keys needs nothing. Chaining the found rules gave key → pickup → door →
  toggle → goal in 500/500 new layouts, 1.006 × the shortest solution, no
  place named by hand and the distractor never picked.
- **Card 005 (running, approved):** learn achievement and walk values from
  full-view frames. Final network (10k episodes, 100k updates, 20 min)
  passes all three criteria on 500 new layouts: contrasts ≥ 99.6% right;
  walk ranking 0.94–0.98; conditions read back from its predictions match
  the truth. Changes that got there: FOIL-gain condition finder (same rules
  as before on exact data), more rooms, longer training. P19 runs (10, 30,
  100, 300 unlocks) in progress: `runs/005_p19.sh`, about 80 minutes.
- **Next:** P19 results, card 005 decision; then card 003's appendix step 2
  (conditions from a learned state, without supplied variables).
- **Housekeeping:** LESSONS.md is over two pages (157 lines) and needs a
  merge pass. Uncommitted: cards 003–004 (`envs/keydoor.py`, `conditions.py`,
  `reach.py`, `tests/test_keydoor.py`; 41 tests pass) and today's document
  edits.
