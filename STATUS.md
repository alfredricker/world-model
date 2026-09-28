# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-27 (late). **Rung:** before rung 1. **Direction
  (agreed):** theory first, condition discovery until solid.
- **Cards 003–004 (keep)** exact checks; **005 (keep)** learning from
  frames; **006 (revise)**; **007 (pass)**; **008 (revise)**; **009**
  shelved; **010 (pass, keep)** threshold-free condition definition;
  **011** skipped.
- **Card 012 (done, partial, revise):** discovery works in the key world
  (right conditions; acting on them with exact walking 95%); acting all
  learned 30.2% (random 0.4%). Method in `discover_logic.py` (run 5,
  `runs/012run5.sh`). Switch/either/both worlds not rerun with the fixes.
- **Card 013 (done, fail at gate, stop):** walking as a chain of place
  conditions no better than a flat value (80–86%); the flat value network
  loses the spatial map.
- **Card 014 (gate passed):** walking by one learned 3 × 3
  step on the encoder's map, "within k steps" = "one step from within
  k − 1". Upper bound (every cell supervised, position given) passes:
  99.97% of moves closer on held-out frames at 30k updates (8k gave
  90–94%; training length was the limit). Readout check passes: the
  learned attention finds the agent in 99.3% / 99.55% (unseen wall column)
  of held-out frames. Gate passed. Next: criteria 1–3 without exact
  distance maps (`main_stage` not written).
- **Card 015 (done, fail):** reconstruction or change prediction make the
  map show objects (most classes 97–100%) but did not raise walking at 8k
  updates. Run 5's encoder at 30k reaches 99.07% / 99.83% without them,
  so the new map is not needed for walking. Decision section open.
- **Pinned (user):** merging duplicate detectors (needs literature);
  demonstrations once random play is too thin.
- **Housekeeping:** LESSONS.md over two pages (~210 lines), needs a merge
  pass. Cards 014 (gate notes) and 015 uncommitted (last commit 6767e26).
