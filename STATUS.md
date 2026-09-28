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
- **Card 014 (done, fail, revise):** local-step walking on the map works
  (99.97% supervised at every cell; 96–98% at the agent only), but
  learning from experience in a top-down frame fails (60–88%, acting 25%):
  the learned attention finds the agent in 17.5%.
- **Card 016 (approved, running):** walking on an egocentric view (whole
  room, agent at the centre, fixed readout). Gate (every cell supervised,
  30k): 99.8% / 97.7%, door way under 99%, loss still falling. Main run
  (learned from experience, 30k): walking 73–97%, unseen column 83–95%,
  acting 48.4% (top-down 25%, run 5 30.2%). 90k updates, 24 look-ahead
  steps: walking 61–98%, unseen column 69–96%, acting 48.4% again; more
  training does not help. All criteria fail. Maps check: "ready here" is
  right only at the agent (chance elsewhere), so the recurrence does not do
  the walking; suspect the whole-frame summary. Decision pending the user.
- **Card 015 (done, fail):** reconstruction or change prediction make the
  map show objects (most classes 97–100%) but did not raise walking at 8k
  updates. Run 5's encoder at 30k reaches 99.07% / 99.83% without them,
  so the new map is not needed for walking. Decision: stop.
- **Pinned (user):** merging duplicate detectors (needs literature);
  demonstrations once random play is too thin.
- **Housekeeping:** LESSONS.md over two pages (~210 lines), needs a merge
  pass. Cards 014 (gate notes) and 015 uncommitted (last commit 6767e26).
