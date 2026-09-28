# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-27 (evening). **Rung:** before rung 1. **Direction
  (agreed):** theory first, condition discovery until solid.
- **Cards 003–004 (keep)** exact checks; **005 (keep)** learning from
  frames; **006 (revise)**; **007 (pass)**; **008 (revise)**; **009**
  shelved; **010 (pass, keep)** threshold-free condition definition;
  **011** skipped.
- **Card 012 (done, partial, revise):** discovery from one signal works in
  the key world (all expected conditions with the right meanings; acting
  on them with exact walking 95%); acting all learned 30.2% (random 0.4%).
  Fixes kept in `discover_logic.py`: way values trained through the
  encoder (own walking batch, cross-entropy), walk discount 0.8, value
  batch half from condition-on frames, final phase, play starts (run 5,
  `runs/012run5.sh`). Run 6 (condition detectors) was worse and reverted.
  Exact gates pass in all four worlds; switch/either/both not run with
  the fixes. Diagnostics: `tools/card012/`.
- **Card 013 (done, fail at gate, stop):** walking as a chain of short
  place conditions: 80% of moves closer with exact place conditions (gate
  90%), 84% learned, flat value alone 86%. Place conditions learnable (AUC
  0.95–0.998). Cause (discussed with the user): the value network flattens
  the spatial map, so distances are learned layout by layout.
- **Card 014 (draft, for the user's approval):** local-step walking: the
  same recursion computed by one learned 3 × 3 step on the encoder's map
  (value iteration network), agent found by a learned readout, attached to
  012 run 5's network. Gate: supervised upper bound ≥ 99%, readout finds
  the agent ≥ 99%. Criteria: walking ≥ 98%, unseen wall column ≥ 95%,
  acting ≥ 90%.
- **Pinned (user):** merging duplicate detectors (needs literature);
  demonstrations once random play is too thin; learning the walking skill
  (013, 014).
- **Housekeeping:** LESSONS.md over two pages (~210 lines), needs a merge
  pass. LITERATURE.md focus set to 014 (VIN, fixed-horizon TD, HIQL, SoRB;
  VIN and fixed-horizon TD added to papi). Card 013 closure and card 014
  draft uncommitted (012 committed in eb84d4d).
