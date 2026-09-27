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
- **Card 013 (draft, needs the user's approval):** walking in the
  recursion. Each way's walk becomes a chain of place conditions ("within
  4 steps of a ready state", "within 8", ...), found by the same
  recursion; the agent only walks ≤ 4 steps toward the next link. The
  user asked for walking to become part of the recursion; the card's
  design, gate and criteria are for their review.
  - **Next:** user reviews 013; then the bench gate (~10 min); main runs
    (~30 min per world) handed to the user.
- **Pinned (user):** merging duplicate detectors (needs literature);
  demonstrations once random play is too thin; learning the walking skill
  (013 is a first step).
- **Housekeeping:** LESSONS.md over two pages (~210 lines), needs a merge
  pass. LITERATURE.md "Current focus" still names cards 001–002; card 013
  would use HIQL, SoRB and De Asis et al. 2020 (fixed-horizon TD, not yet
  in papi). Card 012 work since fdc46ff is uncommitted.
