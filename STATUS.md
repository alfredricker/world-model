# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-27 (night). **Rung:** before rung 1. **Direction
  (agreed):** theory first, condition discovery until solid.
- **Cards 003–004 (keep)** exact checks; **005 (keep)** learning from
  frames; **006 (revise)**; **007 (pass)** discovery from one goal with a
  richly trained encoder; **008 (revise)** continual learning needed,
  walk-value flicker; **009** shelved; **010 (pass, keep)** threshold-free
  condition definition, exact and from pixels, 19/19.
- **Card 011** skipped with the user. **Card 012 (approved, running,
  unfinished):** discovery end to end from the goal-square signal in the
  key / switch / either / both worlds. Read the card's section 7 first.
  - Speed: compiled training step, parallel data, compact storage →
    learned part ~5–6 min per world (exact gate is the slow part).
  - Exact gate passes (key, switch); exact acting 100%.
  - Run 1 (`runs/012run1_*`): fail, reach values creep up → fake
    movement "ways"; acting ~0%.
  - Run 2 (`runs/012_*`, `runs/012.sh`): double values + walking
    invariance + more data. Key: acting 1%; deeper conditions unstable.
    Switch/either/both were still running in the background when the
    session ended; check `runs/012_{switch,either,both}/result.json`
    and `runs/012_*.out` (they may have finished or errored).
  - **Next decision:** test fixed-horizon reach (De Asis et al. 2020)
    on the bench (`bench_values.py` pattern: learned reach vs exact on
    switch data; bench numbers so far: plain 74% right, double 98% but
    recall 77%). If clearly better, run 3 of card 012; then write
    keep/revise/stop.
- **Pinned (user):** merging duplicate detectors (needs literature);
  demonstrations once random play is too thin; learning the walking
  skill (later card).
- **Housekeeping:** LESSONS.md over two pages (~200 lines), needs a merge
  pass. Nothing committed since card 010's proposal: cards 010–012 code
  and docs are uncommitted (`logic_conditions.py`, `learn_logic.py`,
  `discover_logic.py`, `tests/test_logicdoor.py`, `runs/012.sh`).
  Card 010's `Data` now takes a `keys` argument (default unchanged).
