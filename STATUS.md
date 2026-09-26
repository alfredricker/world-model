# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-26
- **Reframing this session (user-approved):** GOAL.md's main insight (how
  close a goal is → which conditions it needs → subgoals → deliberation
  over conditions). CHARTER ladder reordered to follow it; cards 001 and 002
  reframed; LITERATURE updated. Runs may exceed 10 minutes when that is
  what an informative result needs (user); over 30 minutes go to the user.
- **Rung:** before rung 1. **World:** chained rooms.
- **Active card:** 001, architecture selection (draft). Built this session:
  training collection `runs/train` (hash matches the old repo), simulator
  states for it, fork extras `runs/sampled_dev_extras` (symbolic forks,
  exact goals, true successors, goal pools, 651 matched condition pairs),
  hub H (`src/worldmodel/models/hub.py`, 1.70M parameters), trainer
  (`train.py`), harness (`fork_eval.py`); 39 tests pass.
- **Feasibility gate: not passed**, second round (card 001 section 5).
  User decisions: the upper bound is the egocentric simulator view; goals
  are supplied conditions (4 examples + success signal from labels, C1
  until P20), label-free discovery of conditions deferred. Best so far:
  pooled top-1 0.62–0.65 (swapped 0.42–0.44): one-step conditions learned
  (key pickup 0.88), two-step chains at the swapped level, movement ≤ 0.55
  even on true successors. Bar is 0.9.
- **Throughput:** frames 146 updates/s, egocentric ~113 with condition
  goals; 30k updates ≈ 4.5 min.
- **Next decision (user):** how to separate the head's precision from
  learning from random data: fit the head to true distances (evaluator
  search over training worlds) as a ceiling; then chains (key → door rests
  on 307 unlocks).
- **Planned:** 002, label-free sampler (arm C one-way priority; frames show
  a one-way gap of 5.9 at box opens, 2.5 pickups, 1.6 moves, 0.1 toggles).
- **Literature:** added RUDDER, Align-RUDDER, SoRB, predicate invention,
  prioritised replay. McGovern & Barto, Saulus, Gentner need PDFs.
- **Tooling:** `bin/prun` now skips 32-bit libraries.
