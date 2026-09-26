# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-25
- **Direction change this session (user-approved):** predictions are about
  goal conditions and how actions change how soon they can be reached, not
  fixed windows. GOAL.md: C6 and P12 rewritten, P20 (self-set goals) added,
  C1 allows supplied goals until P20, suggested order updated.
- **Rung:** before rung 1. Rung 1 is now the goal-conditioned fork: rank
  the 5 actions by how they change true steps-to-goal (breadth-first search
  over the simulator, evaluator only), scored on interaction-decisive forks
  above a goal-swapped control.
- **World:** chained rooms (CHARTER.md).
- **Active card:** 001, architecture selection (draft, awaiting user
  review). Hub H (encoder, transition T, quasimetric reachability head d)
  and four one-part variants: A + reconstruction, B contrastive T,
  D factored state with partial goals, E no transition. Optional round 2
  of at most three combinations (CHARTER).
- **Planned:** 002, label-free null-margin sampler on the winner, with the
  event-count curve.
- **Port:** done 2026-09-26; 9 tests pass; hashes match the old repo.
- **Evaluator:** `rooms_goals.py` built, 16 tests pass. Random-walk probes
  gave only 8 distinct unlock-decisive states, so fork states are now drawn
  from each world's full reachable set, stratified (`runs/sampled_dev`:
  300 per scored stratum over 120 worlds). Next: the feasibility gate
  (hub on simulator state), then the training code and throughput profile.
- **Deferred:** Mamba as the recurrent core (rung 2, one-component card);
  training on imagined rollouts (later rungs).
- **Literature:** reading gap closed; QRL is the hub's structural source,
  expectile preferred per OGBench; B's loss attribution fixed. Saulus and
  Gentner not in papi; "Jev" needs a link. papi summaries need a Gemini key.
- **Lessons:** delete `migration/lessons/` once the user has skimmed
  LESSONS.md.
