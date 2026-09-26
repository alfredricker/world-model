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
- **Blocked on:** the code port (`migration/run_port.sh`, awaiting user
  go-ahead); `src/` is empty. Then the goal-conditioned fork evaluator and
  its tests, then the throughput profile that fixes 001's budget.
- **Deferred:** Mamba as the recurrent core (rung 2, one-component card);
  training on imagined rollouts (later rungs).
- **Literature:** Saulus and Gentner not in papi; "Jev" unknown, needs a
  link. papi LLM summaries fail without a Gemini key.
- **Lessons:** delete `migration/lessons/` once the user has skimmed
  LESSONS.md.
