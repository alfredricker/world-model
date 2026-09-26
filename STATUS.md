# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-26
- **Reframing this session (user-approved):** GOAL.md now states a main
  insight: the agent tells how close a goal is from its state, infers from
  reached goals which conditions made them possible, pursues unmet
  conditions as subgoals down to actions, and deliberates over chains of
  conditions (P21, which absorbed P13). P12 now holds the "how close" part;
  P19 counts a change in how reachable a goal is as a sign of importance.
- **Ladder (CHARTER.md) reordered:** 1 how close a goal is (fork, now with
  rank correlation as a second pass criterion); 2 infer a goal's conditions
  (new, subgoal-choice test); 3 act on subgoals; 4 deliberate over
  conditions; then memory, relations, composition, skills.
- **Rung:** before rung 1.
- **World:** chained rooms (CHARTER.md).
- **Active card:** 001, architecture selection (draft, awaiting user
  review). Hub H and four one-part variants: A + reconstruction,
  C subgoal head (HIQL; replaced contrastive B), D factored state with
  partial goals, E no transition. Two scored metrics: interaction top-1
  and rank correlation with true steps. New condition diagnostics:
  condition gap on matched state pairs, and one-way gap d(z', z) − d(z, z').
- **Planned:** 002, label-free sampler on the winner: uniform, null-margin,
  and one-way priority (run only if 001's one-way gap separates
  interactions that cannot be undone).
- **Port:** done; 9 tests pass. **Evaluator:** `rooms_goals.py`, 16 tests
  pass; fork states in `runs/sampled_dev`. Next: matched-pair builder for
  the condition gap, the feasibility gate (hub on simulator state), then
  training code and throughput profile.
- **Deferred:** Mamba as the recurrent core (memory rung); training on
  imagined rollouts (later rungs).
- **Literature:** added to papi: RUDDER, Align-RUDDER, SoRB, predicate
  invention, prioritised replay. McGovern & Barto 2001, Saulus and Gentner
  need PDFs. papi summaries need a Gemini key.
- **Lessons:** delete `migration/lessons/` once the user has skimmed
  LESSONS.md.
