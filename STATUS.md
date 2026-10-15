# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-14, `master`, nothing committed (the archive move is staged). **Rung:** 1, not passed.
  **Architecture:** version 27 (nothing adopted). **Main world: Craftium** (CHARTER "Current
  world"). Crafter is a development world; brain parts may be built there while Craftium lacks
  their inputs, each card naming its Craftium follow-up.
- **Objective (the user, this week):** integrated learning: one transition model over vector
  states for every action and the world's own changes, conditions read from it, perception
  parts feeding it.
- **Built so far:** series 105's one transition model over tokens (105.6, 105.8), the view placed
  by the pose (105.10); conditions read by intervention where two copies agree (105.4, 106).
  [108](experiments/108-learned-belief/card.md) stopped: a prediction of what a place will hold
  is not identity. Craftium runs under the caps; 107 (depth) stopped.
- **Direction (the user, 2026-10-14):** GOAL P12 replaced: conditions evaluated from state and
  memory, no forecast of when. CHARTER: System 1 (one step, always on) and System 2 (statements,
  evaluated selectively). Telling look-alikes apart deferred.
- **Roadmap (CHARTER, new):** the model's trajectory in 7 steps; next cards come from the first
  step not built. Steps 1–2 built in Crafter; step 3, belief over conditions, in progress.
- **[109](experiments/109-conditions-from-memory/card.md) fail → revise** (conditions from memory,
  world 6 "searched caves"). A network over the remembered places around a queried place, trained
  only to predict what is seen when a place comes into view, beat recall (−0.317 against −0.415
  nats) and both baselines (+0.63 over static memory). Choice right 99% (gems) and 95%
  (berries); in paired histories the choice followed memory in 98% and 100%. Missed the bound
  by 0.21 nats against 0.15 allowed: never-seen cave ends asked about from afar (0.42 near, 0.11
  far, against 0.5), because a cave's end is only ever first seen just after its mouth.
- **Decisions for the user:**
  1. **Card 109.1 (not drafted):** build training targets also from memory as it stood k steps
     before the sighting, and name only places the agent could know of (109's bound knew where
     unseen caves were). Shall I draft it?
  2. **The ladder:** rung 1 still tests "how close a goal is" (steps to the goal), which P12 no
     longer asks. Proposed: rung 1 "know the conditions of effects and goals" (scored as 105.4
     and 106 score them), rung 2 "evaluate them from state and memory" (109).
  3. **GOAL's other "how soon" lines:** the opening paragraph ("infer proximity"), C6, P19 and
     the suggested order still say it.
  4. Earlier: 107's tuned head; GOAL C1 and depth; 105.5 in Crafter; 106's next filter.
- **Docs:** cards 001–101 moved to `experiments/archive/`. LESSONS 1,992 words (two view-placing lessons merged to make room for 108's); the
  paper's abstract overclaims (open). Every run is capped (CHARTER rule 5). Craftium's
  environments: `tools/card107/setup.sh`.
