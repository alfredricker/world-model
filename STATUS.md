# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-10-18, `master`, uncommitted: ladder, cards 110–112, 111.3. **Rung:** 1 (new test), not passed.
  **Architecture:** version 27 (nothing adopted). **Main world: Craftium** (CHARTER "Current
  world"). Crafter is a development world; brain parts may be built there while Craftium lacks
  their inputs, each card naming its Craftium follow-up.
- **Objective (the user, this week):** integrated learning: one transition model over vector
  states for every action and the world's own changes, conditions read from it, perception
  parts feeding it.
- **Built so far:** one transition model over tokens (105.6, 105.8), the view placed by the pose
  (105.10); conditions by contrast over memory (111). 107 (depth) and 108 (belief) stopped.
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
  2. **GOAL's other "how soon" lines:** the opening paragraph ("infer proximity"), C6, P19 and
     the suggested order still say it.
  3. **[110](experiments/110-confidence/card.md) stop** (P5): familiarity on System 1's vectors misses
     what the model never learned; filtering unfamiliar states drops true conditions. Earlier: 107; C1.
  4. **[112](experiments/112-experiment-on-open-answers/card.md) stop** (P15): trying where memory is
     open beat random play by only +0.011. **[111.3](experiments/111.3-closing-by-co-changes/card.md)
     keep:** an answer stays open while a fact a co-occurring change needs went untested; closed
     answers complete 1.0 (0.54 before, large memory), one doubt each. Next, choose: (1) conditions
     through co-changes (lamps lighting recall 0.65 → 1.0 in the diagnostic); (2) set up doubted
     facts (rung 4); (3) Craftium.
- **Docs:** cards 001–101 in `experiments/archive/`. LESSONS 2,220 words. The paper's abstract
  overclaims (open). Every run capped (rule 5). Craftium: `tools/card107/setup.sh`.
