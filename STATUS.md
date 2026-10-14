# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-14, `master`, nothing committed. **Rung:** 1, not passed.
  **Architecture:** version 27 (nothing adopted). **Main world: Craftium** (CHARTER "Current
  world"). Crafter is a development world; brain parts may be built there while Craftium lacks
  their inputs, each card naming its Craftium follow-up.
- **Objective (the user, this week):** integrated learning: one transition model over vector
  states for every action and the world's own changes, conditions read from it, perception
  parts feeding it.
- **Series 105, what holds:** one transformer over tokens with a bounded log-spaced position
  bias (105.6), trained on the next code's likelihood (105.8), with the view placed by the pose
  (105.10, kept). Conditions are read by intervention (105.4), in sleep only where two trained
  copies agree (106, revise).
- **Craftium:** runs under the caps in lockstep at about 58 steps per second; 107 (depth by view
  synthesis) stopped: terrain depth fixed, cows lost in it.
- **[108](experiments/108-learned-belief/card.md) stop** (learned belief). The k-step query cost
  one-step changes (do's changes −15.7 and −2.8 points); returns 83.8% against counted motion's
  87.9%. Finding: a prediction of what a place will hold is not identity.
- **Direction changed (the user, 2026-10-14):** GOAL P12 replaced: the agent knows the
  conditions a goal needs and evaluates them from state and memory, with no forecast of when.
  CHARTER: System 1 (the one-step model, always on) and System 2 (statements: effect,
  conditions, spoilers; evaluated selectively from memory). Telling look-alikes apart deferred.
- **Active card: [109](experiments/109-conditions-from-memory/card.md), conditions from memory
  (draft).** A breaking world of searched caves: gems never come back, berries regrow; belief
  from sight, else recall of revisits, else recall of first sightings. About 12 minutes.
- **Decisions for the user:**
  1. **Approve 109.**
  2. **The ladder:** rung 1 still tests "how close a goal is" (steps to the goal), which P12 no
     longer asks. Proposed: rung 1 "know the conditions of effects and goals" (scored as 105.4
     and 106 score them), rung 2 "evaluate them from state and memory" (109).
  3. **GOAL's other "how soon" lines:** the opening paragraph ("infer proximity"), C6, P19 and
     the suggested order still say it.
  4. **Card folders:** 167 at the top of `experiments/`; a proposal to archive 001-101.
  5. Earlier: 107's tuned head; GOAL C1 and depth; 105.5 in Crafter; 106's next filter.
- **Docs:** LESSONS 1,992 words (two view-placing lessons merged to make room for 108's); the
  paper's abstract overclaims (open). Every run is capped (CHARTER rule 5). Craftium's
  environments: `tools/card107/setup.sh`.
