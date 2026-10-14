# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-13 (evening), `master`, nothing committed. **Rung:** 1, not passed.
  **Architecture:** version 27 (nothing adopted). **Main world: Craftium** (the user, 2026-10-13;
  CHARTER "Current world"). Crafter is a development world; brain parts may be built there
  while Craftium lacks their inputs, each card naming its Craftium follow-up.
- **Objective (the user, this week):** integrated learning: one transition model over vector
  states for every action and the world's own changes, conditions read from it, perception
  parts feeding it. Night log: [overnight-10-13.md](overnight-10-13.md).
- **Series 105, what holds:** one transformer over tokens with a bounded log-spaced position
  bias (105.6), trained on the next code's likelihood (105.8), with the view placed by the pose
  (105.10, kept by the user). Conditions are read by intervention (105.4), and in sleep only
  where two trained copies agree (106, revise: Crafter makes 0.99; world 1's impossible lamps stay).
- **New today: Craftium** (Luanti/VoxeLibre, a 3D Minecraft-like world) as the step between
  Crafter and robotics. Smoke tests, no card (setup in [card 107](experiments/107-depth-from-own-motion/card.md)):
  - it runs under the caps in lockstep at about 58 steps per second and reports the pose;
  - pretrained trackers keep a cow only while it stays in view (CoTracker3 78%, 13% after it
    returns); a belief with exact positions and the pose brings back every cow within 100 steps.
- **CHARTER amended (the user):** "Belief supplies 'which one'": new tokens are matched to
  remembered ones by the effect model's predictions; the pose still places the view; which
  tokens are kept and how they move unseen are learned. The user: long absences need not be
  tracked yet, and no declared motion priors.
- **107 stop** (depth by view synthesis with the known pose). Gate passed (the loss is lowest at
  true depth). Tuning fixed the terrain's scale (0.74 → 0.99; 74% of pixels within 25%), but
  cows are lost in the patch-scale depth (31% too near), so the belief stays at the no-memory
  level (returns 50%, bar 85%). The user: "it doesn't need to be perfect".
- **Active card: [108](experiments/108-learned-belief/card.md), learned belief (draft, awaiting
  approval).** Built on Crafter by the user's choice: creatures matched to files by the transition
  model's prediction at every gap k (1 in view, up to 50 unseen; a "k steps pass" input, no new
  output). Bars: returns 88% (102.3: 81.1%), IDF1 0.85 (098.6: 0.717). About 35 minutes, capped.
  Follow-up in Craftium: creature tokens from perception, then the port.
- **Decisions for the user:**
  1. **Approve 108** to run.
  2. **107's tuned head:** keep it as Craftium's terrain depth, or leave it.
  3. **GOAL C1** names outlines and correspondence for a pretrained front end, not depth.
  4. Earlier: 105.5 (planning, revise), 106's next filter. 105.3's history arms on hold.
- **Docs:** LESSONS 1,988 words (near three pages); the paper's abstract overclaims (open). Every
  run is capped (CHARTER rule 5). Craftium's environments: `tools/card107/setup.sh`.
