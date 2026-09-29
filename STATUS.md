# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-28. **Rung:** before rung 1. **Architecture:** version 3
  in code (`src/worldmodel/spatial_state.py`); the direction below replaces
  its readiness and walking parts.
- **Direction (with the user, 2026-09-28,
  [card 022](experiments/022-effects-post-mortem/card.md)):** the primitive
  is the effect of an action on conditions, learned from the agent's own
  outcomes; moves are actions like any other, and walking must not read
  the simulator once conditions are learned. Readiness is removed.
- **Cards 023–026:** conditions as "approachable by moving closer"; card
  026 (keep) grows subgoals only when stuck, depth first with backtracking:
  100% in the key world, 16.4 steps, no random moves, 58 stored conditions
  (limit 32; one hard layout added 28; some look like duplicates).
- **Cards 016, 021 (learned):** learned walking 48.4%; architecture 3's
  conditions failed even with exact labels. Card 012's learned conditions:
  95% with exact walking, 30.2% all learned.
- **Theory session (with the user):** objects are what conditions are
  about; kinds are learned where the agent acts and recognised everywhere;
  relations fall out of conditions. Notes in card 027's appendix. GOAL.md
  hypotheses drafted, awaiting the user.
- **Card 027 (four runs; pass; keep):** kinds are decided by counting
  what actions do in front (pixels, no labels): the 8 expected kinds in
  both worlds; acting 100%, 1.01× the simulator's steps. Target rule:
  refuse moves that end the way's condition; mark actions that do not turn
  the parent true. Networks that discover kinds merge rare look-alikes;
  withheld appearances are placed by colour, not shape (4 of 33).
- **Card 028 (draft, awaiting the user's approval):** one model learned
  by counting from the agent's own pixels says what every action does,
  moves included (facts: tiles in view and the held tile; effects; rules
  for when an effect happens). Every condition, walking, target check and
  tree label is computed on its predictions; nothing reads the simulator.
  Literature focus refreshed for it. After it: transfer (new colours,
  relations, recognising unseen things).
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md needs a merge pass (over two
  pages). Nothing from cards 021–027 is committed.
