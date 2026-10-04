# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-04. **Rung:** before rung 1. **Architecture:** version 9
  (kept with card 050, 2026-10-04): version 8 with recall through the
  conditions that matter (049) and a query's own tries first (050).
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions, not
  imagined action sequences (GOAL.md P21). User, 2026-10-01: codes for
  identity, vectors for similarity; fixes must be general principles.
  User, 2026-10-03: before scaling (BabyAI, card 052), fix memory:
  recall should read only the conditions that matter for each action,
  generative and preventive, so irrelevant tokens cannot veto a memory.
- **Next decision:** card 051 (the planner, draft design), step 1:
  near-linear cost per step with no change in decisions, on spare seed
  399 under 10 minutes. Cards renumbered 2026-10-04 to follow their
  dependencies: 049-050 recall (version 9), 051 planner, 052 staged
  encoder (BabyAI).
- **Recent cards:**
  - [052](experiments/052-staged-encoder/card.md): draft, waits on 051;
  - [051](experiments/051-planner-chains-and-rules/card.md): draft design
    (kept links, rules from recall, near-linear cost);
  - [050](experiments/050-own-tries-first/card.md): pass, keep (version 9);
  - [049](experiments/049-recall-through-conditions/card.md): fail, revise;
  - [048](experiments/048-chained-rooms/card.md): pass, keep.
- **Before BabyAI or Crafter (the user, 2026-10-03):** recall and the
  planner's reasoning over conditions must be near-linear per step,
  independent of memory size; today a lifetime costs about T² and card
  049's admission is n² (ARCHITECTURE.md, "Known limits").
- **Open in version 9:** no commitment between steps (two doors 0/150);
  "unknown" read as "fails"; parts mix kind and colour; conjunctions
  spliced; no relations; a small fixed world.
- **papi:** GLM-5.3 via OpenRouter; needs credit for summaries.
- **Pinned:** card 048's first follow-up (hand changes imagined, not
  spliced; card 051's change 6); card 029's arm 3 in the both world;
  demonstrations once random play is too thin; GOAL.md hypotheses draft.
