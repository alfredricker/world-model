# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-10, `master`. **Rung:** 1, not passed. **Architecture:**
  version 27 (cards 099–101, with the user's approval; versions 21–24 kept
  overnight, the user may revert). **Direction:** CHARTER's "Current
  direction", rule 8; the standing evaluation is its three MiniGrid tiers.
- **Best version:** 27. Tier 1 **200/200** in 24.3 steps; tier 2 **100/100**
  in 30.2; tier 3 **30/30** in 118.8, 0.012 s per step; fresh tier 3 seeds
  **100/100** (version 26: 98/100, no different).
- **Cleanup after the audit of version 24** (the user's request):
  - [099](experiments/099-version-24-without-dead-parts/card.md) **keep** (v25): arm C alone for
    interactions; the old recall, router and fits are no longer built.
  - [100](experiments/100-one-held-intention/card.md) **keep** (v26): the intention is the only memory of
    choices; its chain breaks ties.
  - [101](experiments/101-one-rule-with-no-plan/card.md) **keep** (v27): one rule with no plan (the act
    whose outcome is least known, held, made whatever is predicted)
    replaces inform and trying. No fallback loop; both lost seeds won;
    about 5 more steps on seeds both win. Decoy check not run: arm C is
    not held out from the decoy fold.
- **Next from the audit:** 102 (where to drop: the plan's drop that blocks
  its own route; 1003160 is now won only by the held act); moves by arm
  C; the surprise trigger, which is untested.
- **Crafter, offline:** [098](experiments/098-crafter-tokens/card.md), [098.2](experiments/098.2-tokens-across-light/card.md) and [098.3](experiments/098.3-still-by-the-view/card.md) revise (same code by
  day and night 95%); [098.1](experiments/098.1-crafter-interactions/card.md) uninterpretable, needs a no-action baseline.
  Backlog 091–097 done ([overnight-10-10.md](overnight-10-10.md)). Waiting: 086; colour paused.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
