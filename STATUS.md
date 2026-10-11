# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-11, `master`. **Rung:** 1, not passed. **Architecture:**
  version 27 (cards 099–101, with the user's approval; versions 21–24 kept
  overnight, the user may revert). **Direction:** CHARTER's "Current
  direction", rule 8; the standing evaluation is its three MiniGrid tiers.
- **Best version:** 27. Tier 1 **200/200** in 24.3 steps; tier 2 **100/100**
  in 30.2; tier 3 **30/30** in 118.8, 0.012 s per step; fresh tier 3 seeds
  **100/100** (version 26: 98/100, no different).
- **Cleanup after the audit of version 24:** [099](experiments/099-version-24-without-dead-parts/card.md) **keep** (v25,
  arm C alone for interactions), [100](experiments/100-one-held-intention/card.md) **keep** (v26, the intention
  the only memory of choices), [101](experiments/101-one-rule-with-no-plan/card.md) **keep** (v27, one rule with
  no plan: the act whose outcome is least known). Decoy check not run.
- **Now: Crafter** (the user, 2026-10-10; CHARTER's "Current world"
  amended). [098.5](experiments/098.5-tokens-that-persist/card.md) **revise**: the revised files level
  holds identity on the true state (IDF1 0.949; first build 0.853) and
  leaves MiniGrid unchanged, but on the encoder's tokens one false "moved
  by itself" event made all grass a mover. [098.6](experiments/098.6-what-moves-by-itself/card.md) **revise**: a file
  when tokens like it have needed memory of their own (the user's rule).
  True state 0.934, static tokens 0.39% of files, MiniGrid identical;
  tokens 0.717 (night 82% unfollowed): "like this" at the identity
  radius mixes movers with scenery. Next, the user's choice: the width of
  "like this" by held-out prediction, or night tokens (encoder). Then
  the network card and [098.4](experiments/098.4-what-the-action-causes/card.md) (draft).
- **Crafter, offline:** [098](experiments/098-crafter-tokens/card.md), [098.2](experiments/098.2-tokens-across-light/card.md) and [098.3](experiments/098.3-still-by-the-view/card.md) revise (same code by
  day and night 95%); [098.1](experiments/098.1-crafter-interactions/card.md) uninterpretable, needs a no-action baseline.
  Backlog 091–097 done ([overnight-10-10.md](overnight-10-10.md)). Waiting: 086; card 102 set aside; colour paused.

### Fred note
Delete when this gets its set of cards.
Currently memory is evaluated as a distance between states. We are still treating "hand, front, and View" separately, whereas we should work with everything as a token, with its what and where information. I want to try out a neural network between state and recall that learns how to route states to past events. From this some nice properties might emerge:

1. Smart forgetting: For a set of nearly identical states, a neural network will learn to prefer only a subset of them for solving similar problems, which allows us to prune memory with near zero weights that has existed for a long time. 
2. The same network can learn to just output actions directly (System 1)
3. Learning position relations
4. Linear cost of memory growth with environment size (if engineered correctly)
