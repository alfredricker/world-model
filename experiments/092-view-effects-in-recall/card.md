---
id: "092"
title: view effects in recall, queried by effect, so the planner chains toward tiles not yet seen
rung: 1
serves: [P12, P15, P10, P16, P9]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 092: view effects in recall, queried by effect

The user (2026-10-08), after card 091's tier 2 failures: when it completes
an episode the agent should be able to reuse the goal hierarchy that
worked ("find the door", "see what's behind this door"), and infer from
the state which hierarchy applies. Then: rather than a separate store of
problems, recall already holds states and effects, and acting toward a
goal is a tree of conditions, so the hierarchy should come from querying
recall, and the learned recall (card 091's router), with the goal
condition. Card 091's forgetting and System 1 come after this card.

Approved under the user's overnight authority (2026-10-08: "work on 092
without stopping … don't ask for my approval on anything"); the
literature is in LITERATURE.md's current focus.

## 1. Question

Card 091's tier 2 failures (21 of 100) have no plan until the box is in
view (about 2% of their steps planned): the planner chains backward
through recall only to tiles in this episode's believed map, and recall's
outcomes record whether the front tile or the hand changed, never which
tiles came into view. If each stored try also records the kinds of tiles
that came into the believed view, and the planner can ask recall which
tries produced a condition it needs ("a box is in the believed view"),
ranked by how much their situations resemble the present one, does the
agent plan from the first step? Rung 1; P12 (from any state, how likely a
goal is and which conditions raise it), P15, P10, P16, P9.

## 2. What changes

One component: recall's outcomes gain the view, and recall is queried by
outcome as well as by situation. The planner's backward chaining, card
074's ordering, the router and walking are unchanged.

```
version 18 + 091:  outcome(try) = (front changed, hand changed)
                   need "box in front", no box in the believed map  →  no chain  →  fallback
this card:         outcome(try) = (front changed, hand changed, kinds that came into the believed view)
                   need a tile of kind u, none in the believed map  →  need "u in the believed view"
                   recall by effect: stored tries whose outcome brought u into view, each with its key
                   (front, held, view); weight = how often the effect followed  ×  the router's kernel
                   between the try's situation and the present one
                   →  their conditions become needs (toggle a locked door holding its key ← pick up the key …)
```

- **Which problem the agent is in** is no longer a separate decision:
  the stored tries whose situations are nearest the present one (card
  091's embedding) carry the effects that worked there, so a tier 3
  situation recalls tier 3's tries first, and fragments from other worlds
  where those are absent.
- **How likely** each branch is, is the share of similar tries in which
  the effect followed (how often opening a door like this one revealed a
  box): P12's likelihood of a goal from a state, read from recall.
- **Adjusting:** own tries first (card 050) and failed tries as "not
  under these conditions" (card 072) apply to view effects as to any
  other outcome; the episode's own tries are added as they happen.
- **What "came into view" is:** the kinds in the believed view after the
  try that were not in it before (card 062's believed view, from the
  agent's own frames; nothing read from the simulator).

## 3. Dependencies

Card 091 (router, keys, embedding); card 062 (believed view); card 050
(own tries first); card 072 (failed tries are situational); card 074
(ordering needs in a chain); card 070 (key–door relation). Regression
planning over effects (STRIPS, Fikes and Nilsson 1971) and case-based
planning (Hammond 1989): to be added to LITERATURE.md with `papi` before
approval (CHARTER rule 7).

## 4. Data check

In memory 066 of each tier: how many stored tries brought a kind into the
believed view, by action and kind; specifically tier 2's toggles of a
locked door after which a box came into view, and tier 3's (boxes
revealing keys, doors revealing rooms). If tier 2 holds too few, the
training layouts (seeds from 500,000) add them.

**Result** (`tools/card092/datacheck.py`, `runs/092/datacheck.json`;
kinds that came into the believed view away from the front tile, which
is the front effect recall already has):

| Memory | Toggles | A kind came into view | Mostly |
|---|---|---|---|
| Tier 1 | 67,625 | 776 | the goal (538, every one after a locked door) |
| Tier 2 | 66,843 | 335 | boxes (262, after a locked door), open doors |
| Tier 3 | 84,538 | 5,566 | locked doors, a grey box, a green ball, open doors |

Pick up and drop bring a kind into view rarely (about 0.1% in tiers 1–2,
3–4% in tier 3, mostly open doors seen past the agent). Some come from
fronts that cannot have revealed them (tier 3: 1,276 toggles facing a
wall); recall's weighting by how often the effect followed in similar
situations must outweigh these. In tier 3, keys come from boxes as the
front effect (toggle a box → the front becomes a key), which recall
already holds.

## 5. Feasibility gate

- **Upper bound:** the 21 tier 2 failures with recall's view effects
  replaced by the true ones written by hand (opening the locked door
  shows the box): the planner chains from the first step and reaches the
  goal in most of them. If not, the planner cannot use such needs and the
  card stops.
- **Effect recall:** on held-out tries, predicting which kinds come into
  view (log-likelihood per try) better than each action's frequencies.
- **Trivial baseline:** card 091's agent, tier 2 79%.

## 6. Success criteria and prediction

1. **Tier 2:** better than card 091 on the same 100 seeds (sign test on
   the seeds that differ); planned steps before the box is seen rise from
   about 2% to most.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than the best
   version; tier 3 (if tier 2 is clearly better, the user, 2026-10-08)
   against card 091 on the same 30 seeds.
3. **Tier 3's chains:** how many needs on unseen tiles get a chain from
   recall (a key inside a box, a room behind a door), against card 091's
   agent reaching them by fallback.

**Prediction.** Tier 2 recovers most of the 13 "door never opened"
episodes; the 8 that open the door and then fail on a blocked route gain
less. On tier 3 the gain depends on long chains staying ranked well when
every door and box is a candidate.

**Budget.** Data check and effect recall: minutes. Gate: 21 episodes,
about 10 minutes. Main: tiers 1–2 about 35 minutes; tier 3 about 2 hours,
handed to the user.

## Note: episode-level, multi-condition memory

Worth its own card or experiment, after this one. This card keeps memory
at single tries, one effect each, and builds goal hierarchies by chaining
them. Memory of whole episodes, or of several conditions met together (a
chain of effects that reached a goal, with its order and the situation
at its start), carries long-range order that one-step chaining must
rebuild, and a prior over which branch to try first. Candidates: a stored
episode's chain as a prior over recall's branches; the router trained to
embed sets of conditions as well as single situations. The test would be
tier 3, where chains are long and every door and box is a branch.

## Gate result

- **Upper bound: passes.** Card 091's 21 tier 2 failures with the true
  view effect written by hand (`WM_REVEAL=oracle`; version 19 + card
  087's forward prior, below): 17 solved, most in 26–46 steps
  (`runs/092/gate_oracle_*`). Left: 1002013, 1002015, 1002048 (the
  reveal step acts 240–290 steps without success) and 1002088 (box seen,
  then fails).
- **Effect recall: passes.** Held-out tier 2 toggles (a fifth):
  recall −0.0035 per toggle and kind against each kind's frequency
  −0.0051; on the 48 where a kind came into view, −3.3 against −7.3
  (`runs/092/effects_gate_tier2.json`).
- **Found on the way: forward's prior.** The locked green door was never
  a candidate to open: "green door, open" has no forward tries in tier
  2's memory, and both version 18's and card 091's forward priors call
  it blocked (the router's forward vote, trained on 66 stored tiles, does
  not separate open from closed doors). Card 087's property network
  calls every open door walkable (0.68–1.0) and every locked or closed
  door, key, ball and box blocked. By card 091's own rule (per action,
  whichever predicts held-out tries better) forward's prior is card
  087's (`WM_PROPS=1`), in both arms below, so the arms differ only by the
  reveal step.

## Result: tiers 1 and 2

`runs/092/run.sh`; the same seeds as cards 074.2 and 091.

| Tier | Version 19 (091) | Base: 19 + 087's forward prior | **092: base + reveal** |
|---|---|---|---|
| 1 (200) | 100%, 24.4 steps | 100%, 24.4 | **100%**, 24.4 |
| 2 (100) | 79%, 72.5 steps; random 36% | 96%, 78.1; random 12% | **95%**, **56.4**; random **0.7%** |

- **Criterion 1 (tier 2 better than card 091): met**, 17 seeds won, 1
  lost (p = 0.0001). Against the base arm the reveal step is no
  different in success (0 won, 1 lost) but takes 28% fewer steps when
  solved and nearly no random actions: the gain in success is card 087's
  forward prior with card 091's router (base against 091: 17 won, 0
  lost, p = 0.00002).
- **With card 091.1** (both arms): tier 1 100%; tier 2 95% and 95%, the
  same 5 seeds failing; steps when solved 64.3 (reveal) against 119.4
  (base); random actions 0.4% against 8.7%.

## Result: tier 3

| Arm (091.1 + 087's forward prior) | Tier 3 | Steps in 20 × 300 s (equal load) |
|---|---|---|
| Base | **14/30** (2,055 steps when solved) | 3,044 |
| **092: base + reveal** | **0/20** (stopped at 20 of 30; every episode at the hour with 67–110 steps) | 824 |
| Version 18 | 5/30 | 2,970 |

The reveal step is 3.6 times slower: with the goal unseen, nearly the
whole of a tier 3 episode, it searches the map for a frontier pose (and
a door to open) at every step, where version 18 explored about 25 times
per episode and otherwise continued a guess cheaply. Restoring card
060's "last step's door first" did not change the speed (reverted).

- **Criterion 2: not met** (tier 3 worse than the best, 0/20 against
  14/30, from speed under the hour).
- **Criterion 3:** not measured (no tier 3 episode ran long enough).

## Decision

**Revise.** On tier 2 the reveal step plans from the first step and
halves the steps (64 against 119 when solved) with nearly no random
actions, but adds no successes over the base arm (95% and 95%); on tier
3 its per-step map search makes it too slow for the hour. The base arm
it was measured against became version 20 (card 091.1). Proposed
revision (not run; tier 2 did not fall below 79%, so the user's
overnight rule for auto-approved revisions did not apply): keep the
reveal step's target between steps while it still gives a plan, and
search the frontier incrementally (only places whose belief changed),
gated on tier 3 speed under 20 parallel episodes before the full run.
