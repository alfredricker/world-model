---
id: "092"
title: view effects in recall, queried by effect, so the planner chains toward tiles not yet seen
rung: 1
serves: [P12, P15, P10, P16, P9]
status: draft
verdict:
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
