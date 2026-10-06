---
id: "075"
title: motion as a learned System 1, with conditional waypoints
rung: 6
serves: [P21, P9, P17, P12, P14]
status: draft
verdict:
arch_version: 17
date: 2026-10-06
---

# 075: motion as a learned System 1, with conditional waypoints

Asked for by the user (2026-10-06): "Motion should probably be a learned
system 1, with conditional waypoints ... our mechanism likely won't
transfer to more continuous environments e.g. crafter, minecraft,
robotics."

## 1. Question

Walking today (card 067) is a closeness computed over the believed map's
placements, from learned parts: it needs a discrete graph of places and
moves, which Crafter's larger maps, Minecraft and robots do not offer as
such, and its dependencies are invisible to the conditions (card 074's
route case). Can a network trained on the agent's own moves, seeing the
local scene and a target, give the next move and how soon the target is
reached, with System 2 adding waypoint conditions only where it says the
target is out of reach, and take over from the field on every case,
rare ones included (CHARTER: "weights take over from recall")? P21
("the network maps state and goal straight to an action"), P9, P17, P12,
P14.

**What was tried before.** Card 012: learned walking 30% against 95% for
exact walking, failing in a rarely seen doorway (LESSONS). Cards 041 and
045: a how-soon network on the target's offset and heading alone, so
open space only; obstacles fell to System 2, which searched waypoints
over every pair of standing placements (457,000 pairs kept), and card 067
removed it for scale (declared exception, rule 8: "a network may later
take over where it agrees with the field on every case"). This card
differs in what the network sees: the scene, so it learns obstacles as
well as distance.

## 2. What changes

One component: walking (card 067's field).

| | Version 17 | This card |
|---|---|---|
| How soon a target (a placement facing a token) is reached, and the next move | the field: value iteration over the believed map's placements, per situation and need | **System 1**: a network d(scene, target) → steps for each first move (left, right, forward); the move with the least is taken |
| What it sees | the believed map as a graph of placements | the believed map's tokens around the agent, as vectors with their offsets (card 044's where), and the target's offset and heading; no graph |
| How it learns | not learned (computed from recall's walkability and the learned moves) | from the agent's own stored moves, by hindsight (every later placement in a stored episode a target, `1707_01495`), as a quasimetric distance (`2211_15120`, `2509_20478`): blocked moves included, so walls and closed doors raise the distance |
| A target out of reach | the field crosses openable tokens at one step more; the first crossed becomes ("walk", j) | **waypoint condition**: System 2 asks System 1, for each openable token j in the scene, how soon the target is with j counted as floor; the j that brings it closest becomes ("walk", j) (the condition is what changes reachability) |
| Fallback (CHARTER) | none needed | the field, where System 1 is judged novel outside the network (a scene whose tokens recall has not seen walked among) or disagrees with itself across its three moves |
| Kept per world | nothing beyond placements | the same; nothing per pair |

Unseen targets stay with exploration (card 057). Card 074 reads "can
this be reached?" from whichever walking is in use, so the two cards do
not depend on each other. In worlds without a field, the fallback would
be waypoint conditions alone; that is a later card.

## 3. Dependencies

Card 044's tokens and where; card 045's hindsight training on stored
moves; card 067's field (the evaluator and the fallback); memory as
cards 066 and 068 build it. Hindsight relabelling (`1707_01495`),
quasimetric distances (`2211_15120`, `2509_20478`), learned subgoals over
a goal-conditioned value (`hiql`), waypoints from a learned distance
(`1906_05253`).

## 4. Data check

In the stored moves of tiers 1 and 2: how many cross a doorway, end in a
bump against a wall or a closed door, or pass next to a key or ball; how
many distinct scenes around doorways. Card 012 failed on a rarely seen
doorway, so the rare classes are counted before training and, if thin,
card 068's play starts are used for memory.

## 5. Feasibility gate

- **Upper bound:** the field itself, the target System 1 must agree with
  (version 17: tier 1 100% in 20.8 steps).
- **Trivial baseline:** card 045's open-space network (offset and heading
  only), scored the same way: how often its move is one of the field's
  closest.
- **Fit check:** on 20 held-out layouts, System 1's agreement after
  training, by class, before any episode is run.

## 6. Success criteria and prediction

1. **Agreement (the takeover rule):** on held-out layouts of tiers 1 and
   2, System 1's move is one of the field's closest in at least 99.5% of
   walking decisions, and in at least 99% within each rare class
   (doorways, next to a blocker, target behind the agent); it calls a
   target out of reach exactly when the field does in at least 99%.
2. **CHARTER's tiers, walking by System 1:** tier 1 ≥ 99% with steps
   within 5% of 20.8; tier 2 not worse than the best version.
3. **Fallback and cost:** the field takes at most 5% of walking
   decisions on the tiers; time per step not above version 17's; nothing
   kept per pair of placements.

Reported: agreement by class; the waypoint conditions found against the
field's crossed tokens; fallback share by reason; training time.

**Prediction.** Agreement above 99.5% in open space and lower in
doorways and beside blockers, where moves are rare (card 012's failure);
criterion 1's rare classes are the risk, and may need card 068's play
starts. If criterion 1 holds, the tiers should match version 17, with the
field rarely used.

**Decision rules.** Keep if 1–3 hold (System 1 walks; the field stays
the fallback). Revise once if 1 fails in rare classes only (more data
there). Stop if open-space agreement is below 99% (the scene is not
enough).

**Budget.** Training about 20 minutes on the GPU (over 10 minutes: this
card's approval covers it); fit check a few minutes; tiers as card 073.
