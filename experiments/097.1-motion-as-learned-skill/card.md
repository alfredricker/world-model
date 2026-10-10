---
id: "097.1"
title: walking as a learned skill (System 1), with conditional waypoints
rung: 1
serves: [P21, P9, P17, P12, P14]
status: done
verdict: fail
arch_version: 24
date: 2026-10-09
---

# 097.1: walking as a learned skill, with conditional waypoints

The first step of [card 097](../097-procedural-memory/card.md)'s series
(procedural memory), adapted on 2026-10-09 from [card
075](../075-motion-as-learned-system-1/card.md), drafted 2026-10-06 and
never run. Changes from 075: measured against version 20 and all three
tiers; the takeover rule is CHARTER's current one ("recall and networks,
each where it predicts better", 2026-10-06), which replaced "weights take
over from recall"; the literature review comes before approval. Runs
after card 094: what the network sees becomes card 094's layout and
object files instead of the believed map's tokens.

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
rare ones included (CHARTER: "recall and networks, each where it
predicts better")? P21
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

| | Version 20 | This card |
|---|---|---|
| How soon a target (a placement facing a token) is reached, and the next move | the field: value iteration over the believed map's placements, per situation and need | **System 1**: a network d(scene, target) → steps for each first move (left, right, forward); the move with the least is taken |
| What it sees | the believed map as a graph of placements | the believed map's tokens around the agent, as vectors with their offsets (card 044's where), and the target's offset and heading; no graph |
| How it learns | not learned (computed from recall's walkability and the learned moves) | from the agent's own stored moves, by hindsight (every later placement in a stored episode a target, `1707_01495`), as a quasimetric distance (`2211_15120`, `2509_20478`): blocked moves included, so walls and closed doors raise the distance |
| A target out of reach | the field crosses openable tokens at one step more; the first crossed becomes ("walk", j) | **waypoint condition**: System 2 asks System 1, for each openable token j in the scene, how soon the target is with j counted as floor; the j that brings it closest becomes ("walk", j) (the condition is what changes reachability) |
| Which answers (CHARTER) | the field always | System 1 where it has predicted held-out walking better than the field in similar scenes; the field elsewhere (a scene whose tokens recall has not seen walked among, or System 1 disagreeing with itself across its three moves) |
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

**Literature review (2026-10-10, before approval).** Hindsight
relabelling (`1707_01495`) makes every placement an episode reached a
target, so the agent's own moves, random ones included, are training
data. Quasimetric distances (`2211_15120`; TMD, `2509_20478`) learn
optimal steps-to-go from suboptimal data and stitch routes through
the triangle inequality; here the target is a place in the same scene,
not a second state, so the same optimality is had by Bellman backups
over recorded moves (fitted Q-iteration: steps after a move = 1 + the
least over the next moves), with the quasimetric form kept for when
targets become states. Search on the replay buffer (`1906_05253`) and
HIQL (`hiql`) agree on the risk: a learned distance is reliable only
over short reaches, so far targets go through waypoints (stored
observations in SoRB, subgoals about k steps ahead in HIQL), and SoRB
uses an ensemble's disagreement to refuse unreliable distances. HIQL
notes its value is unbiased only for deterministic dynamics; MiniGrid's
moves are deterministic. The card follows this: System 1 answers only
for targets within its view, the rest falls back to the field (counted;
waypoints for far targets are a later card), and disagreement triggers
the fallback.

**Changes at approval (2026-10-10, overnight).** Measured against
version 24 (best: tier 1 200/200 in 24.3 steps; tier 2 100/100 in 30.2;
tier 3 30/30 in 128.8, fresh 99/100). Card 094 stopped, so the network sees the believed
map's tokens: the 13 x 13 believed view around the agent (pl.view, as
card 070's token vectors) and the target's place and heading marked in
four planes. The trivial baseline is the exact open-space distance
(offset and heading only, by search): the ceiling of card 045's
network. Run in two stages: the fit check (offline agreement, criterion
1) first; the agent's runs (criteria 2 and 3) only if the fit check
passes. Data: the agent's own episodes on training seeds (test seeds +
500; tiers 1 and 2: 300 episodes, tier 3: 200), with 30% of actions
replaced by a random move so that bumps and detours are seen;
evaluation on the test seeds' unperturbed episodes (tier 1: 200, tier 2:
100, tier 3: 30 test + 70 fresh), at every step where the field walks
toward a need.

## 4. Data check

In the stored moves of tiers 1, 2 and 3: how many cross a doorway, end in a
bump against a wall or a closed door, or pass next to a key or ball; how
many distinct scenes around doorways. Card 012 failed on a rarely seen
doorway, so the rare classes are counted before training and, if thin,
card 068's play starts are used for memory.

## 5. Feasibility gate

- **Upper bound:** the field itself, the target System 1 must agree with
  (version 20: tier 1 100% in 24.4 steps).
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
   within 5% of 24.4; tiers 2 and 3 not worse than the best version
   (20: 95%, 14/30).
3. **Fallback and cost:** the field takes at most 5% of walking
   decisions on the tiers; time per step not above version 20's; nothing
   kept per pair of placements.

Reported: agreement by class; the waypoint conditions found against the
field's crossed tokens; fallback share by reason; training time.

**Prediction.** Agreement above 99.5% in open space and lower in
doorways and beside blockers, where moves are rare (card 012's failure);
criterion 1's rare classes are the risk, and may need card 068's play
starts. If criterion 1 holds, the tiers should match version 20, with the
field rarely used.

**Decision rules.** Keep if 1–3 hold (System 1 walks; the field stays
the fallback). Revise once if 1 fails in rare classes only (more data
there). Stop if open-space agreement is below 99% (the scene is not
enough).

**Budget.** Training about 20 minutes on the GPU (over 10 minutes: this
card's approval covers it); fit check a few minutes; tiers 1–2 about
25 minutes; tier 3 about 2 hours, handed to the user.

## 7. Result

### Data check

`runs/097.1/datacheck.json`. The agent's own moves on training seeds
(30% random): tier 1 9,780 moves, tier 2 11,889, tier 3 39,960; bumps
347, 590 and 1,645; moves at a doorway 1,361, 1,379 and 7,918 (39, 333
and 2,054 distinct doorway scenes); beside a key, ball or box 999,
5,006 and 16,522. The rare classes are not thin. All 800 training
episodes succeeded despite the random moves.

### Fit check, first version

`tools/card097.1/learn.py`, `runs/097.1/learn_v1.json`. One network
(a 13 x 13 convolutional encoder of the believed view and the target's
planes; steps to go per move, by fitted Q-iteration, 36 target updates;
3.5 minutes on the GPU), judged at the test episodes' walking decisions
(version 24, unperturbed) against the moves the field rates closest.

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Walking decisions (target in view) | 3,409 (100%) | 1,850 (100%) | 8,166 (94.1%) |
| System 1 agrees with the field | 99.30% | 99.46% | 99.02% |
| Open space | 99.96% | 100% | 99.04% |
| Doorway | 99.47% | 98.39% | 99.30% |
| Beside a key, ball or box | 97.08% | 99.33% | 99.16% |
| Target behind the agent | 92.08% | 96.32% | 96.84% |
| Baseline: open-space distance | 96.16% | 98.00% | 95.13% |

Criterion 1 not met (99.5% overall, 99% per class), though System 1
beats the open-space baseline in every tier, and open space is above
99% (so not the stop rule). The misses are not rare scenes: they are
long routes (12–13 steps, the target behind the agent past a wall)
where the moves' true distances differ by one step and the estimates
are off by about one (forward 12.21 and right 11.51, the field forward),
and the tie rule (forward among moves within 0.5 of the least), which
sometimes takes a move one step worse.

**Revision (the card's one).** The card's planned revision, more data
in rare classes, does not fit this diagnosis, so the revision instead
follows search on the replay buffer's two fixes for distance estimates
(`1906_05253`): steps to go learned as a distribution over whole steps
(0–40; the next placement's distribution moved one step on, a
categorical Bellman backup), and an ensemble of three copies that
answers only where its copies pick the same move (otherwise the field,
the card's "System 1 disagreeing with itself"). Distances are whole
numbers, so moves are compared after rounding, with forward, left,
right among equals. 48 target updates per copy.

### Fit check, revised

`runs/097.1/learn_r1.json`; three copies, 6.3 minutes each on the GPU.

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Walking decisions | 3,409 | 1,850 | 8,166 |
| System 1 answers (target in view, copies agree) | 97.1% | 99.3% | 92.8% |
| Field: target out of view / copies disagree | 0% / 2.9% | 0% / 0.7% | 5.9% / 1.4% |
| **System 1 agrees with the field, where it answers** | **100%** (3,311) | **100%** (1,837) | **99.85%** (7,574) |
| Open space | 100% | 100% | 99.86% |
| Doorway | 100% | 100% | 100% |
| Beside a key, ball or box | 100% | 100% | 99.90% |
| Target behind the agent | 100% | 100% | 98.87% (4 of 353 wrong) |
| Baseline: open-space distance, same decisions | 98.49% | 98.15% | 95.91% |

Criterion 1's agreement is met on tiers 1 and 2 (100% overall and in
every class), and on tier 3 (not required) overall but not in the
target-behind class. Its last clause, calling a target out of reach
exactly when the field does, was not measured: the recorded decisions
are those where the field found a route. Stage 2 runs.

### Stage 2: the agent walks by System 1

`tools/card097.1/system1.py` (`WM_WALK_S1=1`), `runs/097.1/s1/`. Where
the planner's step walks toward a need, the move is System 1's if a
target is in view and the three copies agree; otherwise the field's.
The field still decides whether a need can be reached, its cost and
crossings, and exploration's walking. Version 24 otherwise.

| | Version 24 | System 1 walking |
|---|---|---|
| Tier 1 (200) | 200/200, 24.3 steps | 200/200, 24.3 steps |
| Tier 2 (100) | 100/100, 30.2 steps | 100/100, 30.2 steps |
| Tier 3 (30) | 30/30, 128.8 steps | 30/30, 128.8 steps |
| Tier 3 fresh (100) | 99/100, 144.5 steps | 98/100, 143.3 steps |
| Walking moves made by System 1 (tiers 1 / 2 / 3 / fresh) | — | 97.1% / 99.3% / 92.3% / 94.2% |
| Its move differs from the planner's | — | 0 / 0 / 0 / 1,793 |
| Seconds per step (tiers 1 / 2 / 3) | 0.0017 / 0.0057 / 0.0339 | 0.0122 / 0.0418 / 0.0866 |

On tiers 1–3 System 1 never chose differently from the field, so every
episode is the same. On the fresh seeds 12 episodes differ by one move
(11 of them one step longer, one shorter), and seed 1003154 is lost:
1,780 of the 1,793 different moves are there. At step 40, holding a
box, the agent must walk to a target in view that is 20 steps away by
the route (around walls; about 10 in open space); the field says
forward, all three copies say left. Two steps later they say forward
into a wall; after the bump the planner finds no plan, the fallback
turns the agent back, System 1 turns it to the wall again, and this
repeats until the step limit.

That miss is of a kind the fit check had already shown. On tier 3's
recorded decisions, by the field's route length and the route's detour
beyond the open-space distance (`runs/097.1/by_route.json`): up to 12
steps with a detour of at most 4, 7,430 of 7,431 agree; up to 12 steps
with a longer detour, 101 of 104; over 12 steps with a longer detour,
26 of 33. The route then leaves the 13 x 13 view, so the scene cannot
show it, and the copies, trained on the same data, share the error, so
their agreement does not flag it. One such miss on the agent's path led
to states no recorded episode visited, where the errors repeated.

1. **Agreement:** met on tiers 1 and 2 (100% where System 1 answers,
   every class); out-of-reach calls not measured.
2. **Tiers:** tier 1 200/200 in 24.3 steps (same steps); tiers 2 and 3
   no different. Fresh tier 3 one episode worse (98 against 99 of 100).
3. **Fallback and cost:** not met. The field takes 2.9%, 0.7% and 7.7%
   of walking decisions (tier 3 above 5%: targets out of view 6.7%,
   copies disagreeing 1.0%); time per step is 2.6–7 times version
   24's, since three copies run on the CPU for each walking step and
   the field still runs. Nothing is kept per pair of placements (met).

## 8. Decision

**Stop** (System 1 does not take over walking in MiniGrid). What stands:
a network that sees only the believed view around the agent and a
target, trained on the agent's own moves by hindsight, picks one of
the field's closest moves at every walking decision of tiers 1 and 2
where it answers, and walking by it leaves tiers 1–3 unchanged. What
fails: it is not trustworthy once a route leaves its view, an ensemble
of copies trained on the same data does not detect that, and here the
exact field on a small graph is cheaper than the network. The next
step belongs to Crafter, where no graph of placements exists: System 1
answers only within the horizon its view covers (`1906_05253`'s
maximum edge distance, HIQL's subgoals a few steps ahead), and waypoints
from stored observations take the rest. The recorder (`WM_WALKREC=1`),
the trained copies (`runs/097.1/walk_net_r1_s*.pt`) and the hook
(`WM_WALK_S1=1`) remain.
