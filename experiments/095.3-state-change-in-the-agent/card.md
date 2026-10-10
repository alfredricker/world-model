---
id: "095.3"
title: arm C in the agent — recall predicts pick up, toggle and drop by one rule over every token
rung: 1
serves: [P6, P10, P17, P3]
status: approved
verdict:
arch_version: 20
date: 2026-10-10
---

# 095.3: arm C in the agent

A step of [card 095](../095-state-change-prediction/card.md)'s series.
[095.1](../095.1-which-tokens-change/card.md) found offline that arm C
(arm B's vote over stored token instances, arm A's network as its prior)
learns by itself that only the token in front and the hand change, and
gives results as relations (the red-box error gone for every colour).
[095.2](../095.2-recall-at-scale/card.md) made its vote 31× faster on
tier 3 with an index and forgetting (18× less RAM), without changing a
prediction. This card puts it in the agent. The user, 2026-10-09: the
criteria may trade "a margin of prediction / planning success decrease
in favor of a large performance gain in speed and RAM".

## 1. Question

Does the agent plan as well or better when recall's prediction for pick
up, toggle and drop is arm C rather than version 20's two-slot classes,
and faster? Rung 1; P6, P10, P17, P3.

## 2. What changes

```
version 20: (a, front u, hand h, view set) → router + own tries → class (front?, hand?) → stored result or transport
this card:  (a, front u, hand h, context)  → arm C on the front's and the hand's tokens  → P(change) and the
            relation's result per token (B: index over the pruned store; A: the network) → own tries first
```

- **Queries.** The planner keeps asking "what if I do a with u in front
  and h in the hand?" (`World.outcome`, `kd.predict`). Each becomes two
  arm C instances: u at the front's place, h at the hand's. Their
  context (095.1's 13 places within two steps, and the hand) is the
  believed view when the query is the agent's real situation; otherwise
  (imagined poses, stored situations, guesses about tokens elsewhere)
  only the front and the hand are known, and conditions on the unknown
  places are left out of the distance (marginalised), not matched to
  "absent". Arm A gets the same known tokens.
- **Category and result.** P(front changes) and P(hand changes) give
  the four categories as a product; `cat_of` keeps its 0.5 threshold.
  The result per changed token is B's relation, else A's, as in 095.1.
  Other tokens: none change in MiniGrid (095.1); 095.5 lets the planner
  read them.
- **Own tries first,** as version 20: this episode's tries at the same
  admitted fields count before the vote, P = (N_own + β P_C)/(N_own + β),
  version 20's β; cleared each episode.
- **Memory:** 095.2's pruned store per tier (groups, merged changed
  instances), built from memory 066's interaction tries; arm A from
  095.1.

**Unchanged:** moves; `templates` and `opened` (stored situations by
outcome class, used to find conditions: 095.5 replaces them); trying,
conflicts, ties, properties; the encoder. The router (card 091) is no
longer read for these three actions.

## 3. Dependencies

095.1 (arm C, kept), 095.2 (index and forgetting, kept), version 20's
planner and own-tries rule (cards 038, 051, 072, 074). The planner
reads recall through `World.outcome` (`tools/card038/vector_planner.py`),
`cat_of`, `predict` (card 072's guesses rank by its probabilities) and
`result` (`tools/card051/index.py`); every caller goes through these.

## 4. Data check

On 5 tier 2 and 3 tier 3 episodes with version 20 (about 10 minutes):
- planner queries by source: real situation, imagined, stored
  situation, guess; the share where only the front and the hand are
  known;
- a timer around recall's interaction calls: the share of a step's time
  they take on tier 3. If under half, criterion 2's 2× cannot come from
  this card, and the margin does not apply.

## 5. Feasibility gate

- **Upper bound (offline, on 095.1's held-out tries):** arm C with only
  the front and the hand known (the rest marginalised): exact next state
  within 0.1 points of arm C with the full context, and the box test
  100%. If it fails, imagined queries need the believed map around the
  imagined pose first.
- **In the agent:** on 20 tier 1 episodes, arm C's prediction for each
  real try matches what happened in at least 99.9% of tries.
- **Trivial baseline:** version 20 (tier 1 200/200, tier 2 95/100, tier
  3 14/30 at 1.45 s per step, 15 episodes out of time).

## 6. Success criteria and prediction

1. **Tiers** (CHARTER's three): tier 1 ≥ 99%; tiers 2 and 3 no worse
   than version 20 — or, **only if criterion 2 is met**, within a margin:
   tier 2 ≥ 92/100 (version 20 minus 3) and tier 3 ≥ 12/30 (minus 2).
2. **Speed:** tier 3 seconds per step at most half of version 20's
   (≤ 0.72 s, same 30 seeds, same one-hour cap); tier 2 not above
   version 20's (0.026 s). Peak RAM per worker reported.
3. **Relations in the agent:** on tier 2, no pick up the planner
   predicts puts a different box in the hand than the one in front
   (version 20: the red box, on its box failures).

**Prediction.** Tier 1 200/200. Tier 2 97–100: the box failures go.
Tier 3: if interaction recall is most of a step, 2–5× faster steps,
fewer episodes out of time, 14–18/30; if not, about version 20's speed
and successes.

**Budget.** Data check 10 minutes; gate 15 minutes; tiers 1–2 about 20
minutes. Tier 3 (30 seeds, one-hour cap, about 100 minutes) is handed to
the user as a command.

## Gate result (2026-10-10)

- **Offline** (`tools/card095.3/gate.py`, `runs/095.3/gate.json`): arm C
  with only the front and the hand known loses nothing. Exact next state
  on held-out tries: tier 1 100% (full context 100%), tier 2 100%
  (99.99%), tier 3 99.83% (99.84%), within 0.01 points; box test 100%
  for every colour. The marginalised store is smaller (tier 3: 47,000
  groups against 194,000).
- So **every** query uses this one rule, keyed on (action, front, hand),
  and the believed view is not needed: imagined, stored and real queries
  are answered alike. The data check's count of query sources is
  dropped; its timer stays.
- **In the agent** (`tools/card095.3/statechange.py`, 20 tier 1
  episodes): 20/20 solved; arm C's prediction matched all 40 real
  interaction tries (category and result); no pick up predicted another
  token in the hand than the one in front (210 predictions).

Gate passed.
