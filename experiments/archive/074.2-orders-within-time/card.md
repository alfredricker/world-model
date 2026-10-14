---
id: "074.2"
title: card 074.1's orders within time, by caching what conditions read
rung: 6
serves: [P21, P17, P12]
status: done
verdict: pass
arch_version: 17
date: 2026-10-06
---

# 074.2: card 074.1's orders within time

Card 074.1's revision for cost (the user, 2026-10-06: "do the 74.2
revision"). Its arm B (found orders followed, ties kept, two-part needs
split) met criteria 1 and 2 and failed criterion 3: 15 tier 2 episodes
ran out of time, against version 17's 6.

## 1. Question

Card 074.1 profiled tier 2 seed 1002078: 118 of 120 s went to card
074's order tests, nearly all in recomputing card 043's conditions in
each imagined situation. With what those conditions read cached, so
that every decision is the same, does arm B meet card 074.1's three
criteria? P21, P17 (the cost of deliberation), P12.

## 2. What changes

Nothing the agent decides. Two caches, both dropped exactly when what
they read can change:

| | Card 074.1, arm B | This card |
|---|---|---|
| Recall's distance between a stored try's (held, view) and the situation's | computed at every call (607,298 calls, 352 distinct, on seed 1002078's 35 steps) | cached per kind by the four handles it reads; dropped when the kind's weights change |
| "Doing a on u makes c true" for a part condition c, in situation st | computed at every call, recursively through the conditions it names | cached by (c, st); dropped with the planner's imagined effects at every `forget()` (a try added, a hypothesis changed); conditions naming a facing need are not cached (they read refused places) |

`tools/card074.2/cache.py`, `WM_CACHE=1`; `WM_CACHE=0` records each
episode's actions without caching, for the exactness check.

Also fixed: card 074.1's counter named `steps` overwrote each episode's
step count in its records (successes, timeouts and the paired tests
were unaffected; steps, time per step and the loop shares were not).
It is renamed, and this card re-measures them.

## 3. Dependencies

Card 074.1's arm B (`tools/card074.1/ties.py`, `WM_TIES=1 WM_SPLIT=1`)
on card 074 and version 17.

## 4. Data check

Not needed: the same episodes as card 074.1.

## 5. Feasibility gate

- **Exactness:** arm B with and without the caches, on tier 2's first 10
  episodes and the decoy world's red fold with the key known (100): the
  same action at every step (up to the shorter run where one runs out of
  time). Any difference stops the card.
- **Cost check:** time per step on those 10 tier 2 episodes, with and
  without the caches, and against version 17's 0.12 s.

## 6. Success criteria and prediction

Card 074.1's criteria, for arm B with the caches:

1. **The two cases:** decoy world with the key known, 100 of 100 in every
   fold; of version 17's 47 looping tier 2 episodes, at least half no
   longer loop.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than version 17
   (25%; McNemar, p < 0.05); tier 3 cannot be built.
3. **Cost and transfer:** tier 2 episodes out of time not more than
   version 17's 6; card 072's decoy folds with trying no fold worse
   (McNemar, p < 0.05).

**Prediction.** The same decisions, 2–3 times faster in the slow steps
(the distance cache alone halves seed 1002078's profile). Tier 2's
timeouts fall below 6 only if the slow episodes were slow throughout;
seed 1002078 has one step of about 60 s with both caches, so some may
remain.

**Decision rules.** Keep (version 18: card 074.1's arm B, card 043's
splice removed) if 1–3 hold. Revise if only 3 fails, with the slow
steps traced. Stop if the exactness check fails.

**Budget.** Gate about 15 minutes; main run about 25 minutes (decoy
world and folds 12 × ~70 s, tier 1 a minute, tier 2 about 10 minutes).

**Gate result.** Exactness: the same action at every step in all 10 tier
2 episodes and all 100 decoy episodes (`tools/card074.2/exact.py`). Cost,
10 tier 2 episodes: 0.262 s per step without the caches, **0.090 s**
with them (version 17: 0.12). Passed.

## 7. Result

`runs/074.2/` (`run.sh`; `tools/card074.2/summary.py` → `summary.json`).
Steps are true counts here (card 074.1's were not; see its correction).

| | Version 17 | Card 074.1, arm B | This card |
|---|---|---|---|
| Decoy world, key known (crit. 1) | 99 of 100 per fold, 21.6 steps | 100 every fold | **100 of 100 every fold**, 27.5 steps |
| Version 17's 47 tier 2 loops (crit. 1) | — | 38 solved | **47 no longer loop, 42 solved** |
| Tier 1 (crit. 2) | 100%, 20.8 steps | 100% | **100%**, 24.4 steps |
| Tier 2 (crit. 2) | 25% | 51% | **59%**; better than version 17 (42 vs 8, p < 0.001) and than 074.1's arm B (8 vs 0, p = 0.008) |
| Tier 2 out of time (crit. 3) | 6 | 15 | **4** (0.124 s per step; version 17 0.121) |
| Decoy folds with trying (crit. 3) | 94–96% | 93–96% | 93–96%; no fold worse (every p = 1.0) |

- **Criteria 1, 2 and 3: met.** Tier 3 cannot be built.
- The 8 episodes this card solves and 074.1's arm B did not are all
  ones B ran out of time on; the caches made the same decisions there,
  sooner.
- **Longer routes in tier 1** (24.4 steps against 20.8; 157 of 200
  episodes longer, 2 shorter). Traced, seed 1001004 (30 steps against
  18): facing the key, the agent turns away, walks to face the door, then
  returns for the key. Both needs of "open the door" are unmet and no
  order is found, so card 074.1's tie rule decides: fewer pick ups,
  toggles and drops first, and facing the door needs none. The rule
  ranks by acts before steps, so a far need with no act beats a near one
  with an act. The decoy world's 27.5 steps (21.6) are likely the same
  cause (not traced).
- Remaining tier 2 failures (41): 4 out of time, 24 other, 7 mostly
  random, 6 still loop (card 074.1 traced a full hand beside the key in
  two of them).

## 8. Decision

**Keep** (version 18: card 074.1's arm B with these caches; card 043's
splice removed). Every criterion holds, with the same decisions as card
074.1's arm B and tier 2 at 59% against version 17's 25%. Left for a
later card: the tie rule's cost ranks acts before steps and lengthens
routes by 17% in tier 1; ranking ties by the steps to the achiever's act
(the whole plan, not the first need) is the candidate.
