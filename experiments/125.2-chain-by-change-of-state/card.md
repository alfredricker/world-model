---
id: "125.2"
title: the chain by change of state — which touches set a condition, read from memory between steady readings
rung: 2
serves: [P6, P4]
status: done
verdict: pass
arch_version: 27
date: 2026-11-10
---

# 125.2: the chain by change of state

Roadmap step 2 (event segmentation), rung 2. The last small revision
before card 126 (the user, 2026-11-10). Card
[125.1](../125.1-conditions-read-where-steady/card.md) kept the steady
rule and failed its chain test, which asked for the single touch that
sets a lock and used a truth that ignored the contact sensor's lag.
Its diagnostics suggested the chain is exact when the unit is the
change of state. This card judges that on fresh episodes.

Biological counterpart: as card 125.1 (event models updated at
boundaries); credit for a change goes to what happened between two
stable states, as in contingency learning (Cheng and Novick 1990).

## 1. Question

From memory alone, can the agent tell when a lock's state changed and
where it has to touch to change it? P6, P4.

## 2. What changes

One component: the chain's unit and its test. The records, places,
steady rule, candidate patches and two states are card 125.1's.

```
125.1:  a touch sets the condition if the state differs before the next touch
125.2:  an interval = two consecutive touches of an episode with a steady reading of the condition;
        it changed if their states differ; its candidate causes are the touches from the first
        up to before the second; the cause place is the place most changed intervals share
```

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

As card 125.1's table. One addition:

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A condition changes only through the agent's touches | fails: day and night, creatures | the same | mostly holds | fails with other agents | holds in this scene; changes with no touch between are not handled |

## 3. Dependencies

Cards 125 and 125.1 (scripts reused unchanged).

## 4. Data check

Fresh episodes 400–499 of the state dataset as the memory. No test
set is needed: the chain is read inside the memory and scored against
the true button states.

## 5. Feasibility gate

`chain.py` on card 125.1's memory (episodes 600–699, already seen),
run 2026-11-10.

| | Drawer | Window |
|---|---|---|
| Intervals; of them changed | 1,480; 216 | 1,500; 181 |
| Detected changes that are true | 100% | 100% |
| True changes detected | 100% | 100% |
| Cause place is the true button's place | yes | yes |
| Changed intervals holding a touch at the cause place | 100% | 100% |
| Unchanged intervals holding one | 3.3% | 3.0% |
| Trivial baseline: the best other place, in changed intervals | 29% | 32% |

Truth of the cause place: the place where that slide's button flips
most, counting flips from 5 frames before sensed contact (the lag
found in card 125.1). Upper bound: the table itself is at the ceiling.
Gate passed.

## 6. Success criteria and prediction

On episodes 400–499, for every slide place:
1. at least 95% of detected changes are true, and at least 95% of
   true changes are detected;
2. the cause place is the true button's place;
3. at least 90% of changed intervals hold a touch at the cause place,
   and at most 10% of unchanged intervals do.

**Prediction.** All three pass; the gate was at the ceiling and the
procedure has no fitted setting beyond card 125.1's.

**Budget.** About 10 minutes in steps under 10 minutes, capped.

## 7. Result

Run 2026-11-10 on episodes 400–499 (2,106 touches, four slide places;
the two drawer places share one condition, the two window places
found the same button from slightly different candidate patches).

| | Drawer | Window (two places) | Bar |
|---|---|---|---|
| Intervals; of them changed | 1,577; 230 | 1,584; 219 and 1,574; 215 | |
| Detected changes that are true | 100% | 100% | 95% |
| True changes detected | 100% | 100% | 95% |
| Cause place is the true button's place | yes | yes | yes |
| Changed intervals holding a touch at the cause place | 99.6% | 100% | 90% |
| Unchanged intervals holding one | 3.0% | 2.8–2.9% | 10% |
| The best other place, in changed intervals | 23% | 25% | |

All three criteria pass at every slide place. A condition has a
steady reading before 80% of touches. In the few intervals with a
single touch between the readings and that touch at the cause place
(2 to 4 per condition), the state changed every time.

## 8. Decision

**Keep.** With cards 125 and 125.1 this gives, from a memory of 100
episodes of play, with no training, outlines or labels: the patches
that decide whether a pull works (the lock's button), their two
states, when the state changed, and the place to touch to change it.
Limits, all declared: a fixed camera; touch from the state dataset;
two-way outcomes and two-state conditions; conditions that only the
agent's touches change; one setting of each choice (10 places, 10
patches). The cube is untested. Next: card 126.
