---
id: "073"
title: order needs by conflicting states (reasonable goal orderings)
rung: 6
serves: [P21, P12, P3, P19]
status: done
verdict: pass
arch_version: 16
date: 2026-10-05
---

# 073: order needs by conflicting states

**Definition** (asked for by the user, 2026-10-05):

> If need A requires a state that need B destroys (clearing the blocker
> requires an empty hand; fetching the key fills it), and A's result
> survives B (the tile stays clear while the key is fetched), then **A
> comes before B**.

This is Koehler and Hoffmann's reasonable goal ordering (2000, JAIR 12,
Definition 8; papi `1106_0243`): "B ≤r A [means] that if, after the goal
A has been achieved, there is no plan anymore that achieves B without —
at least temporarily — destroying A, then B is a goal prior to A." It
holds at every depth of the chain of conditions, not only between one
achiever's needs (the user: "this should work hierarchically").

## 1. Question

When an achiever's needs compete for the same state (one hand: "key in
hand" against the empty hand that clearing a blocker needs), version 16
cannot tell which comes first and loops. If needs are ordered by the
states they conflict over, and the states that earlier needs produce
stay protected, does the agent clear a blocker before fetching the key,
with no knowledge of hands written in? P21 (working backward over
conditions, no rollouts), P12, P3, P19.

**Why version 16 fails** (card 072's trace, decoy seed 1001029, the
decoy key on the tile in front of the door):
- **The conflict check is symmetric.** Card 051 imagines each need's plan
  and asks whether it breaks what another need's plan relies on. Both
  plans fill the hand (pick up the key; pick up the blocker), so each
  "threatens" the other: in this episode 224 of 508 weighings, 0
  reorders, and card 029's fixed order (the hand first) decides.
- **A state true from the start is not protected.** Holding the key, the
  toggle works as things are, so no hand need is listed. The drop that
  clears the hand for the blocker undoes nothing the planner protects.
- The agent picks the key up and drops it for 640 steps. Six of 100
  decoy layouts do this even with the key known (94%), and tier 2's ball
  in front of the door is the same case.

## 2. What changes

One component: how the planner orders and protects the needs of a
chain (card 051's threats, card 029's protection).

| | Version 16 | This card |
|---|---|---|
| Two needs in conflict | each need's plan imagined; one whose act breaks another's links threatens it; if all threaten, card 029's order | **reasonable orders**, read from conditions: B before A when, in the situation A produces (recall's effect of A's chain's final act on the present), every way to B (the planner's achievers for B) has a need on a part (held or in view) that A fixed and that is unmet there (Koehler and Hoffmann's Definition 10, on recall's conditions); the order holding one way only decides |
| Which needs are compared | an achiever's unmet needs | an achiever's needs, met or unmet, and the needs inside each need's chain, at any depth: a sub-need ordered before a sibling is pursued first (hoisted) |
| Protection | the needs an achiever listed and met | the states produced by needs pursued earlier, and the states the plan relies on that held from the start (a toggle that works with the hand as it is), unless ordered after an unmet need (then they may be undone and re-achieved) |
| Where a dropped object lands | the tile in front | the same; a drop onto a protected tile is refused, and the drop's facing need skips placements facing one |
| Everything else | version 16 | unchanged |

What the agent learns stays in recall: that a pick up needs the held
part empty, that a toggle opens this door with this key. Nothing about
hands is written in; the order is derived from those conditions each
time it plans. Expected order for the looping layout: drop the key, pick
up the blocker, put it down off the protected tile, pick up the key,
open the door.

## 3. Dependencies

Version 16 (card 072); card 029's means-ends search and protection; card
051's chain, threats and kept choices (`tools/card051/`); card 043's
conditions in produced situations; card 047's situations. Koehler and
Hoffmann 2000 (LITERATURE.md, High Level Papers).

## 4. Data check

How often the conflict occurs in the evaluation: decoy layouts whose
decoy key blocks the door's front tile or the way to it (card 072: 6 of
100 seeds, the same in every fold); tier 2, every episode (a ball in
front of the door, by construction); tier 1, none (one key).

## 5. Feasibility gate

- **Upper bound, measured in the test world** (LESSONS): a breadth-first
  oracle on the simulator's state (position, direction, held object,
  object positions, door) for each of the 100 decoy layouts, run on
  every fold's door hue: how many are solvable, and their shortest
  lengths. Criterion 1's threshold is set from it before the main run.
- **Trivial baseline:** version 16 with the key known: 94%.
- **Smoke test:** seed 1001029 with the key known, traced.

**Gate result** (`runs/073/oracle.json`, `tools/card073/oracle.py`): all
100 layouts are solvable in every fold (the layouts do not depend on the
hue), shortest 17.5 steps on average, at most 26. Criterion 1's threshold
is therefore 99 of 100 per fold. Smoke test: seed 1001029 (version 16
loops for 640 steps) succeeds in 21 steps: the agent picks up the decoy
key blocking the way, puts it down out of the way, fetches the red key
and opens the door.

## 6. Success criteria and prediction

1. **The conflict:** the decoy world with the key known (every opening
   in memory; all six folds' door hues): success on at least 99% of the
   layouts the oracle solves, against version 16's 94%.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than version 16
   (2%); tier 3 cannot be built (counted as no different).
3. **No regression in transfer:** card 072's six decoy folds (trying),
   no fold worse than version 16 (McNemar, p < 0.05).

Reported: orders found per episode and the depth they were found at;
hoisted sub-needs; protection refusals; where drops land; time per step;
tier 2's failures traced to their first cause.

**Prediction.** Criterion 1 holds. Tier 2 rises above 2%, but other
causes may stop it next (card 066: recall predicting that a locked door
opens while a ball is held; the final pick up needing the key dropped).
Time per step rises, since each need's situation is imagined.

**Decision rules.** Keep if 1–3 hold. Revise once if 1 fails for a reason
in how orders are derived or protected. Stop if tier 1 falls below 99%.

**Budget.** Gate a few minutes; decoy world with the key known 6 × ~70 s;
folds 6 × ~70 s; tier 1 a minute; tier 2 about 5 minutes.

## 7. Result

`runs/073/` (`run.sh`; `tools/card073/order.py` on version 16, switched
on by `WM_ORDER=1`; numbers by `tools/card073/summary.py`).

| | Version 16 | This card |
|---|---|---|
| Decoy world, key known, each fold's door hue (criterion 1) | 94 of 100 in every fold | **99 of 100 in every fold** (needed 99) |
| Tier 1 (criterion 2) | 100%, 20.8 steps | **100%**, 20.8 steps, no random action |
| Tier 2 (criterion 2) | 2% | **25%** (61 steps when successful; 22–35 in most); 25 solved only with the order, 2 only without (McNemar p = 6 × 10⁻⁶): better |
| Decoy folds with trying (criterion 3) | 95–98% | 94–96%; per fold 0–2 solved only with the order, 2–3 only without (p ≥ 0.25 in every fold): no different |
| Time per step, tier 2 | 0.041 s | 0.12 s; 6 episodes stopped at 5 minutes (version 16: 0) |

- **All three criteria met.** In the key-known world the order was
  weighed about 2,000–2,500 times per 100 episodes, gave an order about
  350 times, always against card 029's fixed order (never a cycle), and
  refused 11–14 drops onto a tile a need above relied on.
- **What still fails, traced:**
  - *A route's tiles are not protected* (seed 1001002, every fold; it
    succeeds in version 16). The decoy key and the red key lie one behind
    the other in a corridor. The order has the decoy picked up first;
    holding it, the red key's plan needs an empty hand, and the decoy is
    put back on the tile just cleared. Card 029's check refuses a drop
    only when it cuts a way that works now, and with the red key still in
    the corridor none does. The tiles a planned route crosses are not
    among the states the plan relies on.
  - *A need on the view can also name the held tile* (tier 2, 47 of 69
    failures within time explore throughout; seed 1002000 traced). Recall
    states the door's need as "the view changed with the blue key held"
    (card 066's "a key of a colour in view" conditions), and this card
    reads that need as fixing the view only, so it misses that it also
    fixes the hand: the key is fetched, dropped for the ball, fetched
    again.
  - *The door's prediction* (tier 2, 14 failures mostly random; seed
    1002008 traced). The order works (the ball moved aside, the key
    fetched), but recall does not predict that toggling opens the green
    door, and trying spends its tries on ways such as "pick up the door"
    (card 068's revision).
  - *Trying gives up a hypothesis after one step without a plan* (decoy
    folds with trying, seeds 1001029 and 1001080 in every fold, solved
    with the key known). Clearing the blocker under the hypothesis meets
    a refused drop; exploration has no retry after a refusal, so the right
    hypothesis is marked tried and never retried (card 072's component).

## 8. Decision

**Keep** (criteria 1–3 met): version 17, version 16 with needs ordered by
the states they conflict over. The first two failures above are in this
card's component (which parts a need fixes; which states a route relies
on) and are the natural next card; the other two belong to card 068's
revision and to card 072's trying.
