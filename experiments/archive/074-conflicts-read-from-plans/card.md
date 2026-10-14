---
id: "074"
title: conflicts and orders read from plans, not from a list of parts
rung: 6
serves: [P21, P12, P14, P17]
status: done
verdict: fail
arch_version: 17
date: 2026-10-06
---

# 074: conflicts and orders read from plans

**Definition** (card 073, from the user):

> If need A requires a state that need B destroys, and A's result
> survives B, then **A comes before B**.

Koehler and Hoffmann (2000, Definition 8): "B ≤r A [means] that if,
after the goal A has been achieved, there is no plan anymore that
achieves B without — at least temporarily — destroying A, then B is a
goal prior to A." This card takes "there is no plan" literally: the
agent's own planner answers it.

## 1. Question

Card 073 decided conflicts with a hand-made list of parts (the hand, the
view, a tile) and protected only the states the planner had written down
as needs. Two failures came from that: a need phrased by recall as "the
view changed, with the key held" was read as touching the view only
(tier 2, 47 of 69 failures within time), and the tiles a later route
must cross were not states at all (decoy seed 1001002). If conflicts are
instead read from plans, with no list of parts, do both cases follow
from the one rule, with card 073's gains kept? And, stored as experience,
do the conflicts it finds let recall predict orders on layouts it has not
seen (the user, 2026-10-06: "could we integrate the remembered conflicts
as part of 074?")? The user: "the state conflict principle should work
generally across environments, states, scenarios". P21, P12, P14, P17,
P10.

## 2. What changes

One component: how the planner decides orders and protection (card
073's).

| | Version 17 (card 073) | This card |
|---|---|---|
| Does B come before A? | B, or every way to B, touches a part A fixes (a list: hand, view, tile) | **by planning**: in the situation where A holds, the planner finds no plan for B with A protected, but finds one when A may be undone |
| The situation where A holds | A's own achiever's effect on the present | the effects of A's chain's interaction acts (pick up, toggle, drop), by recall, in order from the present; no moves imagined |
| What is protected | met needs and "works with the hand as it is" | everything a pending need relies on, tested the same way: an act is refused when, in the situation it produces, a pending need that could be planned with the protected states kept no longer can be; the planner then takes another way (card 029's refused set: another place, another achiever) |
| Part lists (`parts_fixed`, `part_of`) | used | removed |
| Where B is looked for | anywhere in a sibling's chain (kept) | the same |
| Conflicts remembered | none | each weighing is stored as a try of an "order" kind: its key the two conditions (their action, and the vectors of the tiles they name, the held tile included), its outcome "B before A", "A before B" or "no order"; recall (cards 049–051's admitted conditions, own tries first) predicts the outcome for a new pair. **Here the derivation decides; recall's prediction is recorded beside it** |

Case 1 under this rule: putting the decoy down on the corridor tile
leaves "face the door" plannable only by picking the decoy up again,
which undoes "red key in hand"; another floor tile does not, so that
drop is refused and the decoy goes elsewhere. No tile, corridor or
route is named. Case 2: "the view changed, with the key held" fails the
same test against clearing the ball, whatever form recall gives it.

**Where remembered conflicts come from.** Memory of orders is built on
training layouts (tiers 1 and 2 and the decoy world, seeds apart from the
test seeds), then fixed while the test layouts run, like the rest of
memory; within an episode, new weighings are added and dropped at its
end (the user: memory starts afresh each episode). Recall decides orders
only in a later card, and only where it agrees with the derivation on
every case (CHARTER: weights and recall take over from the slower path
only so).

Walking enters only through the planner's own answer to "can this be
reached?", so this card holds whichever walking answers (the field
today; card 075's learned System 1 if kept). Literature: Koehler and
Hoffmann 2000 (`1106_0243`); causal links and threats in partial-order
planning (McAllester and Rosenblitt 1991, SNLP, not in papi), which card
051 drew on.

## 3. Dependencies

Version 17 (cards 072, 073); card 029's means-ends search and refusal;
card 051's kept choices; card 043's conditions in produced situations;
card 067's walking. Koehler and Hoffmann 2000 (LITERATURE.md).

## 4. Data check

In version 17: decoy world with the key known, 1 layout of 100 fails
(seed 1001002, every fold); tier 2, 47 episodes explore throughout in a
pick-up-and-drop loop, 14 end mostly random, 8 other, 6 out of time.

## 5. Feasibility gate

- **Upper bound:** card 073's oracle, all 100 decoy layouts solvable.
  For tier 2, the same breadth-first oracle on its 100 layouts (ball,
  key, door, box), run before the main run.
- **Trivial baseline:** version 17.
- **Smoke test:** decoy seed 1001002 and tier 2 seed 1002000, traced.
- **Cost check:** each test plans B twice; time per step on 10 tier 2
  episodes before the main run. If it is above 3 × version 17's (0.12 s),
  stop here and report.

**Gate result.** Tier 2's oracle (`runs/074/oracle_tier2.json`,
`tools/card074/oracle.py`): all 100 layouts solvable, shortest 22.2 steps
on average, at most 29. Cost check, 10 tier 2 episodes: 0.013 s per step
(version 17, the same seeds: 0.079 s). Smoke tests: decoy seed 1001002
succeeds in 32 steps, walking round the decoy (version 17 loops); tier 2
seed 1002000 succeeds in 261 steps (ball moved aside, key fetched, door
opened; then a long search to free the hand for the box, recall's
conditions for that pick up). Found on the way, and built in: a test plan
must not run tests of its own (Definition 8 asks only whether a plan
exists, so inside a test needs are ordered as in version 16), and an
order chosen for an achiever is kept between steps while it still gives a
plan (tier 2 seed 1002000 turned back and forth: recall's toggle
conditions read what is in view, so a test can turn with the agent).

## 6. Success criteria and prediction

1. **The two cases:** decoy world with the key known, 100 of 100 in every
   fold (seed 1001002 solved); and of tier 2's 47 looping episodes, at
   least half no longer loop (they succeed, or fail for another first
   cause, traced).
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than version 17
   (25%); tier 3 cannot be built.
3. **Cost and transfer:** tier 2 episodes out of time not more than
   version 17's 6; card 072's decoy folds with trying no fold worse
   (McNemar, p < 0.05).

Reported: orders found, refusals and by which pending need, planning
calls per step, time per step, tier 2's remaining failures by first
cause. **Remembered conflicts:** how many weighings memory holds after
training; on the test layouts, how often recall's predicted order agrees
with the derivation, overall and by class (by condition kinds; seen and
unseen tile colours; tiers 1, 2 and the decoy world), and the planning
time it would save. The card that lets recall decide is drafted only if
agreement is at least 99% overall and in every class.

**Prediction.** Criterion 1 holds; tier 2 rises (most of its failures are
case 2). Recall's orders agree with the derivation on the conflicts seen
in training (the same hand, other colours), less on kinds of conflict
training rarely shows. The risk is cost: planning B twice at every weighing may push
more tier 2 episodes past five minutes; memoised plans per situation
should keep it near 2 ×.

**Decision rules.** Keep if 1–3 hold. Revise once if 1 fails for a reason
in how plans are read; revise for cost if only 3 fails. Stop if tier 1
falls below 99%.

**Budget.** Oracle a few minutes; training layouts for the memory of
orders about 10 minutes; decoy world 6 × ~70 s; folds 6 × ~70 s; tier 1
a minute; tier 2 about 8 minutes.

## 7. Result

`runs/074/` (`run.sh`; `tools/card074/conflicts.py` on version 17 in
place of card 073's order, `WM_CONFLICTS=1`).

| | Version 17 | This card |
|---|---|---|
| Decoy world, key known (criterion 1) | 99 of 100 per fold | **94 of 100** per fold: version 16's six loop layouts fail again; seed 1001002 solved |
| Tier 2's 47 looping episodes (criterion 1) | — | 28 no longer loop (19 solved, 9 fail otherwise): met |
| Tier 1 (criterion 2) | 100%, 20.8 steps | 100%, 20.8 steps |
| Tier 2 (criterion 2) | 25% | 21%, 102 steps when successful; 21 solved only here, 25 only in version 17 (McNemar p = 0.66): no different |
| Tier 2 out of time (criterion 3) | 6 | **1**; 0.062 s per step (version 17: 0.12) |
| Decoy folds with trying (criterion 3) | 94–96% | 96–98%; no fold worse |

- **Criterion 1: not met; 2 and 3: met.**
- **Why criterion 1 fails** (decoy seed 1001029, traced): the order kept
  between steps overrode the test. In that episode the test found
  "clear the decoy first" on 318 steps, and on all 318 an order kept from
  an earlier, undecided step replaced it; the agent picks the key up and
  puts it down for 640 steps. The 25 tier 2 episodes version 17 solves and
  this card does not are the same loop (about 330 overrides each). Keeping
  a choice should only break ties the test leaves open.
- **Remembered conflicts** (`runs/074/recall_orders.json`,
  `tools/card074/recall_orders.py`; report only). Memory of 2,654
  distinct weighings from the training layouts. On the test layouts
  recall agreed with the derivation on 95.1% of 17,190 weighings, below
  always answering "no order" (97.7%), and predicted 15% of the 394 real
  orders (own tries 18%, similar tries 0%). An order depends on the
  situation (where the blocker stands), and the stored key holds only the
  tiles the two conditions name. The report also grouped walking needs by
  their tile, not their action (a defect in the scorer). Recall of orders
  needs the situation in its key before it can be scored again.
  Simplification declared: recall here is own tries first and similar
  tries by one fitted width, not cards 049–051's admission.

## 8. Decision

**Revise** (criterion 1 failed for a reason in how orders are chosen):
[card 074.1](../074.1-orders-with-ties-and-split-needs/card.md), with the
user's scope (2026-10-06): an order the test finds is never overridden;
when no order is found, the last step's choice while it still gives a
plan, else the cheaper need (cost changes with every step the agent
takes, so cost first would turn with the agent as the kept order did not);
and two-part needs split into independent conditions, checked in the
situation the earlier one produces, removing card 043's splice.
