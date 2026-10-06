---
id: "074.1"
title: found orders always followed, ties kept, and two-part needs split
rung: 6
serves: [P21, P12, P14, P17]
status: approved
verdict:
arch_version: 17
date: 2026-10-06
---

# 074.1: found orders followed, ties kept, two-part needs split

Card 074's one revision. Its definition stands (Koehler and Hoffmann
2000, Definition 8): need B comes before need A when, in the situation
where A holds, the agent's planner finds no plan for B that keeps A, but
finds one that undoes it.

## 1. Question

Card 074 read orders from plans and fixed what card 073's list of parts
could not: decoy seed 1001002 (the decoy is no longer put back in the
corridor) and 28 of tier 2's 47 pick-up-and-drop loops. It missed its
first criterion, 94 of 100 in the decoy world with the key known, for one
traced reason. An order kept from the last step overrode the order the
test found (seed 1001029: 318 times in 318), and the same override
undid 25 tier 2 episodes version 17 solves.

The needs it orders are also partly made up. When a stored success needs
both the hand and the view changed, card 043 states two needs, each
checked with the other part fixed to the stored try's value (its declared
exception). Tier 2's door gives "the view changed, with the blue key
held": a situation where the key is held while it still lies in view.
The user (2026-10-06): such needs should be "two independent conditions",
ordered by card 074's rule, the second checked in the situation the first
produces, and the "weird splicing patch" abandoned.

With orders the test finds always followed, ties broken by the last
step's choice, and two-part needs split, does card 074's rule meet its
own criteria, with no splice left? P21, P12, P14, P17.

## 2. What changes

One component, the ordering of an achiever's needs, in two parts: how
ties are broken, and how a two-part success is stated as needs to order.
The second reaches into card 043's conditions. The user agreed to this
scope (2026-10-06), and the run separates the two parts (arms A and B
below).

| | Card 074 | This card |
|---|---|---|
| An order the test finds | overridden by the order kept from the last step | **always followed** |
| No order found (a tie) | the kept order, else card 029's order | the last step's choice, while its need is unmet and still gives a plan; else **the cheaper need**: fewer pick ups, toggles and drops in its plan, then fewer steps to the first. Cost changes with every step the agent takes, so it decides only fresh choices |
| A success needing both the hand and the view changed | two spliced needs: the hand changed, with the stored view; the view changed, with the stored hand | **two independent conditions**: "the hand holds h" (the planner's existing `has` condition, card 056; h the stored try's held tile; achieved by any act whose learned effect puts h in the hand), and card 043's one-part view need, checked with the situation's own hand and view. Both are ordered with the achiever's other needs by card 074's rule, so a view need pursued after the hand is checked in the situation the hand's achiever produces |
| Remembered conflicts | recorded, scored report-only | not run; card 074 showed their key needs the situation, a later card |

How the two traced cases should go:
- **Seed 1001029.** The test finds "clear the decoy first", and it is
  followed.
- **Tier 2 seed 1002000.** The door's needs become "hold the blue key",
  "the toggle works with the view as it is" and "face the door". Where
  the key is held, facing the door cannot be planned without putting the
  key down (the ball must be picked up first). So "face the door" comes
  first: the ball is cleared, then the key fetched. The view need is then
  checked where the key is in the hand and no longer on the floor in
  view. If it came only from the splice, it already holds there.

Nothing names a hand, a ball or a door. The hand condition is the
stored success's own held tile, which every world with something to
carry has; the view need is card 043's, unchanged.

In card 074, Recall matched the planner's orders 95% of the time. That is worse than always answering "no order" (98%), and it caught only 15% of the real orders. The reason is that an order depends on where things are, and the stored memory records only the two tiles involved. I've left this out of 074.1; it needs that location information added first, in a later card.

## 3. Dependencies

Card 074 (the order test, protection of what comes after); card 043's
conditions; card 056's `has` condition and its learned achievers; card
051's kept choices; card 029's means-ends search. Koehler and Hoffmann
2000 (`1106_0243`), whose goal agenda orders the parts of a conjunctive
goal as the split does here; STRIPS's conjunctive goals (`strips`).

## 4. Data check

On 10 tier 2 episodes and the decoy world's red fold, key known, before
the main run:
- how many planning steps choose a chain holding a spliced need;
- how often card 074's test finds opposite orders on consecutive steps.
  That would be an order turning with the agent, which keeping a tie
  cannot fix.

If spliced needs are rare in tier 2's failures, arm B is reported but
the decision rests on arm A.

## 5. Feasibility gate

- **Upper bound:** the breadth-first oracles, all 100 decoy layouts
  (card 073) and all 100 tier 2 layouts (card 074) solvable.
- **Trivial baselines:** version 17 and card 074.
- **Smoke test:** decoy seeds 1001029 and 1001002; tier 2 seeds 1002000
  and 1002010, traced, in arm B.
- **Cost check:** time per step on 10 tier 2 episodes. If it is above
  3 × version 17's (0.12 s), stop here and report.

## 6. Success criteria and prediction

Card 074's criteria, unchanged:

1. **The two cases:** decoy world with the key known, 100 of 100 in every
   fold; of version 17's 47 looping tier 2 episodes, at least half no
   longer loop.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than the best
   version (version 17, 25%; McNemar, p < 0.05); tier 3 cannot be built.
3. **Cost and transfer:** tier 2 episodes out of time not more than
   version 17's 6; card 072's decoy folds with trying no fold worse
   (McNemar, p < 0.05).

Two arms:
- **A:** card 074 with the tie rule (the splice kept). Decoy world with
  the key known, and tier 2.
- **B:** A plus the split. Every criterion.

Reported per arm:
- orders found, followed, and ties by how they were broken;
- order flips between consecutive steps;
- spliced needs used (0 in B by construction);
- for B, how often the view need held already when the hand condition
  was met;
- tier 2's remaining failures by first cause, traced.

**Prediction.**
- **The tie rule:** restores card 073's six decoy layouts and the 25 lost
  tier 2 episodes, so criterion 1 holds in A and tier 2 rises above 25%.
- **The split:** tier 2's door view needs vanish once the key is in
  hand. B rises further, by the episodes that still loop in A.
- **The risk:** cost now decides fresh ties in tier 1, where card 029's
  order did. Tier 1 must stay at or above 99%.

**Decision rules.**
- **Keep B** (version 18, card 043's exception removed) if 1–3 hold for B.
- If B fails them and A holds them, **keep A** (version 18 without the
  split), and the split goes to a later card with its trace.
- **Stop** if tier 1 falls below 99% in both arms.

**Budget.** Gate about 10 minutes. Arm A about 15 minutes (decoy world
6 × ~70 s, tier 2 ~8 minutes). Arm B about 25 minutes (decoy world and
folds 12 × ~70 s, tier 1 a minute, tier 2 ~8 minutes).
