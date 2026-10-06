---
id: "078"
title: the route relation as the router's input
rung: 6
serves: [P3, P10, P12]
status: done
verdict: fail
arch_version: 18
date: 2026-10-06
---

# 078: "this tile stands on the way" as the router's input

A quick check agreed with the user (2026-10-06: "unless the route
relation is a quick check"), on card 076's weighings, with no agent
runs.

## 1. Question

Card 077's router, reading what tokens do and card 070's relations, did
not carry orders from tier 2 to the decoy world (16% caught at 3%
precision). Its hypothesis: an order turns on whether a tile stands on
the way from the agent to the need's target, a relation about routes
that no relation between two tokens states. With that relation added to
each token, does the order carry? P3, P10, P12.

## 2. What changes

One input added to card 077's relational router: per token, **on the
way**: making it walkable shortens, or opens, the agent's way to a tile
next to the achiever's target. Computed on the agent's belief only: the
seen tokens of the weighing (their offsets), walkable as recall predicts
(card 077's roles), breadth-first over the four neighbours from where the
agent stands. It is the grid form of walking's "first token a least route
crosses" (card 067): headings are ignored and only tokens within the
13 × 13 window are known (declared simplifications).

## 3. Dependencies

Cards 076 and 077 (weighings, roles, router, scorer).

## 4. Data check

How often a token is on the way, in orders and non-orders of tier 2 and
the decoy world.

## 5. Feasibility gate

Identical inputs (card 077's, plus on the way) give the same outcome on
at least 99% of training weighings.

## 6. Success criteria and prediction

1. **Transfer** (card 077's criterion 1): trained on tier 2 only, on the
   decoy world at least 50% of orders caught at at least 50% precision,
   agreement not worse than always "no order", and more orders caught
   than card 077's relational router.

Also reported: the within-worlds protocol, and the route relation alone
(roles, marks, offsets and on the way, no card 070 relations).

**Prediction.** Most decoy orders have the decoy on the way and most
non-orders do not, so the order carries.

**Decision rules.** Keep (the route relation joins the router's inputs)
if 1 holds; stop otherwise.

**Budget.** About 10 minutes.

## 7. Result

`runs/078/` (`tools/card078/route.py` → `route.json`), on card 076's
weighings; about 2.5 minutes.

**Data check** (some token on the way): decoy world orders 93% (of 76),
non-orders 9.5% (of 11,469); tier 2 training orders 61% (of 389),
non-orders 13% (of 9,928). **Gate:** identical inputs agree on 99.79%.

| Trained on tier 2, scored on the decoy world | Agreement ("no order" 99.34%) | Orders caught | Precision |
|---|---|---|---|
| Route (card 077's router plus on the way) | 98.35% | **41%** | 18% |
| Route only (no card 070 relations) | 98.66% | 39% | 22% |
| Card 077's relational router (refitted) | 97.54% | 5% | 2% |

Within worlds (card 076's protocol): route 97.96% overall against 98.12%
for "no order" (p = 0.16, no different; card 077's relational 97.54%,
worse); decoy world with the key known 50% of orders caught at 20%;
tier 2 71% at 58% (97.0% against 96.2%).

- **Criterion 1: not met** (41% caught at 18% precision; below "no
  order" in agreement).
- **The route relation is what orders turn on:** present in 93% of the
  decoy world's orders and 9.5% of its non-orders, and the one input so
  far with which an order learned in tier 2 carries to the decoy world
  at all (41% against at most 18% in card 077). Card 070's relations add
  nothing on top of it.
- **Fit noise.** The same relational arm caught 16% in card 077 and 5%
  here (one fit each, different random states): differences of about
  ten points between single fits are within noise; 41% against 5–18% is
  not.

## 8. Decision

**Stop** (criterion 1). The route relation separates orders from
non-orders far better than appearance, roles or pairwise relations, and
is the first input that transfers, but not to the declared bar:
precision stays low because a tile on the way is common where no order
is found (9.5% of decoy non-orders against 1 order in 150). Orders stay
the planner's; the user's choice (2026-10-06) is to take the relational
router to effect recall next.
