---
id: "077"
title: a router that reads what tokens do and how they relate, not how they look
rung: 6
serves: [P3, P10, P21, P12]
status: done
verdict: fail
arch_version: 18
date: 2026-10-06
---

# 077: a router that reads roles and relations, not appearance

Agreed with the user (2026-10-06, "let's try this out"): tokens described
by what they do; appearance entering only through relations between
tokens; one relation module, the one recall already uses for "this key
fits this door". This card is the first of two: here the relation is
card 070's, frozen; training it on effects and orders together is a
later card, run only if this one shows the relation path carries.

## 1. Question

Card 076's router, which read each token's appearance, learned orders in
tier 2 (69% caught at 71% precision; "no order" beaten, 97.7% against
96.2%) but did not carry them to the decoy world, whose orders are the
same kind ("hold the key" waits on "face the door") with a second key
blocking instead of a ball: 12% caught at 2% precision. If the router
reads instead what each token does (recall's own predictions) and how
tokens relate (card 070's projection of the difference between two
tokens' vectors, plus their offset), does an order learned where a ball
blocks carry to where a key blocks? P3 (roles apart from the things
filling them), P10, P21, P12.

## 2. What changes

One component: what the router reads. Its form, fit and vote are card
076's.

| Per token | Card 076 | This card |
|---|---|---|
| Appearance | its 32 encoder numbers | **none read directly** |
| What it does | — | recall's predictions, empty-handed, from a start situation of its world: forward moves onto it; forward onto it ends the episode; a pick up changes it; a toggle changes it |
| Relations | the named tokens' vectors, as context | **\|P(z_i − z_j)\|** (8 numbers, card 070's P, frozen) between the token and each named token and the hand, and the offsets between them |
| Where, marks, the query | as card 076 | the same |

The two needs' named tokens enter only through their relations to the
others and their own roles, so "a ball" and "a key" are the same where
they do the same things and stand in the same relations. Literature:
Webb et al., the relational bottleneck
(`the-relational-bottleneck-as-an-inductive-bias-for-efficient`),
relations as the only path from objects to the decision; Battaglia et
al. 2018 (`1806_01261`), relations as edges with relative position;
card 070 (a relation-only path carried to 9 of 9 unseen hues); LESSONS,
Generalisation.

## 3. Dependencies

Card 076's collected weighings (`runs/076/`; collecting changed no
decision), its router form, fit and scorer; card 070's encoder and P;
version 18's recall for what each tile does.

## 4. Data check

From card 076: tier 2 training holds 389 orders in 10,317 weighings;
the decoy world's test holds 76 orders in 11,545 weighings (key known
and trying). If what-it-does gives a ball and a key different profiles
in their worlds, the transfer cannot work through roles alone; reported
before the main run.

## 5. Feasibility gate

- **Inputs:** identical inputs as this router reads them (the query and,
  per token, its role, marks, offset and relations rounded to 0.1) give
  the same outcome on at least 99% of the training weighings. Below
  that, dropping appearance lost what orders depend on, and the card
  stops.
- **Baselines:** always "no order"; card 076's router (appearance) on
  the same splits.

**Data check.** Recall, empty-handed in each world, gives every ball and
every key the same profile (picked up, not walked onto, not toggled), in
tier 2 and in the decoy world (`runs/077/affordances_*.json`); roles alone
do not rule the transfer out. (It also predicts some open doors cannot be
walked onto: card 068's known limit.)

**Gate result.** Identical inputs as this router reads them give the
same outcome on **99.79%** of training weighings (11,375 distinct):
dropping appearance lost nothing orders depend on. Passed.

## 6. Success criteria and prediction

1. **Transfer** (the question). Trained on tier 2 only, scored on the
   decoy world (key known and trying): at least 50% of its orders caught
   at at least 50% precision; agreement not worse than always "no
   order" (McNemar, p < 0.05); and more orders caught than card 076's
   router on the same split.
2. **Within worlds** (card 076's protocol, all training): better than
   "no order" overall and in every source that has orders (McNemar, p <
   0.05).
3. **No loss in tier 2:** tier 2 agreement not worse than card 076's
   router (McNemar, p < 0.05).

Also reported: ablations (roles without relations; relations without
roles), P19's curve, time per prediction.

**Prediction.** A ball and a key share a profile (picked up, not walked
onto, not toggled) and the same relations to the door (in front of it),
so most decoy orders are caught. The risk is false orders: decoy
situations where the key lies elsewhere look like tier 2's only through
offsets the router may weigh loosely.

**Decision rules.** Keep (the relational router is what the shared
relation card builds on) if 1–3 hold. Revise if 1 holds and 2 or 3 does
not. Stop if the gate or criterion 1 fails.

**Budget.** What each tile does: three world setups, about 6 minutes.
Fits: about 30 s each, about 10 fits; scoring a few minutes. No agent
runs.

## 7. Result

`runs/077/` (`tools/card077/affordances.py`, `tools/card077/relational.py`
→ `relational.json`), on card 076's weighings.

**Transfer** (trained on tier 2: 10,317 weighings, 389 orders; scored on
the decoy world: 11,545 weighings, 76 orders):

| Arm | Agreement (always "no order" 99.34%) | Orders caught | Precision |
|---|---|---|---|
| Relational (roles and relations) | 95.89% | 16% | 3% |
| Appearance (card 076's router) | 95.32% | 17% | 3% |
| Roles only | 97.13% | 18% | 5% |
| Relations only | 97.34% | 8% | 2% |

**Within worlds** (card 076's protocol): relational 97.51% overall
(always "no order" 98.12%); tier 2 96.88% against 96.18%, 56% of orders
caught at 60%; decoy world 0 of 76 orders caught. Appearance: 97.15%
overall, tier 2 97.16%. Tier 2, relational against appearance: 105
weighings right only with relations, 129 only with appearance (p = 0.13).

- **Criterion 1 (transfer): not met.** No arm catches more than 18% of
  the decoy world's orders, and every arm is worse than answering "no
  order" there; the relational router is slightly more often right than
  the appearance router (264 against 199 weighings, p = 0.003) but
  catches no more orders.
- **Criterion 2: not met** (worse than "no order" overall and in the
  decoy world). **Criterion 3: met** (tier 2 no different from the
  appearance router).
- **What this rules out.** Appearance was not what stopped the transfer:
  reading roles and relations instead changes little. Nor is it a
  missing contrast: tier 2 has orders and non-orders for the same pair
  of needs (11% orders for "hold the key" with "face the door"; decoy
  world 4%).
- **What separates an order from a non-order is not found.** In both
  worlds most orders have a carriable tile next to the door (tier 2 41
  of 52; decoy 64 of 69), but so do many non-orders (63 and 177); the
  hand's contents and the agent's distance to the door do not separate
  them either. A hypothesis, not tested: an order turns on whether the
  tile stands on every way from the agent to the door, a relation about
  routes that no relation between two tokens states, and that the
  planner gets from walking.
- Cost: 0.1 ms per prediction.

## 8. Decision

**Stop** (criterion 1 failed). Describing tokens by what they do and
relating them through card 070's projection keeps everything orders
depend on (the gate) and costs nothing in tier 2, but does not carry an
order from one world to another. The shared relation card is not drafted
on this result. The candidate the result points to is a relation about
routes, from the agent's own walking (card 067 already finds the first
token a least route crosses); it needs its own card and the user's
go-ahead.
