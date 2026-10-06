---
id: "076"
title: a learned router from the situation's tokens to remembered orders
rung: 6
serves: [P10, P21, P12, P17]
status: draft
verdict:
arch_version: 18
date: 2026-10-06
---

# 076: a learned router from the situation's tokens to remembered orders

The first card of the user's note in STATUS.md (2026-10-06): recall
should work with everything as a token, a what and a where, through a
network that learns to route the situation to past events, instead of a
distance between three separate parts (the tile ahead, the hand, the
view).

## 1. Question

Card 074 stored each order the planner worked out (a "weighing" of two
needs: must B come before A?) and scored recall over those stored
weighings beside the planner, on layouts it had not seen. Recall agreed
with the planner on 95.1% of 17,190 weighings, below always answering
"no order" (97.7%), and caught 15% of the 394 real orders. Its key held
the two needs and the tiles they name, positions dropped (card 044's
declared exception), and an order depends on where things are: the ball
must be cleared first when it stands in front of the door, not when it
stands in a corner. With the whole situation as tokens and a learned
router choosing which stored weighings it should reach, does recall
predict the planner's orders on unseen layouts? Orders are the test bed:
the data exists, the failure is measured, and the agent does not change.
P10 (retrieval that helps where the current key cannot), P21 (a fast
path for what the planner derives slowly), P12, P17.

## 2. What changes

One component: recall's metric for stored weighings. The planner (version
18, card 074.2) still decides every order; recall is scored beside it, report only.

| | Card 074 | This card |
|---|---|---|
| A stored weighing | the two needs' kinds and actions; the vectors of the tiles they name and of the held tile | the same needs, and **the situation as tokens**: every token's what (its 32 encoder numbers) and where (its offset from A's target, the tile A's achiever acts on; the hand as a token of its own), marked by whether a need names it |
| How two weighings are compared | equal codes, else one fitted width over three vectors | equal codes first (own tries, cards 042 and 050), then a **learned router**: a small shared network maps each token to a vector, relations between token pairs enter as their offset, and the sum over tokens (Deep Sets) is the situation's embedding e; a stored weighing counts by exp(−‖w ⊙ (e_query − e_stored)‖₁) |
| Fitted by | leave-one-out likelihood, one width | leave-one-out likelihood of the stored outcomes, leaving out **whole layouts** (card 071: twins of one layout make the fit favour identity), with real orders weighted by their rarity |

Why a distance on e, and not a dot product between a query and a key:
a learned dot-product score between two things' vectors fitted three
colours and carried to none, while a distance along learned weights
carried (LESSONS, Generalisation; cards 069–070). Why codes first:
the planner needs crisp identity, and the two levels must be mixed, not
switched (cards 042–047).

**CHARTER.** "Recall is a learned metric on [the encoder's vectors], one
per action" (Current direction). The router is read as that metric,
extended from weights per number to a small network over the tokens; it
reads the same encoder and adds no output head or second space. "A new
comparison ... does not add a side network (C2)": if the user reads the
router as a side network, this card needs a CHARTER amendment first.
"Recall reads three roles ... positions dropped" (card 044's declared
exception) is what this card begins to remove, for orders only.

**Baselines.** Always "no order"; card 074's recall; and a hand-built
recall (this card's first draft): card 049's admission over chosen
candidates, the offset between the two named tiles and the code tuples
next to each. LESSONS: a hand-built mechanism can be a baseline, never a
result.

Literature: Pritzel et al. 2017 (`1703_01988`) and Blundell et al. 2016
(`1606_04460`), recall as a weighted vote over stored experiences in a
learned embedding; Zaheer et al. 2017 (`1703_06114`), sums over a
set's elements as the form of a function of a set; Battaglia et al.
2018 (`1806_01261`), relations as edges carrying the pair's relative
position; Goyal et al. 2022 (`2202_08417`), a retrieval process learned
with the agent.

## 3. Dependencies

Card 074.2's planner (its orders are the truth recall is scored
against); card 074's weighings and scorer (fixed: walking needs grouped
by action, not tile); card 044's tokens (every tile a what and a
where); cards 049–050's admission, own tries first and leave-one-out
fit; card 070's encoder. Established methods above.

## 4. Data check

Weighings with their token situations on training layouts (seeds from
500,000: tier 1 60 episodes, tier 2 40, decoy world 100, as card 074).
Report: weighings, real orders, distinct layouts, and orders per source.
If the real orders number under 300, more training layouts first.

## 5. Feasibility gate

- **Upper bound:** the router fitted and scored on the training
  weighings (no layout left out): can these inputs decide orders at all?
  Below 99% agreement there, the inputs miss something and the card
  stops.
- **Trivial baselines:** always "no order"; card 074's recall.
- **Cost:** time per prediction against a plan test (card 074.1: about
  0.05 s per test in tier 2).

## 6. Success criteria and prediction

On the test layouts (the seeds of cards 073–074.2), report only:

1. **Better than no order:** agreement above always answering "no
   order", overall and in each source (tier 1, tier 2, decoy world with
   the key known, with trying); McNemar, p < 0.05.
2. **Real orders caught:** at least 80% of the planner's orders
   predicted, at least 80% of predicted orders real.
3. **Transfer:** criterion 2 on weighings whose colours or arrangement
   training never showed; and the router at least as good as the
   hand-built recall on every source.

Also reported: agreement against the number of real orders in training
(10, 30, 100, 300; P19's curve); time per prediction; which tokens the
router weighs most in a few traced orders.

**Prediction.** Orders that turn on what stands next to A's target
(tier 2's ball before the door) are caught; orders that turn on a route
several tiles away are missed more often, since a sum over tokens keeps
weak pair relations. The hand-built recall does as well on tier 2 and
worse on the decoy world.

**Decision rules.** Keep if 1–3 hold: the next card of the set lets the
router decide orders where it agrees with the planner on every case
(CHARTER: recall and weights take over only so). Revise if only 3
fails. Stop if the upper bound fails.

**The set** (the user's note; each its own card, drafted with the
user): this card (orders); the router in place of pick up, toggle and
drop recall's three roles; forgetting stored tries the router never
reaches; actions straight from the router (System 1).

**Budget.** Collection about 15 minutes; fitting a few minutes on the
GPU per arm; scoring a few minutes. No tier runs: the agent does not
change.
