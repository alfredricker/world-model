---
id: "076"
title: a learned router from the situation's tokens to remembered orders
rung: 6
serves: [P10, P21, P12, P17]
status: done
verdict: fail
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

**Data check.** The first training layouts held 136 real orders (tier 2
124, decoy world 12, tier 1 none), under 300; 80 more tier 2 layouts
(seeds 502100–502179) brought training to 11,535 weighings and 401
orders (tier 2 389, decoy world 12) in 280 layouts. Test: 21,363
weighings, 402 orders. Collecting changed no decision: every test
episode took card 074.2's actions (two tier 2 episodes were cut by the
time limit at different steps, with the same actions before).

**Gate result.**
- Identical inputs (the same query and the same code at every offset)
  give the same outcome for 99.79% of training weighings: the tokens
  nearly determine the planner's order.
- The router fitted and scored on the training weighings, no layout
  left out: **98.63%** (always "no order": 96.52%; 93% of orders caught
  at 74% precision). **Below the 99% gate.**
- Cost: 0.07 ms per prediction (a plan test: about 0.26 s in card
  074.1's slowest episode).

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

## 7. Result

`runs/076/` (`run.sh`; `tools/card076/collect.py` with `WM_COLLECT=1`;
`tools/card076/router.py` → `router.json`). Agreement with the planner
on the test layouts; orders caught and the precision of predicted
orders.

| | Always "no order" | Card 074's recall | Hand-built recall | Router |
|---|---|---|---|---|
| All (21,363; 402 orders) | 98.12% | 97.89%; 1% caught | 97.74%; 16% caught at 31% | **97.45%**; 57% caught at 38% |
| Tier 2 (8,525; 326) | 96.18% | 95.74%; 2% | 95.51%; 19% at 34% | **97.72%** (better, p < 0.001); 69% at 71% |
| Decoy, key known (4,828; 26) | 99.46% | 99.40%; 0% | 99.30%; 0% | 96.71%; 12% at 2% |
| Decoy, trying (6,717; 50) | 99.26% | 99.14%; 0% | 99.03%; 0% | 97.14%; 6% at 2% |
| Tier 1 (1,293; 0) | 100% | 100% | 100% | 100% |
| Unseen arrangement (211; 17) | 91.94% | 91.94%; 0% | 91.94%; 0% | 96.21% (p = 0.049); 76% at 76% |

- **Criterion 1** (better than "no order" overall and in each source):
  **not met**. Better in tier 2; worse overall and in both decoy sources
  (it predicts orders there that the planner does not find).
- **Criterion 2** (80% caught at 80% precision): **not met**: 57% at
  38% overall; 69% at 71% in tier 2.
- **Criterion 3:** colour transfer could not be measured (every code
  the test needs named also appears in training); on arrangements
  training never showed, 76% caught at 76% (17 orders). Not at least as
  good as the hand-built recall in every source: worse in the decoy
  world. **Not met.**
- **Why the decoy world fails.** Its orders are the same kind as tier
  2's ("hold the key for the toggle" waits on "face the door": 25 of 26
  with the key known), a kind training saw 63 times, at least 51 of them
  in tier 2 (the decoy world gave 12 orders in all), where a ball blocks the door. With a second key blocking
  instead, the router did not carry it, and it predicts orders on decoy
  situations the planner leaves unordered (136 false orders with the key
  known).
- **P19's curve** (training layouts added until it holds 10, 30, 100,
  300 orders): 15%, 19%, 33%, 59% of the test's orders caught, at
  18–37% precision; agreement 97.1–97.7%, below "no order" throughout.

## 8. Decision

**Stop** (the decision rule: the upper bound failed, 98.63% against
99%). The two gate measures disagree, which matters for what comes
next: identical token situations agree with each other on 99.79% of
weighings, so the situation as tokens does carry the order (card 074's
key did not); what fell short is the router's fit, trained mostly on
tier 2 (389 of 401 orders) and weighted toward catching orders at the
cost of agreement. In tier 2 it already beats "no order" (97.7% against
96.2%), and the hand-built and card 074 recalls do not. A new card, for
the user to decide, would need training orders from every world in
proportion, a gate on the inputs' own consistency, and a fit that does
not trade agreement for recall of orders.
