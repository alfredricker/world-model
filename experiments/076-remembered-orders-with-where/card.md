---
id: "076"
title: remembered orders that record where things are
rung: 6
serves: [P21, P10, P12, P17]
status: draft
verdict:
arch_version: 17
date: 2026-10-06
---

# 076: remembered orders that record where things are

## 1. Question

Card 074 stored each order the planner worked out (the "weighing" of two
needs: does B have to come before A?) and asked whether recall over those
stored weighings would have predicted the planner's answer on layouts it
had not seen. It would not. Recall agreed with the planner on 95.1% of
17,190 weighings, below always answering "no order" (97.7%), and caught
15% of the 394 real orders. The stored key held only the two needs and
the tiles they name, and an order depends on where things are: "clear
the ball before fetching the key" holds when the ball stands in front of
the door, not when it stands in a corner. With where things are in the
stored weighing, does recall predict the planner's orders on unseen
layouts well enough that a later card could let it decide them (System
1 for ordering, P21)? P21, P10, P12, P17.

## 2. What changes

One component: what a stored weighing records, and so what recall can
compare. The planner still decides every order (card 074.1's rule);
recall is scored beside it, as in card 074.

| | Card 074 | This card |
|---|---|---|
| A stored weighing | the two needs' kinds and actions; the vectors of the tiles they name and of the held tile; the outcome | the same, plus **where**: each named tile's offset from the other, and from the agent, in the frame of A's target (the tile A's achiever acts on); and the tokens next to each named tile, as code tuples at their offsets |
| What recall compares | own tries (equal codes), else one fitted width over the three vectors | cards 049–050's recall: candidate conditions admitted by their leave-one-out gain (cost log of the number of candidates), own tries first; the new candidates are "the offset between the named tiles is d", "the agent is within d of A's target", "a token with code tuple t lies at offset o from a named tile" |
| The scorer | walking needs grouped by their tile, not their action (a defect) | grouped by kind and action, as every other need |

Offsets are taken relative to A's target, as Pasula, Zettlemoyer and
Kaelbling (2007, `1110_2211`) name a rule's objects by their relation
to the action's target, so that a stored weighing carries to a new
layout where the same arrangement stands elsewhere. Nothing names a
ball, a door or a corridor: which offsets and neighbours matter is
admitted from the stored outcomes, as recall's other conditions are
(card 049). Nothing is kept per pair of placements (the user,
2026-10-05): only offsets between the tokens the two needs name.

## 3. Dependencies

Card 074.1's order rule (its derivation is the truth recall is scored
against); card 074's stored weighings and scorer; cards 049–050's
admission and own tries first; card 044's tokens (a what and a where per
tile). Pasula et al. 2007 (`1110_2211`); Koehler and Hoffmann 2000
(`1106_0243`).

## 4. Data check

On card 074.1's training layouts (tiers 1 and 2 and the decoy world,
seeds from 500,000, as card 074): how many weighings, how many real
orders, and how many distinct (needs, offset) arrangements the orders
come in. If the real orders number under 100, more training layouts are
run before the main scoring.

## 5. Feasibility gate

- **Upper bound:** recall with the evaluator's ground truth of the
  arrangement (the simulator's positions of the two named objects
  relative to A's target, and the objects next to them) in place of the
  agent's tokens. If this cannot beat "no order", where is not what
  orders depend on, and the card stops.
- **Trivial baselines:** always "no order"; card 074's recall (no where).
- **Cost check:** recall per weighing against planning it (card 074:
  the plan tests one weighing needs).

## 6. Success criteria and prediction

On the test layouts (the seeds of cards 073–074.1), report only:

1. **Better than no order:** agreement with the derivation above always
   answering "no order", overall and in each source (tier 1, tier 2,
   decoy with the key known, decoy with trying); McNemar, p < 0.05.
2. **Real orders caught:** at least 80% of the derivation's orders
   predicted, at least 80% of predicted orders real.
3. **Unseen arrangements:** criterion 2 holds separately on weighings
   whose tiles' colours memory never saw in that arrangement.

A later card lets recall decide only where it agrees on every case
(CHARTER: weights and recall take over only so), so 1–3 open that card;
they do not change the agent.

**Prediction.** The offset between the named tiles and the tokens next
to them admit first; most tier 2 orders ("clear what stands in front of
the door before fetching its key") are then caught. Orders that turn on
a route around a blocker several tiles away are missed, since no
offset between the two named tiles shows them.

**Decision rules.** Keep (the stored weighing with where, and the card
that lets recall decide is drafted) if 1–3 hold. Revise if only 3 fails.
Stop if the upper bound fails.

**Budget.** Training layouts about 10 minutes (as card 074); scoring a
few minutes; no tier runs, since the agent does not change.
