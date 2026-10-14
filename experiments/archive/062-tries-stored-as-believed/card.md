---
id: "062"
title: tries stored with the believed view
rung: 1
serves: [C2, C1, P2, P10, P12]
status: done
verdict: pass
arch_version: 13
date: 2026-10-05
---

# 062: tries stored with the believed view

Drafted and started overnight on 2026-10-05, after card 061 (AGENTS.md,
"Overnight sessions"; the user's objective: a view smaller than the
map). The numbers below were fixed before the main run.

## 1. Question

Version 13 still stores each pick up, toggle and drop of its starting
memory with what was in view in the 13 × 13 window that holds the whole
map, the last part of card 057's declared exception. If each stored try
instead carries what the agent believed was around it at that step, seen
through the 7 × 7 occluded view, does the agent still discover its
conditions (the user's question for the encoder, card 054) and act as
well? C2 (one representation for learning and acting), C1, P2 (belief),
P10 (memory of particular tries), P12 (conditions).

## 2. What changes

One component: what a stored try records as in view.

| | Version 13 | This card |
|---|---|---|
| A stored pick up, toggle or drop: what was in view | the set of appearances in the 13 × 13 window | the set of appearances in the agent's believed 13 × 13 view at that step: the tiles it has seen in this episode through the 7 × 7 occluded view, placed by its own motion (card 061's transformations; a forward step happened when the view changed), a place never seen showing the unseen appearance, as in acting |
| The place ahead and the hand | from the view | unchanged (always in view) |

The stored play is replayed episode by episode from card 033's
collection (the same seeds, chunks and sampling); every stored row's
action and every stored try's view were checked against the stored play
before use (`runs/062/believed.json`: all equal in all four worlds).

## 3. Dependencies

Version 13 (card 061); card 049's admitted conditions; card 050's own
tries first; card 054's encoders.

## 4. Data check

About 300,000 stored tries per familiar world (306,096 in the key
world). In the replay the believed view holds 6.2 distinct appearances
on average; no try was made with a believed tile that disagreed with
the simulator (beliefs never went stale, since only the agent changes
these worlds and it sees what it changes).

## 5. Feasibility gate

- **Upper bound:** version 13 (the full window's "in view").
- **Trivial baseline:** none needed beyond version 13; the question is
  whether anything is lost.

## 6. Success criteria and prediction

On the encoder of seed 399, then 400–402 for criterion 1:
1. **Conditions discovered** as in card 054's criterion 3: for toggle,
   the key world admits the hand or the hand–front relation, and the
   switch world admits a condition on what is in view (the switch).
2. **Acting:** familiar worlds ≥ 99% with steps within 5% of card
   061's; chained rooms with one door ≥ 95%; no wrong remembered tile.

Reported: the share of stored tries whose "in view" set changed.

**Prediction.** Both hold. Most tries come late in a 640-step episode,
when the room has long been seen, so the believed view mostly equals the
full one; where it does not (early tries), what is missing is what the
agent had not seen, which should not have been a condition anyway.

**Decision rules.** Keep (version 14: the starting memory learned
entirely under the 7 × 7 view) if both hold; one declared revision if
one fails; stop if the familiar worlds fall below 90%.

**Budget.** About 10 minutes.

## 7. Result

`runs/062/fam_{399..402}.json`, `chain_399.json` (seed 402 reran alone
after running out of GPU memory beside four other runs).

**What changed in memory.** 92–96% of the stored tries per world now
record a different "in view" set: 6.2 distinct appearances on average
against 7.2 in the full window (the key world: 295,167 of 306,096).

**1. Conditions discovered** (toggle; pick up and drop admit the
hand–front relation as before):

| World | Full window (card 061) | Believed view |
|---|---|---|
| Key | relation or hand (all seeds) | the same on every seed |
| Switch | a grey switch in view | a yellow switch in view (every seed) |
| Either | grey switch + relation or hand | yellow switch + relation or hand |
| Both | hand + switch + relation | hand + switch (grey or yellow) + relation or hand |

The switch condition turned from its negative form to its positive one.
In the full window the switch was always in view, so "a grey switch in
view" carried the same information as "the switch is on"; in belief the
switch may not have been seen yet, and only "a yellow (on) switch in
view" predicts the door opening. Criterion 1 holds on all four seeds.

**2. Acting** (seed 399; seeds 400–402 the same): familiar worlds 100% at
18.13, 19.13, 17.13 and 23.47 steps, card 061's to the hundredth;
chained rooms with one door 100% at 1.61 × the shortest route; no wrong
remembered tile. Criterion 2 holds.

## 8. Decision

**Keep** (version 14: the starting memory learned under the 7 × 7 view).
With cards 061 and 062, nothing the planner remembers comes from a view
the agent does not have: how moves change where things are, how a tile
looks once left, and what each stored try had around it are all learned
from MiniGrid's 7 × 7 occluded view and the agent's belief, and the
agent discovers the same conditions (the switch now in its positive
form) and acts exactly as before. Card 057's declared exception is
closed for the planner's memory. Not re-checked tonight: the encoder's
transition term (card 054), which trains on the tile ahead and the hand
(both always in the 7 × 7 view) but was not audited for anything else.
The tile under the agent in the replay is the tile with the agent
removed, which equals card 061's learned undraw on every tile the agent
steps off.
