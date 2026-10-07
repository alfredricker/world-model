---
id: "082"
title: a network that reasons over recalled tries, with a learned router
rung: 6
serves: [P3, P10, P21, P6]
status: draft
verdict:
arch_version: 18
date: 2026-10-06
---

# 082: reasoning over recalled tries

Drafted at the user's request (2026-10-06), after the literature review
in [LITERATURE.md](../../LITERATURE.md) (current focus) and CHARTER's
"recall and networks, each where it predicts better". Report only;
putting it into the agent is a later card.

## 1. Question

Recall compares the query with each stored try separately, then votes.
It never compares stored tries with each other, so it cannot induce
from memory that "in every opening the door and key were alike in
colour, in every failure they were not". In card 070's colour folds, one
colour's door openings are removed from memory. Version 18 then predicts
that colour's key will not open its door, in 6 of 6 folds. Cards 077–080
reached 4–6 of 6 only with inputs built by hand: hand-picked roles, card
070's fixed projection, a readout grouped by roles.

Can one network do what was hand-built there, by reading a set of
recalled tries, comparing them with each other and with the query, and
predicting the query's outcome? It is trained only by hiding the agent's
own stored tries and predicting them from the rest. It must use the same
weights for pick up, drop and toggle, see no action label and get no
hand-built feature. The test is whether it opens the left-out pair in
every fold, without new errors, and actually reads its memory to do so.

- **P3:** a known relation applied to new participants.
- **P10:** recall used to infer, consolidating repeated patterns.
- **P21:** reasoning over memory as System 2.

## 2. What changes

One component: recall's prediction for a query with no identical stored
try. Version 18 votes over its neighbours; this card retrieves rows and
reasons over them.

**Rows.** Each stored try of an action is a row: the tile in front, the
tile held, the tiles in view, and its outcome counts. The query is a row
with its outcome hidden.

**Relations only** (the Abstractor and ESBN; CoRelNet's finding that
raw features added beside relations break transfer):
- Tile vectors come from the frozen encoder. They are normalised over
  the tiles of the episode, as ESBN normalises over its context.
- Eight learned projections Φ_h each give a similarity ⟨Φ_h z_a, Φ_h
  z_b⟩ between two tiles. No tile vector passes forward, so a door's or
  colour's identity cannot be memorised.
- **Within a row:** the relations between front and held, and between
  each view tile and the front and the held, pooled over the view.
- **Between two rows:** the relations between their fronts, between
  their helds, and between their view sets.
- Nothing chooses which pairs are compared: every slot is compared with
  every slot.

**The router** (stage 1; Goyal et al.):
- A score for each stored row, from the between-row relations to the
  query.
- The top 64 rows are retrieved.
- The log of each retrieved row's weight is added to the reasoner's
  attention on that row, so the router trains through the reasoner's
  loss.

**The reasoner** (stage 2; Non-Parametric Transformers, Attentive Neural
Processes):
- Three attention layers over the rows, with between-row relations as
  attention biases.
- Retrieved rows attend to each other; the query attends to them.
- Width 64, four heads.
- Output: the query's outcome distribution.

**Training on the agent's own memory:**
- **One action per step.** Its stored tries form the memory; no label
  says which action.
- **Queries hidden by combination.** 32 (front, held) combinations
  become queries, with all their rows removed from the memory (card 071:
  leave whole combinations out).
- **Loss:** cross-entropy against each query's stored counts. One
  network per fold, trained on that fold's memory.

**Why this pushes the network towards reasoning rather than memorising**
(Chan et al.): the outcome can't be read from the query alone. With no
action label and no tile vectors, which rule applies and which tiles
behave alike show up only in the retrieved rows. No synthetic data and
no colour re-mapping are added; both would be a prior written by hand.

**Rule 8:** inputs are the encoder's vectors and the agent's stored
outcomes, and the same weights serve every action.

**Declared exception:** forward moves are not included yet. Their memory
has another form (card 061's transformations); the card that puts the
reasoner into the agent adds them.

## 3. Dependencies

Card 079's fold setup and toggle table (`tools/card079/effects.py`).
Version 18's memory for pick up, drop and toggle. Card 070's encoder.
Literature: the current focus in LITERATURE.md.

## 4. Data check

Per fold and action:
- stored rows and (front, held) combinations: about 1,300 rows in about
  120 combinations, measured 2026-10-06;
- outcome categories in use;
- how often a toggle query's top 64 rows include a stored opening.

## 5. Feasibility gate

Run before the folds, nothing removed from memory, seeds 79, 80, 81.

- **Upper bound:** all 56 cells of card 079's toggle table right, and
  all cells of the pick-up and drop tables right. Those tables use the
  same fronts with held nothing or one key; the evaluator's truth is
  that a key or ball can be picked up only empty-handed, and a held tile
  can be dropped only onto the floor.
- **Trivial baselines:** version 18 (0 of 6 fold pairs), and always
  "nothing happens".
- **Time:** one fit measured. If the folds would take over 30 minutes,
  they go to the user as commands.

## 6. Success criteria and prediction

Each criterion must hold at **each** of seeds 79, 80 and 81 (cards
079–080: single fits were noise):

1. **Transfer:** the left-out pair is predicted to open in 6 of 6 folds,
   and every other cell of the toggle table is right in every fold.
2. **One mechanism for every action:** the same network gets the pick-up
   and drop tables right in every fold.
3. **It reasons over memory:** with the retrieved rows' outcomes
   shuffled among them at test time, the left-out pair opens in at most
   2 of 6 folds. A network that kept answering "opens" would be
   answering from its weights, not from what memory says (Kossen et
   al.'s test).

**Also reported:**
- leave-combination-out accuracy on each action's stored combinations;
- at seed 79 only: no router (every row retrieved) and no memory (the
  query row alone);
- attention from the left-out pair's query to stored openings of other
  colours.

**Prediction.** Criteria 1 and 2 hold: relations alone carry colour
(card 080's readout, card 070). Criterion 3 is the uncertain one. With
three actions and one world, the data are close to Chan et al.'s
memorising regime, so the network may learn the colour rule in its
weights and ignore its memory.

**Decision rules.**
- **Keep** if 1–3 hold. The next card arbitrates between this network
  and version 18's recall by held-out reliability (CHARTER), inside the
  agent, on the decoy folds and the three tiers.
- **Revise** if 1 and 2 hold but 3 fails: a good System 1 readout, but
  not reasoning over memory. Training needs more variety.
- **Revise** if only 2 fails.
- **Stop** if 1 fails.

**Budget.** The gate, about 10 minutes, measures one fit. The folds are
7 memory setups of about 70 s each, plus 21 fits and the seed-79 arms.
That is an estimate of 20–40 minutes, to be confirmed by the gate.
