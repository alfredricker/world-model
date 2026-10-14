---
id: "084"
title: conditions over roles and relations, a level of recall between own tries and neighbours, admitted by evidence
rung: 6
serves: [P3, P12, P8, P19]
status: done
verdict: fail
arch_version: 18
date: 2026-10-07
---

# 084: conditions over roles and relations

Drafted at the user's request (2026-10-07), after reading cards 076–083
with the user. That series sought general structure: orders, "pick it up
and move it", "a key opens the door of its colour". Structure carried
only when a relation between roles was an input: "on the way" (card
078) and a readout linear in card 070's relation (card 080). A learned
similarity over whole situations interpolated (076, 077, 079, 082–083).

## 1. Question

Version 18's conditions name particular tiles. Its own-situation key is
the front and held tiles' code tuples, and its neighbours weigh the
door's identity about 40 times the relation (cards 070–071). So "a key
opens the door of its colour" is stored as one fact per colour, and in
card 070's colour folds the left-out pair is predicted not to open (0 of
6). Suppose recall can also state a condition over **roles and
relations** (what a tile does under the other actions; which side of a
cut card 070's relation falls on), admitted by how well it compresses
the stored outcomes. Does it then carry the rule to the left-out pair,
keep every other cell right, and fix tier 2's errors about the hand?
P3, P12 (conditions in general terms), P8, P19.

## 2. What changes

One component: recall for pick up, toggle and drop. A level of cells
is added between a query's own tries and its neighbours.

```
version 18:  P = (N_own + β P_nb) / (|N_own| + β)
this card:   P = (N_own + β P_1) / (|N_own| + β)
             P_ℓ = (N_ℓ + γ P_ℓ+1) / (|N_ℓ| + γ),  ℓ = 1 … k,  P_k+1 = P_nb
             N_ℓ: stored tries equal to the query on the first k − ℓ + 1 admitted conditions
```

- **Own tries, unchanged** (card 050). One failed try still overrules
  everything below, so card 072's trying still works.
- **Cells.** Level 1 uses every admitted condition. Each coarser level
  drops the condition admitted last, so the most general one goes
  last. This is MacKay and Peto's back-off again. γ is fitted by
  leaving one stored key out, as β is.
- **Neighbours, unchanged:** version 18's kernel, reached only when
  every cell is empty.

**Candidate conditions**, none of which reads a tile's vector:
- **Roles:** what version 18's recall predicts each tile does,
  empty-handed, under the other actions (card 077's bits: forward moves
  onto it, forward onto it ends the episode, a pick up changes it, a
  toggle changes it). These are given for the front tile and the held
  tile, without the action's own bit.
- **Relation side:** ‖P(z_front − z_held)‖₁ (card 070's frozen P) below
  or above a cut. The cuts are midpoints between consecutive stored
  values.
- **Identity:** card 049's candidates (the front, held and view code
  tuples).

**Admission by evidence.** Conditions are added greedily while a
condition raises the evidence by more than log(number of candidates),
which is card 049's cost. The evidence is the Dirichlet-multinomial
marginal likelihood of the stored outcome counts, summed over the cells,
with one pseudo-count per class (card 010). Unlike leave-one-out, it
prefers one cell holding every colour to six pure cells, one per colour
(appendix).

**Unchanged:** the planner, walking, the encoder, P (no online refit)
and moves. For a locked door, the planner asks which hand recall
predicts makes the toggle work (card 043), so a new prediction changes
its choice of key with no change to the planner. Card 074.1's split
need "hold h" still names a stored tile; lifting it is a later card.

## 3. Dependencies

- Card 079's colour folds and 56-cell toggle table
  (`tools/card079/effects.py`).
- Card 082's pick-up (28) and drop (42) tables.
- Card 077's roles.
- Version 18's recall and groups (`tools/card051/index.py`,
  `tools/card050/own.py`).
- Card 070's encoder and P.
- Evidence and back-off: card 010; MacKay and Peto (LITERATURE.md).

## 4. Data check

Per fold and action:
- stored keys and outcome counts;
- admitted conditions in order, with their gains in bits;
- the cells the left-out pair falls in, and what they hold;
- toggles on closed, unlocked doors, which share a locked door's roles.

In tier 2: pick ups with the hand full, and toggles at the locked door
while holding the ball.

## 5. Feasibility gate

- **Upper bound:** with nothing removed from memory, every cell of the
  toggle (56), pick-up (28) and drop (42) tables is right. Version 18
  gets them all.
- **Trivial baseline:** version 18, which opens the left-out pair in 0
  of 6 folds.
- Admission is deterministic: no seeds and no gradient fits, so cards
  079–083's spread between seeds does not arise.

## 6. Success criteria and prediction

1. **Transfer:** the left-out pair is predicted to open in 6 of 6
   folds, and every other cell of the three tables is right in every
   fold.
2. **Tier 2's hand:** from tier 2's start view, with the evaluator's
   truth, every cell is right of pick up {box, ball, key} × held
   {nothing, ball, key} and of toggle the locked door × held {nothing,
   ball, key}. ARCHITECTURE.md reports that version 18 predicts the
   door opens while the ball is held.
3. **Own tries still overrule:** in each fold, after one failed try of
   the left-out pair is added, that pair is predicted to fail (card
   050; P11).

Also reported:
- leave-one-key-out likelihood per action, against version 18;
- time to admit and time per query;
- P19's curve: memory with 1, 2, 3 and 5 colours, and where the
  relation cell is first admitted ahead of identity.

**Prediction.**
- Toggle admits the front tile's roles, then the relation's side. The
  cut falls between the largest stored matching relation and the
  smallest mismatch (about 1.4–1.8 against 2.94), so red's pair (1.80)
  opens, where card 079's vote failed.
- Identity comes later, for closed doors.
- Pick up admits the held tile's "a pick up changes it" (the hand is
  full).
- Drop admits the front tile's "forward moves onto it".

**Risks.**
- Noisy renders may push a mismatch across the cut.
- Raw stored counts may admit identity for rare exceptions. Counts per
  key are reported.

**Decision rules.**
- **Keep** if 1–3 hold. The next card puts the level into the agent:
  the decoy folds with trying (version 16: 0.06–0.09 wrong-key tries
  per episode; version 18 no different) and CHARTER's three tiers.
- **Revise** if 1 holds and 2 or 3 fails.
- **Stop** if the gate or criterion 1 fails.

**Budget.** Seven fold setups at about 70 s each, and admission takes
seconds. Tier 2's memory setup is timed at the gate. If it would take
the card past 10 minutes, it goes to the user as a command.

## 7. Result

`runs/084/` (`tools/card084/lifted.py`; `none`, the six folds, `tier2`;
about 2 minutes per fold setup, 3 for tier 2, three folds at a time).
γ was fitted by leaving one try out, as β is (section 2 said "one stored
key"). A diagnostic arm, run beside the declared one after the gate
showed the order of admission, scores the evidence on whether the action
changed anything (the category) rather than on the outcome class, which
includes the tile a place became.

**Gate: passed.** With nothing removed, every cell right: toggle 56,
pick up 28, drop 42 (version 18 the same). With no cells, the formula
reproduces version 18 on every cell of every table.

**What was admitted** (gate; the same in every fold):

| Action | Declared arm (outcome class) | Diagnosis (category) |
|---|---|---|
| Toggle | front identity (11,920 bits), then rel:P > 1.96 (2,084) | the same (9,446; 2,158) |
| Pick up | front identity, then the held tile's "forward moves onto it" | the same |
| Drop | held identity, front "forward moves onto it", front "toggle changes it" | held "forward moves onto it", front identity |

The cost of a condition is about 7 bits. Stored tries number about
528,000 per action.

| Fold | Left-out pair opens (both arms) | Other toggle errors | Pick up / drop | Criterion 3 |
|---|---|---|---|---|
| red (1.80) | no (cut 1.58: the pair falls above it) | 0 | 28 / 42 | already "no" |
| green, blue, purple, yellow, grey | no | 0 | 28 / 42 | already "no" |

**Tier 2** (declared arm): toggle 3 of 3 (version 18: 2 of 3, the ball
"opened" the door); pick up 7 of 9 (version 18: 6 of 9). Still wrong:
picking up the box while holding the ball or the key. The pick-up cut
(rel:P > 7.52) stands in for "the hand is empty", since the empty hand's
relation to any tile is large; memory holds no try at the box with
something held.

- **Criterion 1: not met** (0 of 6 in both arms; version 18 0 of 6).
- **Criterion 2: not met** (pick up 7 of 9).
- **Criterion 3:** met only trivially: the pair was already predicted
  to fail.
- **Why.** The front tile's identity is admitted first for every
  action, in both arms, and with 1, 2, 3 or 5 colours in memory alike.
  Toggle's roles under the other actions cannot tell a locked door from
  a closed one or a wall (none can be walked onto or picked up), and
  closed doors are common (308 stored toggles). Identity splits all of
  them in one condition, for one cost, and on about 528,000 stored tries
  its likelihood gain dwarfs the Occam cost of its extra cells. The
  relation is then admitted only inside each door's cell. Since the
  back-off drops the condition admitted last first, the cells a query
  falls back to are "this door" and then "this door with a small
  relation": the left-out door's cell holds only its failures (10–14
  stored keys), and no cell holds other colours' openings without
  identity. The rule is in memory (in the gate, every door with a small
  relation opens, and the cut lies in the gap, 1.96 between 1.80 and
  2.94), but only as one rule per door.
- Cost: admission 0.2–0.3 s per action, 0.2 ms per query.

## 8. Decision

**Stop** (criterion 1). Evidence over the stored tries found the
relation (its cut sits in the gap between matching pairs and
mismatches), but as a refinement of the door's identity, because
identity is the one condition that also tells locked doors, closed doors
and walls apart, and raw try counts make its many cells cheap. A chain
of back-off in admission order then keeps the rule per door. A revision
would need both of two things the result points to: cells that drop
identity before the relation (a back-off over sets of conditions, most
specific first, not over the order of admission), and either evidence
per distinct stored situation rather than per try, or a role that tells
a locked door from a closed one without its colour. Neither is drafted;
the choice is the user's.

## Appendix: why evidence, CHARTER, priors, literature

**Why evidence.** Leave-one-out cannot prefer the rule. Every try has
twins at its own door, so one cell per colour predicts as well as one
cell for "a door and a key whose relation is small" (LESSONS,
Generalisation). The evidence can prefer it: splitting a cell whose
outcomes are already pure costs about log n per new cell. This is the
size preference of Popper and the Apperception Engine without their ban
on constants, since identity is still admitted where the data need it.

**CHARTER.** A condition is "a region of [the encoder's vectors] … what
recall treats as alike for an action", and a relation is "a constraint
between two things' vectors in one part". Roles come from what tiles
do. The cells are recall's own groups (version 10), re-read whenever
admission is refitted. Stored tries are never merged, and no network is
added.

**Declared priors (C3):**
- the candidate family (four role bits, a cut on one relation,
  identity, view tuples);
- the Dirichlet pseudo-count;
- the cost per condition;
- the back-off order.

**Literature:**
- Pasula, Zettlemoyer and Kaelbling 2007 (`1110_2211`): rules name
  objects by their relation to the action's target.
- Letham et al. 2015 and card 010: rule evidence with a cost per
  condition.
- Plotkin 1970: the same difference becomes one variable.
- `2005_02259` (Popper) and `1910_02227` (Apperception Engine): size
  favours rules without constants.
- `tenenbaum-griffiths-generalization`: under the agent's own play,
  only a prior separates the rule from per-colour facts.
- MacKay and Peto 1995: the back-off.
