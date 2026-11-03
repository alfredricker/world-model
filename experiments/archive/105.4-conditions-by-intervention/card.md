---
id: "105.4"
title: conditions by intervention — what a predicted change depends on, read from the transition model
rung: 1
serves: [P12, P3, P4]
status: done
verdict: revise
arch_version: 27
date: 2026-10-13
---

# 105.4: conditions by intervention

Series [105](../105-one-transition-model/card.md), approved for the overnight
session of 2026-10-13. Offline; no agent.

## 1. Question

A condition is a fact about tokens that an effect depends on (CHARTER, GOAL
P12, 2026-10-13). Suppose a change is predicted, such as a lamp lighting, a
tool appearing in the inventory, or the agent's place advancing. Then remove,
move or replace one input token at a time. Does the set of interventions that
undo the prediction recover the world's real conditions, with no forms
supplied? And are they relations that hold at new positions? Rung 1; P12, P3,
P4.

## 2. What changes

No new model: the best of 105.1–105.3 is read.
- **The interventions,** on a state where the model predicts a change in
  token i:
  - remove token j;
  - move token j by one place;
  - replace token j's vector by another token's vector seen in the same
    state.
- **What counts as a condition of i's change:** the interventions on j that
  remove i's predicted change.
- **The statement of a condition:** j's vector and j's position relative
  to the tokens of the change, and the vectors that keep the change (the
  region).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Removing or moving one token is a meaningful question | yes | yes | yes, in imagination | general |
| Conditions involve few tokens | yes | mostly | mostly | measured, not assumed |

## 3. Dependencies

- 105.1–105.3's models.
- Interventional explanations of a learned model: Pearl 2009; Woodward 2003.
  Both are to be added to LITERATURE with this card.

## 4. Data check

Held-out states with a predicted change, counted per kind of change (by the
evaluator), at least 100 each.

## 5. Feasibility gate

- **Upper bound:** the evaluator's own rules, written out for scoring only:
  - world 1: a box beside the lamp; the cell beyond free for a push;
  - Crafter: a table within reach and the materials for making; something
    in the way for a blocked move.
- **Trivial baseline:** "every token near the change is a condition".

## 6. Success criteria and prediction

1. **The right conditions.** Precision and recall against the evaluator's
   rules, both at least 0.9, for world 1's lamps and pushes, Crafter's
   make, and blocked moves.
2. **Relations, not places.** The same conditions are found at positions
   never seen in training (world 1's test columns; Crafter's table at new
   places).

**Prediction:** world 1 passes. Crafter's make finds the table and the wood,
and misses the iron tools' furnace (too few examples, as in 102.1).

**Decision rules:**
- **Keep** if 1 and 2 hold.
- **Revise** if precision fails because the model reads tokens that do not
  matter.
- **Stop** if recall fails because the model's predictions do not depend
  on the real conditions.

**Budget:** a few thousand forward passes per world; minutes.

## 7. Result

Run 2026-10-13 ([results.json](results.json)). Tools:
`tools/card105.4/world1.py` and `tools/card105.4/crafter.py`.
- **Models:** world 1 used 105.6's three models; Crafter used 105.8's two
  token models.
- **Intervention:** only replacement was run. Every other tile j was
  replaced by each other kind's vector seen in the same state, and
  carried slots by an empty slot's vector. Removal and moving were not
  run: in a grid of tiles a place cannot be empty, and a moved tile would
  overlap another.
- **Truth (scoring only):** world 1's own step function under the same
  replacement, and Crafter's written rules (data.yaml) for making and
  walking.
- **Events:** up to 300 per kind; only those whose unaltered outcome the
  model predicts right are asked. The baseline is "everything within one
  step of the change or the agent" (world 1), or "the 3 × 3 and every
  carried slot" (Crafter).

**World 1** (test worlds, lamps at columns 5–7, never seen in training):

| Event | Precision (seeds 1021, 7, 3) | Recall | Baseline precision / recall |
|---|---|---|---|
| Lamp lights (213) | 1.0, 1.0, 0.99 | 1.0, 1.0, **0.75** | 0.19 / 1.0 |
| Lamp goes out (141) | **0.66, 0.67, 0.67** | **0.40, 0.40, 0.40** | 0.38 / 1.0 |
| Push (300) | 1.0 | 1.0 | 0.14 / 1.0 |
| Move (300) | 1.0 | 1.0 | 0.09 / 1.0 |
| Blocked move (300) | 1.0 | 1.0 | 0.13 / 0.91 |

**Crafter** (held-out worlds):

| Event | Precision (1021, 7) | Recall | Events exactly right | Baseline precision |
|---|---|---|---|---|
| Make, a tool made (232, 228 asked) | 0.69, **0.93** | 0.998, 1.0 | 94%, 96.5% | 0.12 |
| Move, blocked (243, 245 asked of 300) | **0.23, 0.31** | 1.0, 1.0 | 74%, 76% | 0.13 |
| Move, succeeded (300) | 0.97, 0.98 | **0.76, 0.70** | 75%, 71% | 0.12 |

1. **The right conditions: fail** for lamps going out (world 1), and for
   moves and for seed 1021's makes (Crafter). Pass for world 1's pushes,
   moves, blocked moves and lamps lighting (two seeds of three), and for
   seed 7's makes.
2. **Relations, not places: pass where 1 passes.** World 1's events are
   all at lamp columns never seen in training. Crafter's makes find the
   table wherever it stands in the 3 × 3.

**Where it fails, and why:**
- **World 1, lamps going out:** two causes.
  - **A real gap.** A box at the lamp's other side would keep it lit
    (423 such cases), and the model misses every one. It learned "the lamp
    goes out when its box is pushed away", not "lit while any box is
    beside it". Training rarely has two boxes beside a lamp.
  - **Impossible states.** All the extra conditions (493) come from
    replacing the pushed box, which leaves a lit lamp with no box beside
    it, a state the world cannot produce. The model answers "stays lit";
    the rule recomputes "unlit". That is the user's earlier point: check
    conditions only in situations the world's effects can produce.
- **Crafter, moves and makes:** the failures trace back to perception.
  - **Moves that succeed:** the misses are all the grass target. In 71
    events neither a tree nor a cow put there stops the predicted move.
  - **Blocked moves and makes:** the extra conditions are mostly a cow,
    a zombie or a table placed two to four tiles from the agent.
  - **The source:** 103.4's placing is right on only 76% of blocked
    moves, and it errs most when a creature is about (zombies 47%). So the
    model was trained on targets where the agent walked through blockers,
    more often with creatures in view. Its predictions then depend on
    creatures far away, and the conditions read from them do too.

## 8. Decision

**Revise** (2026-10-13). The method works where the model was trained on
true targets. In world 1 the conditions of pushes, moves and blocked moves
are exact at every event, against a baseline precision of 0.09–0.14. In
Crafter the conditions of making are exact in 96.5% of events in one
seed. Where it fails, the failure is in what the model learned, not in
reading conditions from it:
- **Crafter:** perception's misplaced blocked moves teach the model a
  dependence on distant creatures. That is the precision clause of the
  rules (tokens that do not matter).
- **World 1:** the two-box case is too rare in training. The recall
  clause's stated cause, "predictions do not depend on the real
  conditions", holds only for that case.

Next:
1. Fix the placing on blocked moves with the model's own predicted next
   place (105.8's revision). Then rerun this card's Crafter part.
2. Ask conditions only in states the world's effects can produce (the
   user's rule of 2026-10-01). The interventions should be limited to
   replacements the agent has seen happen.
3. Leave the two-box gap to the agent's own experiments. It is a case to
   try, not to engineer.
