---
id: "083"
title: reasoning over recalled tries, comparing tiles through card 070's learned relation
rung: 6
serves: [P3, P10, P21, P6]
status: draft
verdict:
arch_version: 18
date: 2026-10-06
---

# 083: reasoning with a learned relation

Drafted at the user's request (2026-10-06), after card 082.1.

## 1. Question

Cards 082 and 082.1 failed their gates. Their network learned its own
comparison between tiles and fitted pick up and drop, but never learned
from about 36 door and key combinations that a key opens the door it
matches in colour; on toggle it did no better with memory than without.
Where the relation came already learned (card 070's projection P,
trained with the encoder), the colour folds mostly passed: card 079's
vote 4 of 6, card 080's readout 6 of 6 at two of three seeds. If the
reasoner compares tiles through card 070's relation, and only has to
read rules from the tries it recalls, does it pass the gate, carry the
relation to the left-out colour, and use its memory to do so? P3, P10,
P21, P6.

## 2. What changes

One component against card 082.1: how two tiles are compared.

| | Card 082.1 | This card |
|---|---|---|
| Comparing tiles a and b | eight learned similarities ⟨Φ_h z_a, Φ_h z_b⟩, Φ trained with the reasoner | card 070's relation \|P(z_a − z_b)\| (8 values), P frozen as trained with the encoder in card 070 |
| View sets | mean of the view's projections | the same, with P |

Everything else is card 082.1's:
- the same rows, router and reasoner;
- pick up, drop and toggle in one network, with no action label;
- training with both kinds of query, combinations weighted equally;
- 3,000 steps, seeds 79, 80 and 81.

Only relations pass forward: card 070's relation is a difference
between two tiles, never one tile's vector.

**Known limit.** P was trained in card 070 for toggle's relation.
Whether it helps pick up and drop is part of criterion 2. If this card
works, the next question is whether the encoder can learn such relations
for every action without a dedicated objective for each.

## 3. Dependencies

Card 082's script (`tools/card082/reason.py`), card 070's encoder and P
(`runs/070/encoder.pt`).

## 4. Data check

As card 082.

## 5. Feasibility gate

As card 082: nothing removed, seeds 79, 80 and 81, every cell of the
toggle (56), pick-up (28) and drop (42) tables right. Also at seed 79:
no router (every row in context), and no memory (the query alone).

## 6. Success criteria and prediction

Card 082's three criteria, each holding at each of seeds 79, 80 and 81:

1. **Transfer:** the left-out pair opens in 6 of 6 folds, with no other
   toggle error.
2. **One mechanism for every action:** the pick-up and drop tables are
   right in every fold.
3. **It reasons over memory:** with the retrieved outcomes shuffled at
   test time, the left-out pair opens in at most 2 of 6 folds.

**Prediction.**
- **The gate passes.** Card 080's readout fitted all 56 toggle cells
  with this relation.
- **Criterion 1 holds.**
- **Criterion 3 is the open question.** With the relation given, the
  network could learn "a small relation means it opens" in its weights,
  like card 080's readout, and ignore its memory. The no-memory arm in
  the gate shows how much it can do from its weights alone.

**Decision rules.**
- **Keep** if 1–3 hold. The next card arbitrates between this network
  and version 18's recall by held-out reliability (CHARTER), inside the
  agent, on the decoy folds and the three tiers.
- **Revise** if 1 and 2 hold but 3 fails: a System 1 readout, not
  reasoning over memory.
- **Revise** if only 2 fails.
- **Stop** if the gate or criterion 1 fails.

**Budget.** Gate about 5 minutes; folds 6 × about 3.5 minutes.
