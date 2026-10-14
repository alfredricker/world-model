---
id: "071"
title: recall's weights fitted by leaving whole combinations out
rung: 6
serves: [P3, P19, C5, P12]
status: done
verdict: fail
arch_version: 15
date: 2026-10-05
---

# 071: recall's weights fitted by leaving whole combinations out

Card 070's revision (the user, 2026-10-05: "lets proceed").

## 1. Question

Card 070's encoder carries "this key fits this door" (66 of 66; 9 of 9
for unseen hues), and recall admits it as `rel:P`, but recall weighs the
front tile's identity about 40 times the relation, so a key never seen
opening its door is predicted not to (every decoy fold, 7% success).
Recall's weights are fitted by predicting each stored try from all the
others, and every try has twins with the same front and held tiles, so
identity predicts as well as the relation. If instead each try is
predicted only from tries of **other** front and held combinations, does
the fit weigh the relation, so that the agent opens a door with a key it
has never seen open it? P3, P19 (from other combinations' tries), C5
(transfer), P12 (the door becomes a condition again).

## 2. What changes

One component, the relation as recall uses it (card 070's revision): the
objective that fits recall's weights.

| | Card 070 | This card |
|---|---|---|
| Admission's likelihood (card 049, over card 051's groups) | each stored try predicted from every other try (its own counts removed) | each stored try predicted from tries whose (front, held) combination differs from its own (by the tiles' codes); a combination's own tries are left out together |
| The joint refit of the admitted weights, and the online refit of P | the same likelihood | the same new likelihood |
| Prediction | card 050: a situation's own tries first, then the neighbours weighted by these weights | unchanged |

Why this objective: recall's neighbours matter exactly when a situation
has no tries of its own (card 050's own tries first), so the weights should
be the ones that best predict a combination from other combinations. A
fit on single tries rewards identity, which says nothing about a
combination not yet tried. Leaving groups out, not single points, is the
standard cross-validation for held-out combinations (LESSONS: keep new
combinations apart from new members).

Everything else is card 070's: its encoder (`runs/070/encoder.pt`),
`rel:P` in place of `rel:p`, version 15's planner and walking. Against
version 15 this changes the encoder, the relation candidate and this
objective, all parts of the one relation; it is adopted only as a whole.

## 3. Dependencies

Card 070's encoder, code and decoy folds; card 049's admission, card
050's own tries first, card 051's groups (`tools/card051/index.py`).

## 4. Data check

Per fold, before the run: the number of (front, held) combinations among
toggle's stored tries, and that the fold hue's opening combination has
none (card 070: 55–88 tries removed per fold).

## 5. Feasibility gate

- **Upper bound:** card 070's encoder separates every same-colour pair
  from every other (66 of 66), so a recall that weighed only `rel:P` and
  the front tile's kind would be right on all 36 pairs.
- **Trivial baseline:** card 070, the same folds: 35 of 36, the fold's
  own pair wrong; 7% success.
- **Smoke test:** fold red, 4 episodes, before the main run.

## 6. Success criteria and prediction

Card 070's six decoy folds (every hue in memory, less the fold hue's
opening tries), 100 episodes each, P refitted online (one arm: card 070's
two arms did not differ).

1. **Recall:** in every fold, all 36 door and key hue pairs right
   (card 070's "35 of 36" admitted exactly the failure that mattered).
2. **Judged by tries:** success ≥ 99% in every fold, the decoy tried at
   most once per episode on average.
3. **CHARTER's tiers**, run only if 1 and 2 hold: tier 1 ≥ 99%; tiers 2
   and 3 not worse than the best version.

Reported: the admitted conditions and their weights per kind (front
parts against `rel:P`), and pick up's and drop's admitted conditions,
which the new objective also changes.

**Prediction.** Toggle's weight on the front tile's colour falls and the
relation decides across doors: criterion 1 holds. Criterion 2 is less
sure: the agent may also need to know that the open door of a hue it
never opened is walkable, which memory has (walking through stays in the
folds).

**Decision rules.** Keep if 1–3 hold (version 16: card 070's encoder,
`rel:P`, this objective). Revise once if 1 holds and 2 fails. Stop if 1
fails.

**Budget.** About 2 minutes per fold (12 minutes); tier 1 a minute; tier
2 about 15 minutes.

## 7. Result

`runs/071/decoy_*.json` (card 070's encoder; one arm, P refitted online).
Data check: toggle's memory holds 129 (front, held) combinations; the
fold's opening pair has 0 stored tries in every fold.

| Fold | Recall right of 36 | Wrong pair | Success | Toggle's weights: front parts 0–3; `rel:P` | View conditions admitted |
|---|---|---|---|---|---|
| red | 35 | red on red: "no" | 7% | 1.4, 3.3, 11.3, 0.01; 0.75 | 6 |
| green | 35 | green on green | 7% | 0.1, 1.3, 10.0, 0.02; not admitted | 8 |
| blue | 35 | blue on blue | 7% | 0.1, 0.9, 11.1, 0.02; not admitted | 8 |
| purple | 35 | purple on purple | 7% | 5.1, 3.4, 5.9, 0.02; 0.84 | 6 |
| yellow | 35 | yellow on yellow | 7% | 2.5, 4.0, 8.5, 0.01; 0.74 | 6 |
| grey | 35 | grey on grey | 7% | 2.2, 4.5, 7.5, 0.01; 0.74 | 6 |

Card 070's weights were 38–124 on the front parts and 3 on `rel:P`.

- **Criterion 1: not met** in any fold; 2 and 3 were not reached (tiers
  not run).
- The new likelihood moved the weights as intended in size: other
  colours' openings now weigh up to 0.01 for "red key on red door"
  (card 070: 10⁻⁵⁶), but the nearest stored try is still the red door
  with the blue key (0.25), a failure (`tools/card070/why_fold.py` with
  `WM_HOLD_BY=combination`). It also admitted 6–8 conditions on what is in
  view (the goal in view at weight 46), and in two folds `rel:P` was not
  admitted at all.
- **Why the front tile keeps its colour** (card 070's encoder, MiniGrid's
  tiles, report only). Every part carries both the tile's kind and its
  colour at similar sizes: in part 2, two colours of locked door differ by
  0.75, a locked door and a key, box or ball of its colour by 0.88–0.94
  (parts 0–1 alike; part 3 also carries open, closed and locked). Toggling
  needs the kind (a door, not a wall or a box), and one weight per part
  cannot keep the kind and drop the colour.

## 8. Decision

**Stop**, as declared when criterion 1 fails; version 15 unchanged. The
fit by combinations is the right question for recall's neighbours, but
recall reads the front and held tiles through fixed quarters of the
vector, which mix kind and colour. What it points to: the tiles' own
conditions read through learned projections of the whole vector, as the
relation already is (card 070), so that "a locked door" can be a region
that ignores colour; the user's call.
