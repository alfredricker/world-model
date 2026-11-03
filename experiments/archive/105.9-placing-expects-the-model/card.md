---
id: "105.9"
title: perception expects what the model predicts — the view's motion from the transition model, not the action's habit
rung: 1
serves: [P6, P3, P7]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 105.9: perception expects what the model predicts

Series [105](../105-one-transition-model/card.md), written in the overnight
session of 2026-10-13 under the user's approval of the series; the tenth
and last new card of the night. Offline; no agent. It follows
[105.8](../105.8-likelihood-of-the-next-code/card.md) and
[105.4](../105.4-conditions-by-intervention/card.md) (both revise).

## 1. Question

Card 103.4's placing finds the view's motion by scoring each candidate
shift on the frame's evidence plus a prior p(shift | action). The prior is
the agent's own past placements. A move action moves the view about 90% of
the time, so a move into a tree or a zombie is expected to move it too.
That placing is right on 99.6% of moves that succeed but only 76% of
blocked ones (zombies 47%). Its errors taught 105.8's model that the agent
sometimes walks through blockers, and 105.4 read that back as conditions on
distant creatures.

105.8's model predicts the agent's next place. It reads positions only as
differences, so its prediction does not depend on the view's origin. With
correct correspondence it gets 100% of blocked moves. Suppose the placing's
prior is the model's predicted next place instead of the action's habit.
Is the placing right on blocked moves, and do 105.8's moves and 105.4's
Crafter conditions then pass? Rung 1; P6, P3, P7.

## 2. What changes

One component, the placing's prior:
- **Before:** p(shift | action), counted on training play.
- **Now:** p(shift | state, action) from the transition model. For each
  view tile, the probability that the agent is there next (105.8's
  read-out) is put on the shift that would centre the view on it. The
  result is mixed with 1% of the old prior, so that no shift is
  impossible.

The placing's evidence, its tempering (τ = 300) and its candidates are
103.4's, as card 103.5 fitted them. The model is 105.8's token model with
seed 1021, trained on placements from the old prior. Its prediction on the
training worlds is therefore in-sample: one step of expectation and
maximisation, not a fixed point.

After placing, 105.8's token model is retrained on the new places (seeds
1021 and 7) and scored as in 105.8. 105.4's Crafter part is rerun on it.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The view moves with the agent | yes | sometimes (scrolling games) | yes (a head camera) | learned (103.1) |
| The model's predicted next place can be read as a view shift | yes (the agent stays centred) | yes where the view follows | yes, as predicted ego-motion | declared: the agent's tokens are found by perception (the code at the view's centre) |
| Perception may use prediction's expectation | yes | yes | yes (predictive coding) | general |

## 3. Dependencies

- 103.1's placing (`tools/card103.1/place.py`), with 103.5's fitted
  parameters.
- 105.8's model and runner (`tools/card105.2/run.py`).
- 105.4's Crafter scorer.
- Predictive processing: perception as inference under the model's
  expectations (Rao and Ballard 1999; Friston 2005). To be added to
  LITERATURE.

## 4. Data check

All fourteen sources are re-placed: the curriculum's worlds 0–9 and random
play's workers 0, 1, 8 and 9. Blocked moves: 4,373 in training, 1,896 in
the test.

## 5. Feasibility gate

- **Upper bound:** 105.8's model with the evaluator's origins. It gets
  99.99% of moves and 100% of blocked moves.
- **Trivial baseline:** the old prior, at 76% on blocked moves in the test.

## 6. Success criteria and prediction

1. **The placing.** Right on at least 95% of blocked moves in the test,
   and at least 99.5% of moves that succeed (the old prior's level).
2. **The model's moves.** The agent's next place is right on at least 98%
   of move steps, blocked included, in both seeds (105.8's criterion 2).
3. **Nothing else lost.** 105.8's makes, place and role test stay within
   2 points.
4. **The conditions.** In 105.4's Crafter part, precision and recall of at
   least 0.9 for blocked and successful moves.

**Prediction:**
- Criteria 1 and 2 pass: with the model's prior, a move into a tree
  expects no motion, and the frame confirms it.
- Criterion 4 improves but stays short on blocked moves near creatures. A
  creature that moves on the same step makes the evidence ambiguous, and
  the model trained once on the old places still carries some of their
  error.

**Decision rules:**
- **Keep** (the placing takes its prior from the model) if 1–3 pass.
- **Revise** if 1 passes but 2 or 3 does not.
- **Stop** if the placing is no better than 85% on blocked moves.

**Budget:**
- Placing fourteen sources: about 10 minutes, in short runs.
- Retraining: two runs of about 5 minutes.
- Scoring conditions: two runs of a minute.

All capped.

## 7. Result

Run 2026-10-13 ([results.json](results.json)). The placing took 3.5 minutes
for fourteen sources, and each retraining about 5 minutes.

| Measure | Old prior (105.8) | Model's prior | Bar |
|---|---|---|---|
| 1. Placing: blocked moves right, held out (1,896) | 76% | **85.0%** | ≥ 95% |
| — curriculum test worlds 8, 9 | 86.6%, 86.2% | 95.2%, 93.3% | |
| — random play workers 8, 9 | 69.3%, 63.4% | 78.9%, 73.9% | |
| 1. Placing: moves that succeeded, held out | 99.4–99.95% | 99.1–99.95% | ≥ 99.5% |
| 2. Agent's next place, all moves (seeds 1021, 7) | 97.25%, 97.5% | 97.6%, 97.6% | ≥ 98% |
| — blocked / succeeded | 76.9–79.4% / 99.8–99.9% | 83.1% / 99.4% | |
| 3. Makes that made a tool | 97.0%, 95.3% | 96.2%, 96.2% | within 2 points |
| 3. Do's changes (saplings left out) | 82.7%, 86.7% | 84.9%, 91.1% | within 2 points |
| 3. Role test | 94.4%, 96.4% | **91.8%, 89.3%** | within 2 points |
| 4. Conditions, blocked moves: precision | 0.23, 0.31 | 0.46, 0.40 | ≥ 0.9 |
| 4. Conditions, successful moves: recall | 0.76, 0.70 | 0.76, 0.82 | ≥ 0.9 |
| 4. Conditions, makes: precision / recall | 0.69 / 1.0, 0.93 / 1.0 | 0.87 / 1.0, 0.70 / 1.0 | ≥ 0.9 |

- **Criterion 1: fail.** 85.0% against 95%, on the stop line.
  - **Curriculum:** the model's prior fixed most blocked moves (93–95%).
  - **Random play:** it did not (74–79%). Half of random play's blocked
    moves are at night, and zombies still block unseen on 54–57% of
    tries. The model that gave the prior was trained on the old
    placements, so it had learned zombie blocks badly: one step of
    expectation and maximisation carried the old error forward.
- **Criterion 2: fail** (97.6%). Blocked moves gained 4–6 points;
  successful moves lost 0.4, where random play's new placements broke a
  few of them.
- **Criterion 3: fail** on the role test (−2.6 and −7.1 points). The same
  test has varied from 87.8% to 96.4% between runs of one setup (105.8),
  so this is weak evidence of harm.
- **Criterion 4: fail,** though every move measure improved. Blocked-move
  precision roughly doubled, and the share of events exactly right rose
  from 74–76% to 82–83%.

## 8. Decision

**Stop** (2026-10-13), this form: one step of the model's prior. The rule
says stop if the placing is no better than 85% on blocked moves; it
reached 85.0%. Criteria 1, 2 and 4 fail and 3 regresses. The direction
held: every measure of moves and conditions improved when perception
expected what the model predicts, and the curriculum's worlds reached the
bar. What fell short is a single step from a model trained on the old
errors, on random play's night and creatures.

Two ways forward for the user:
- **Iterate:** place, retrain and place again, until the placements stop
  changing.
- **Give the model the same evidence as the placing:** train the model's
  prediction of the agent's next place on both frames, not only the
  first, so that it does not inherit the placing's errors.

Either is a new card. Tonight's ten are used.
