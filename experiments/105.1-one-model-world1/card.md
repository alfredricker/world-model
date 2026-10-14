---
id: "105.1"
title: one transition model in breaking world 1 — moves, blocking, pushes and lamps, every token's next vector
rung: 1
serves: [P6, P3, P7]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 105.1: one transition model in breaking world 1

Series [105](../105-one-transition-model/card.md), approved by the user for the
overnight session of 2026-10-13. Offline; no agent.

## 1. Question

In breaking world 1 the agent walks, is blocked by walls, and pushes boxes. A
lamp lights while a box stands beside it, and the lamps stand in columns 1–3
in training and 5–7 in test. Can one transition model, with no part for
moves, predict every token's next vector? Does the lamp's relation carry to
columns never seen? Card 102.1's every-token arm learned it column by column:
2 of 4 runs carried. Rung 1; P6, P3, P7.

## 2. What changes

The series' model (105, section 2) replaces 102.1's arm A.
- **Outputs:** next vectors, not change probabilities, keep, copy, shift or
  set heads.
- **Positions:** read only as differences, through attention, not as raw
  numbers.
- **Tokens:** the 81 tiles of the fixed view (world 1's encoder, card 102.1;
  7 codes, one kind each). Their position is the tile's column and row, and
  the action is a token. The same tile is the same token.
- **Arms:**
  - **R:** relative positions only (the series' form);
  - **A:** raw positions as inputs, as in 102.1's first run.

  Each arm runs two seeds (1021 and 7, as in 102.1).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Tokens are tiles of a fixed view | no (105.2 places them in the world) | yes | no (105.3 uses objects) | declared adapter |
| Positions compared within a frame | yes | yes | yes | declared |
| One model for every action | yes | yes | yes | general |

## 3. Dependencies

- World 1 and its tokens: cards 102 and 102.1 (`runs/card102`,
  `runs/102.1/world1_vec_*.npy`).
- The transformer with a relative-position bias (Shaw et al. 2018, in
  LITERATURE under series 102). 102.1's numbers are the reference.

## 4. Data check

- **Training:** 3,000 episodes, 300,000 steps, 6,589 pushes, 545 lamps lit
  and 392 put out.
- **Test:** 1,000 episodes, 100,000 steps, 2,132 pushes, 204 lit and 140 put
  out (102.1).
- Move steps are 80% of all steps. The share of them that are blocked is
  counted before the run.

## 5. Feasibility gate

- **Upper bound:** arm R on the evaluator's kinds as one-hot tokens. It must
  get the whole next state exactly right on at least 99.5% of test steps.
- **Trivial baseline:** "nothing changes".

## 6. Success criteria and prediction

Scored on the test worlds, decoding each predicted vector to its nearest
code:
1. **Lamps at new columns.** At pushes, a lamp's next code is right in at
   least 95% of the lamps that change and 95% of those that do not, in both
   of arm R's seeds.
2. **The agent's own motion in the same model.** On move steps, the agent's
   place and the place it moves into are right in at least 99%, for blocked
   moves and moves that succeed alike.
3. **The whole next state** (all 81 tokens) is exactly right on at least
   99% of test steps.

**Prediction:** R passes all three in both seeds. A relation between
neighbours is simpler under relative attention than one per column, though
walls can still give the column back. A fails criterion 1 at column 7, as in
102.1.

**Decision rules:**
- **Keep** if R passes all three.
- **Revise** if only criterion 1 fails in one seed.
- **Stop** if R fails criterion 2 or 3, meaning one model cannot carry the
  moves.

**Budget:** four runs of about 8 minutes on the GPU, capped.

## 7. Result

Run 2026-10-13, 15,000 training steps of 128 transitions each, scored on the
99,000 test steps ([results.json](results.json)). A run took 2–10 minutes.
The first launch trained at 1.5 s a step, because the bias between
different frames was read with a gather whose backward pass adds atomically.
Written as a product, a step took 21 ms; the numbers are unchanged by it.

| Measure | R, 1021 | R, 7 | A, 1021 | A, 7 | Gate: kinds, R | Bar |
|---|---|---|---|---|---|---|
| 1. Lamps changing at pushes, right | **99.7%** | 81.6% | 100% | 98.3% | 63.6% | ≥ 95% |
| — at column 7 | 100% | 58.7% | 100% | 95.2% | 33.3% | |
| 1. Lamps not changing at pushes, right | **98.5%** | 94.9% | 96.3% | 94.9% | 98.1% | ≥ 95% |
| 2. Moves that succeed: agent's place and target right | **100%** | 100% | 100% | 100% | 100% | ≥ 99% |
| 2. Blocked moves: the same | **100%** | 88.4% | 100% | 99.98% | 98.5% | ≥ 99% |
| 3. Whole next state right (63 world tiles) | **99.95%** | 97.7% | 99.90% | 99.86% | **99.54%** | ≥ 99%; gate ≥ 99.5% |
| Final training loss | 0.00002 | 0.0005 | 0.00002 | 0.00001 | 0.00004 | |

"Nothing changes" gets the whole next state right on 37.6% of test steps.
The tests hold 354 lamps that change at pushes, 2,723 that do not, 61,826
moves that succeed and 17,412 blocked moves. The table scores the 63 world
tiles; the other 18 tokens never changed.

- **Gate: pass,** just (99.54% against 99.5%). Fed the evaluator's kinds as
  one-hot vectors, the model still missed most lamps lighting at columns 6
  and 7. One-hot kinds give the same model a harder problem than the
  encoder's vectors, not an easier one.
- **Arm R, seed 1021: pass** on all three criteria.
- **Arm R, seed 7: fail** on all three. It never fit its own training data:
  its loss spiked again and again and ended 25 times above the other runs'.
  The failure lies in optimisation, not in what the model can represent.
- **Arm A,** raw positions as inputs: criteria 2 and 3 pass in both seeds.
  Criterion 1 passes in seed 1021, and in seed 7 misses on lamps not
  changing by 0.1 point. The prediction that A fails at column 7 was wrong:
  95–100%.

One model with no part for moves carried walking, blocking, pushing and
the lamps in three of four runs, with both forms of position.

## 8. Decision

**Stop** (2026-10-13), arm R in this form. The rule says stop if R fails
criterion 2 or 3, and seed 7 fails both. The reason the rule gives, that
one model cannot carry the moves, is not what happened. Arm A, also one
model, carried the moves in both seeds, and so did R with seed 1021. A and
R differ only in the position term, and A fit its training data in both
seeds, so the term is the suspect. It is an unbounded network of the
position difference, added to every attention logit. Swin Transformer V2
(Liu et al. 2022) uses the same kind of term in a tamer form. The network
reads log-spaced differences, sign(d)·log2(1 + |d|)/log2(8), so that large
offsets are extrapolated less. In its released code the output is bounded,
16·sigmoid. The paper gives that form for transfer across window sizes,
not for stability; that it also trains in every seed here is the
hypothesis. Card [105.6](../105.6-bounded-position-bias/card.md) changes
that one component and runs three seeds. Arm A is no answer for the
series: its positions are raw numbers in the frame, which in Crafter's
world frame are absolute places (105.2's table at new places).
