---
id: "105.6"
title: a bounded, log-spaced position bias — relative attention that trains in every seed
rung: 1
serves: [P6, P3, P7]
status: done
verdict: keep
arch_version: 27
date: 2026-10-13
---

# 105.6: a bounded, log-spaced position bias

Series [105](../105-one-transition-model/card.md), written in the overnight
session of 2026-10-13 under the user's approval of the series. Offline; no
agent. It follows [105.1](../105.1-one-model-world1/card.md) (stop).

## 1. Question

In 105.1 the series' model read positions only through a learned bias on
each attention logit, a network of the two tokens' position difference
(arm R). With seed 1021 it passed every criterion in breaking world 1. With
seed 7 it never fit its training data: its loss spiked again and again and
ended 25 times above the other runs'. The same model with raw positions as
inputs (arm A) fit both seeds. Suppose the bias network reads log-spaced
differences and its output is bounded, as in Swin Transformer V2. Does R
then train in every seed and pass 105.1's criteria? Rung 1; P6, P3, P7.

## 2. What changes

One component, the position term of arm R (arm "Rb"):
- **Input:** each difference d (in tiles) enters as
  sign(d)·log2(1 + |d|)/log2(8), not as d/8.
- **Output:** the bias is 16·sigmoid(network output), so it lies between 0
  and 16. The learned constant between two different frames is bounded the
  same way.

Everything else is 105.1's: the network's size (two hidden layers of 64),
the transformer, the loss, the data, 15,000 steps and the scoring. Swin V2
gives the log-spaced input for transfer to larger windows, not for
stability. Its bound appears only in the released code
(`16 * torch.sigmoid`). That the pair also trains in every seed is the
hypothesis under test.

An exact speed-up, not a change: when every transition in a batch has the
same position differences, the bias network runs once and is shared. In
world 1 the view is fixed, so this always holds. In 105.1, R took 3–4
times as long as A.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Positions compared only as differences within a frame | yes | yes | yes | declared (105) |
| A unit of distance, so that log-spacing has a scale | the tile | the cell | a declared length | declared |
| Bias bounded (0 to 16) | yes | yes | yes | declared, from Swin V2's code |

## 3. Dependencies

- 105.1's data, code and numbers (`tools/card105.1/run.py`, arm "Rb").
- Liu et al. 2022, Swin Transformer V2 (arXiv 2111.09883), section 3.3 and
  the released `swin_transformer_v2.py`. To be added to LITERATURE.

## 4. Data check

105.1's: 300,000 training steps and 99,000 test steps. The test holds 354
lamps that change at pushes, 2,723 that do not, 61,826 moves that succeed
and 17,412 blocked moves.

## 5. Feasibility gate

105.1's gate stands: the same model on the evaluator's kinds got the whole
next state right on 99.54% of test steps (bar 99.5%). The trivial baseline
"nothing changes" gets 37.6%.

## 6. Success criteria and prediction

105.1's criteria, in each of three seeds (1021, 7 and 3):
1. **Lamps at new columns.** At pushes, a lamp's next code is right in at
   least 95% of the lamps that change, and 95% of those that do not.
2. **The agent's own motion.** On move steps, the agent's place and the
   place it moves into are right in at least 99%, for blocked moves and
   moves that succeed alike.
3. **The whole next state** (the 63 world tiles) is exactly right on at
   least 99% of test steps.
4. **The fit.** The final training loss is under 0.0001 in each seed.

**Prediction:** all four pass in all three seeds. Bounded logits cannot
grow to dominate the softmax. Criterion 1's unchanged lamps are the
closest: 94.9–98.5% in 105.1.

**Decision rules:**
- **Keep** (Rb becomes the series' form) if all four pass in all three
  seeds.
- **Revise** if every seed fits (criterion 4) but criterion 1 fails in one
  seed.
- **Stop** otherwise. The position term then goes back to the user, with
  arm A's numbers.

**Budget:** three runs, each about 3–4 minutes with the shared bias, one
after another, capped (CHARTER rule 5).

## 7. Result

Run 2026-10-13, 15,000 training steps per seed, scored on the 99,000 test
steps ([results.json](results.json)). A run took about 2.5 minutes, against
8–10 for 105.1's R: the bias network now runs once per batch.

| Measure | Rb, 1021 | Rb, 7 | Rb, 3 | 105.1's R, 7 | Bar |
|---|---|---|---|---|---|
| 1. Lamps changing at pushes, right | 100% | 100% | 100% | 81.6% | ≥ 95% |
| 1. Lamps not changing at pushes, right | 98.75% | 98.9% | 96.7% | 94.9% | ≥ 95% |
| 2. Moves that succeed: agent's place and target right | 100% | 100% | 100% | 100% | ≥ 99% |
| 2. Blocked moves: the same | 100% | 100% | 100% | 88.4% | ≥ 99% |
| 3. Whole next state right (63 world tiles) | 99.97% | 99.97% | 99.89% | 97.7% | ≥ 99% |
| 4. Final training loss | 0.00001 | 0.00002 | 0.00002 | 0.0005 | < 0.0001 |

All four criteria pass in all three seeds. Seed 7's loss still spiked once
(0.012 at step 4,000), but it recovered within 2,000 steps; 105.1's R never
did. Lamps changing were right at every column, 7 included, in every seed.
The prediction held, including the closest call: unchanged lamps in seed 3
at 96.7%.

## 8. Decision

**Keep** (2026-10-13). The bounded, log-spaced position bias (arm Rb)
becomes the series' form, and 105.2–105.5 use it. One transition model,
reading positions only as differences, now carries walking, blocking,
pushing and the lamp's relation at columns never seen, in every seed. The
change touched only the position term. That suggests the unbounded bias,
not relative attention as such, caused 105.1's failure in seed 7. Three
seeds show this; they do not prove it.
