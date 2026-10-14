---
id: "020"
title: conditions read the shared spatial computation
rung: 0
serves: [P1, P4, P12]
status: done
verdict: pass
arch_version: 2
date: 2026-09-28
---

# 020: conditions read the shared spatial computation

## 1. Question

Can the condition head retain the same spatial generalization as readiness
by reading its shared feature computation? Before rung 1; P1, P4 and P12.
Card 019's recurrent readiness reached 99.80% against 96.68% for a wide
filter, but its separate flat condition readout scored 97.85%, below 99%.

## 2. What changes

One component: condition readout. Keep card 019's recurrent readiness
processor, input state, data, labels, optimiser and training budget.
Replace `MLP(flatten(M), c)` with a linear readout of the same 128 spatial
features used for readiness, at the actual agent's pose. These features are
computed from M and c by the same weights. No extra encoder or teacher
features are introduced. Rerun the flat-readout arm as the control.

## 3. Dependencies

Cards 017–019 localize information loss, geometry shortcuts and offset
sharing. Card 019 passes readiness's 99% component bar, so this head is
attached to a tested dependency. No walking claim follows from this test.

## 4. Data check

Card 018's exact same counterfactual pairs: 128 training layouts / 512
frames and 128 test layouts / 512 frames, equal positive and negative
counts. Train wall columns 3 and 4; test column 5. Root goal positions
are varied only in this evaluator-labelled diagnostic. Encoder initialization
comes from run 5, which saw column 5; this limits the transfer claim.

## 5. Feasibility gate

The exact simulator labels all pairs. Shared recurrent features already
predict readiness at 99.80%; in this narrowly controlled test, whether
the goal is already reachable is the complementary label. The heads still
learn separate outputs; that relation is not supplied to the model.
Constant predictions score 50%; the flat condition head scores 97.85%.

## 6. Success criteria and prediction

1. Shared-spatial condition accuracy ≥ 99% on test pairs, against 97.85%.
2. Readiness accuracy remains ≥ 99%, against 99.80% in card 019. Report
   positive recall and false positives, not just overall accuracy.

Prediction: both heads generalize when both use the spatial computation.
Both arms train for 1,000 updates, batch 16, Adam 0.0003, seed 17. Budget:
210 seconds total, external four-minute timeout. This tests one condition
and its readiness; full-tree retention and learned walking remain untested.

## 7. Result

| Criterion | Flat condition readout | Shared spatial readout | Required / verdict |
|---|---|---|---|
| Conditions, held-out frames | 98.05% | 99.61% | ≥ 99% / pass |
| Readiness, held-out frames | 99.80% | 99.61%; recall 99.61%, false positives 0.39% | ≥ 99% / pass |

Both arms fit both training targets at 100%. The comparison took 16.9
seconds, against a 210-second budget. The repeated control's condition score
differs slightly from card 019's 97.85%; neither passes. This is one seed,
one controlled condition and its readiness, 512 held-out frames in 128 new
layouts. It is not full-tree retention, learning from experience, or walking.
The repository suite passes 56 tests, including the exact local-input
counterexample and gradients from both heads into the shared spatial processor.

## 8. Decision

**Keep.** Adopt the shared spatial readout as experimental architecture 3.
Both required heads now pass the same controlled test. Keep the full-tree
gate and ordinary-experience evaluation as prerequisites for any walking
claim; this component result does not bypass them.

## Appendix: command

```bash
timeout 240s bin/prun python tools/card017/bench_context.py --counterfactual-goals --condition-comparison --out runs/020_shared_conditions
```
