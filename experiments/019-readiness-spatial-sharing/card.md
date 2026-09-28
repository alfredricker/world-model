---
id: "019"
title: share readiness computation across distance
rung: 0
serves: [P1, P4, P12]
status: done
verdict: fail
arch_version: 2
date: 2026-09-28
---

# 019: share readiness computation across distance

## 1. Question

Does sharing local computation across distance fix the readiness head's
unseen-offset errors? A component diagnostic before rung 1, serving P1,
P4 and P12. Card 018's wider head reached 96.68% readiness and 98.44%
condition accuracy, below 99%; all 17 readiness errors had a goal five
tiles ahead, whereas training positives put it two to four tiles ahead.

## 2. What changes

One component: the readiness readout. Replace its single wide convolution,
with separate weights per offset, by a masked 3 × 3 projection followed
by five applications of the same learned 3 × 3 feature update:
`h0 = ReLU(local(M) + context(c)); h_next = ReLU(h0 + conv(h))`.
The radius remains six, and the final projection still gives readiness
per way and pose. The initial kernel excludes the queried centre, but
subsequent steps can receive its information indirectly; this is declared.
Keep encoder, context, condition head, labels, splits and optimiser fixed.
The wide-convolution arm is rerun alongside as the control.

## 3. Dependencies

Card 017's exact information-loss audit; card 018's controlled goal-position
pairs and error analysis; VIN's shared local spatial computation (listed
in LITERATURE.md). This feature recurrence is our adaptation, not an exact
transition model, a Bellman operator, or a cited convergence guarantee.

## 4. Data check

Exactly card 018's data and seeds: 128 layouts / 512 frames per split,
256 positives and 256 negatives. Training wall columns 3 and 4, test 5.
Counterfactual goal positions give opposite readiness with the same local
view. All labels are evaluator labels, not new agent experience. Run 5's
encoder initialization previously saw column 5; retain that limitation.

## 5. Feasibility gate

Exact simulator answers and distinguishable full RGB maps provide the
information upper bound. Constant readiness and local-only readiness
score 50%. The wider head fits training at 100%; this test concerns spatial
generalization. No walking or full-tree stage is unlocked by this diagnostic.

## 6. Success criteria and prediction

1. Recurrent-readout readiness ≥ 99% on held-out pairs, against the wide
   head's 96.68%; report positive recall and false positives separately.
2. The unchanged condition head ≥ 99% on those frames (wide: 98.44%).

Prediction: shared spatial updates generalize to the untrained offset.
Both arms: 1,000 updates, batch 16, Adam 0.0003, seed 17. Budget: 210
seconds total, external four-minute timeout. One comparison, no sweep.

## 7. Result

| Criterion | Wide filter | Shared recurrence | Required / verdict |
|---|---|---|---|
| Readiness, held-out frames | 96.68% | 99.80%; recall 99.61%, false positives 0% | ≥ 99% / pass |
| Condition head, held-out frames | 98.44% | 97.85% | ≥ 99% / fail |

Both arms fit training readiness and conditions at 100%. Both arms together
took 17.3 seconds, against a 210-second budget. Sharing the local computation
fixes readiness's distance-transfer failure; the flat condition head remains
below its bar. Exact numbers are in `results.json`.

## 8. Decision

**Revise.** Retain the recurrent readiness component, which passed its bar,
and test the failing condition readout attached to it in
[card 020](../020-shared-condition-readout/card.md). Overall success is not
claimed because condition retention failed.

## Appendix: command

```bash
timeout 240s bin/prun python tools/card017/bench_context.py --counterfactual-goals --recurrent-comparison --out runs/019_spatial_sharing
```
