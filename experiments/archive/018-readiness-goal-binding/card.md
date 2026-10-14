---
id: "018"
title: readiness must use the goal position
rung: 0
serves: [P1, P4, P12]
status: done
verdict: fail
arch_version: 2
date: 2026-09-28
---

# 018: readiness must use the goal position

## 1. Question

Can the shared spatial learner predict whether opening a door enables a
goal when training requires using the remote goal position? This is a
component diagnostic before rung 1, serving P1, P4 and P12, not a walking
or learned-condition-discovery result.

Card 017 identified equal local inputs with opposite readiness labels.
Expanding only readiness's spatial radius let it fit training pairs at 100%,
versus 50% for radius one, but it scored only 18.0% on a new wall column.
The existing condition head also failed transfer (6.25%). The wider map
can fit the answers; the data allow geometry to substitute for the goal.

## 2. What changes

One component: the evaluator-labelled upper-bound data. Architecture 2,
its shared encoder, losses, optimiser and radius-one control are unchanged.
For each layout and queried door side, render a pair differing only in
the goal's position: one goal on each side of the door, both outside the
local 3 × 3 neighborhood. Inventory, doors and other objects are identical.
The action's true readiness must change. This prevents constant geometry
within a pair from explaining the target. These constructed scenes are
confined to a probe; they are not added to the agent's experience buffer.

## 3. Dependencies

- Card 017's exact input audit: the radius-one head loses needed information.
- Its shared-state implementation and structural tests, architecture 2.
- The existing exact simulator, used only for construction and evaluation.
  The learner sees rendered RGB; no object or position labels are inputs.

## 4. Data check

128 training layouts with wall columns 3 and 4; 128 new test layouts with
column 5. Each gives four frames (two agent sides × two goal sides), with
256 positives and 256 negatives per split. Check equality of the local
input signatures and opposite simulator labels for every pair before fitting.
Run 5's convolution weights initialize both arms; they previously saw column
5, so this is transfer of the newly fitted readouts, not a pristine encoder
generalization claim. One seed, 17; no sweep.

## 5. Feasibility gate

Upper bound: direct simulator checks identify the correct answer for every
pair. The full RGB map distinguishes goal positions. The radius-one input
is identical within each pair, so every such classifier has a 50% accuracy
ceiling. Constant predictions also score 50%. The fit below tests whether
the shared pixel learner can learn to use the available distinction.

## 6. Success criteria and prediction

1. **Readiness:** radius-six held-out accuracy ≥ 99%, versus the local-only
   control's ceiling of 50%; report recall and false positives separately.
2. **Shared conditions:** the condition head's held-out accuracy ≥ 99% on
   whether the goal is already reachable. Both losses train the same encoder.
3. **Control:** radius-one readiness accuracy ≤ 50%, with exactly matched
   local inputs, showing that global layout is necessary for this test.

Prediction: changing the goal while holding geometry fixed removes the
previous shortcut. Train each arm for 1,000 updates, batch 16 (eight balanced
pairs), Adam at 0.0003. Budget: 210 seconds total, external timeout 240 seconds;
no run over 10 minutes. Passing is only an oracle-labelled component result.

## 7. Result

| Criterion | Local-only control | Radius-six head | Required / verdict |
|---|---|---|---|
| Readiness, held-out frames | 50.00% | 96.68%; recall 93.36%, false positives 0% | ≥ 99% / fail |
| Shared condition head | 99.22% | 98.44% | ≥ 99% / fail for the tested wider arm |
| Local-only ceiling | 50.00% | — | ≤ 50% / pass |

Both heads of the wider arm fit training at 100%. Runtime was 14.2 seconds
for both arms, against a 210-second budget. All 17 readiness errors occur
when the goal is five tiles ahead; training positives place it two to four
tiles ahead. The wide kernel gives each offset separate parameters, so the
unseen offset lacks a trained rule. Raw metrics and errors are in `results.json`.

## 8. Decision

**Revise.** Controlled goal placement raises held-out readiness from card
017's 17.97% to 96.68%, but the declared criteria still fail. Preserve this
diagnostic's paired construction and test sharing the readiness computation
across distance in [card 019](../019-readiness-spatial-sharing/card.md).

## Appendix: command

```bash
timeout 240s bin/prun python tools/card017/bench_context.py --counterfactual-goals --out runs/018_goal_binding
```
