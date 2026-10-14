---
id: "021"
title: full-tree gate for the shared spatial learner
rung: 0
serves: [P1, P4, P12]
status: abandoned
verdict:
arch_version: 3
date: 2026-09-28
---

# 021: full-tree gate for the shared spatial learner

## 1. Question

Can architecture 3 represent the entire existing condition tree, first with
exact supervision and then with cached learned labels? This serves P1, P4
and P12 before walking integration, not a capability-ladder rung.

## 2. What changes

The component-screen data coverage changes; the model and tree stay fixed.
Card 020 tested one condition. Card 017's full-tree screen stopped because
rare ready states were missing. Keep its experience sample and add a separate
evaluator set varying carry, object placement, door, switch and vase state.
Re-render rare positive queried poses with the agent at that pose. These
constructed states train only the upper bound and evaluate both arms.

## 3. Dependencies

Cards 017–020 establish structural checks and the narrow spatial fit.
Card 012 supplies the learned tree and cached labels. Its predictions are
measured, not assumed correct. The exact evaluator defines labels only;
passing its upper bound is a prerequisite for learned fitting. No walking
or fresh condition discovery is included. Sources remain those recorded in
LITERATURE.md; no new paper mechanism or architecture is introduced.

## 4. Data check

Collect 1,500 training and 500 test episodes using the existing random-play
policy; retain 1,024 uniform endpoints plus teacher-positive endpoints.
Training excludes wall column 5. Evaluator augmentation uses 192 independent
layouts per split, 12 configurations per layout, plus terminal goal poses.
Also enumerate ground distractor placements with the matching key held,
door closed and vase intact, to cover escaping a blocked corner by breaking
the vase before opening the door. This augments only the evaluator set.
For each way, retain up to 64 distinct positive map queries per wall column
and render them at the centre. Report counts before fitting. Test layouts
come from the independent test collection; no train/test layout is shared.

The learned tree includes `forward/toggle/toggle/toggle`. Under the exact
key-world definition this readiness is identically false: walking plus
toggles can only add access by breaking the single vase and opening the
single door. Switch toggles do not affect this rule; closing a door cannot
add access. Any reachable goal therefore needs at most two toggles, so the
third toggle cannot newly satisfy the preceding condition. Verify zero
support in the exact audit, retain this output, and score its false positives.
Its recall is undefined, never counted as perfect. All other outputs still
need at least 20 positive and 20 negative examples per required breakdown.

## 5. Feasibility gate

Require adequate coverage for conditions (including the root), readiness at
the agent, other valid poses, and held-out wall column 5. Verify the declared
zero-support exception; any contrary witness rejects that exception.
The upper bound fits exact condition and full-map readiness labels. Require
recall ≥ 99% and false positives ≤ 1% on each supported output/breakdown.
For the certified inactive readiness output require ≥ 20 negatives and
false positives ≤ 1%; it is not pruned or hard-coded in the learner.
Always-false and always-true predictions are the trivial baselines.

## 6. Success criteria and prediction

1. The coverage and upper-bound gates above pass on the full tree.
2. A fresh model trained only on cached labels from ordinary collected
   experience reaches ≥ 95% recall and ≤ 5% false positives on the same
   supported breakdowns; the inactive output has the same 5% false-positive
   limit. Report teacher accuracy and agreement separately.

Prediction: coverage can be repaired, but broad spatial generalization and
teacher errors may fail this stricter test. Use Adam 0.0003, batch 32,
2,000 updates per arm, seed 17. Each invocation is capped externally at
540 seconds; a coverage-only audit precedes fitting and is cached. The
fitting invocation has a 480-second internal cap and stops after a failed
gate. A failed fit can receive the duration diagnostic declared below;
it remains part of this card. No parameter sweep is included. The card
stays open until the user approves its closing decision.

## 7. Result

The initial coverage audit took 23.6 seconds and left two related readiness
breakdowns below the required 20 positives: both had six on wall column 5.
Their witnesses require breaking a vase before opening the door. Distractor
placement enumeration was added before model fitting; thresholds are unchanged.

| Criterion | Result | Verdict |
|---|---|---|
| Coverage | 11,037 training / 11,341 test frames; minimum supported readiness positives 157 / 128, against 20 required; inactive branch zero as predicted | Pass |
| Exact full-tree fit | 2,000 updates in 30.4 seconds; 20 required breakdowns fail the 99% recall / 1% false-positive limits | Fail |
| Duration diagnostic | 10,000 updates; worst training condition recall 99.47%, worst false positives 1.43%; held-out false positives reach 38.28%, against ≤1% | Fail |
| Learned-label arm | Not started because the upper bound failed | Not assessed |

After the initial 2,000 updates, held-out condition recall is 72.90–90.15%
and false positives 3.13–32.53%
across non-root conditions, against ≥99% and ≤1%. The supplied terminal root
is perfect. Training condition recall is only 78.32–92.33%, showing that
failure to fit is present before the geometry-transfer question. Readiness
has 9 failing breakdowns, including 9.01% false positives for pickup that
unblocks the goal, against ≤1%. Detailed counts and teacher errors are in
`results.json`. The cached teacher also misses some rare branches, so its
labels cannot simply be assumed sufficient for later learning.

The duration diagnostic took 157.1 seconds, against its 420-second internal
cap. It substantially improves training fit, but 7 training and 32 test
breakdowns still fail. At 10,000 updates, held-out non-root condition recall
is 97.64–99.49% and false positives are 2.05–38.28%. Pickup that unblocks
the goal has 80.18% readiness recall on wall 5, against ≥99% required.
More updates alone therefore do not establish full-tree generalization.

The error is present on familiar wall columns and observed experience too:
worst condition false positives are 38.85% on column 3, 17.48% on column 4,
48.66% on column 5, and 28.98% on the 1,471 sampled experience endpoints.
These are per-output maxima, not aggregate error rates. Constructed rare
states and held-out geometry are not the only sources of failure.
All 59 repository tests pass, with five existing warnings.

## 8. Decision

**Stop** (2026-09-28, the user). The upper bound failed: conditions and
readiness from architecture 3's shared spatial processor, trained with
exact labels, do not generalise to new frames (false positives up to 38%,
also on familiar wall columns). Its question, readiness for the full tree,
is removed by the direction change in
[card 022](../022-effects-post-mortem/card.md). The failure of the
conditions readout to generalise stands as evidence for later cards.

## Appendix: duration diagnostic, declared before its run

The initial 2,000-update fit has poor training-set condition recall, so
extend one unchanged training trajectory to 10,000 updates before attributing
failure to the representation. Only duration changes: same cached frames,
exact labels, uniform frame sampler, architecture 3, Adam 0.0003, batch 32
and seed 17. Fit from the same initialization with uninterrupted optimizer
state. Evaluate at 2,000, 6,000 and 10,000 updates. This is an upper-bound
diagnostic, not a learned-agent main run.

At 10,000 updates, require the existing ≥99% recall and ≤1% false-positive
limits on both training and held-out frames. The training set has no wall-5
breakdown; all supported conditions and other readiness breakdowns remain.
Keep the same inactive-output exception. Both training and held-out gates
must pass before learned-label fitting. Incomplete fitting is not a pass.

Prediction: training fit improves; held-out rare conditions may still fail.
The initial run profiles about 66 updates/second, so 10,000 updates should
take about 155 seconds plus evaluation. Internal cap: 420 seconds; external
timeout: 480 seconds. Learned fitting and walking remain stopped meanwhile.

The repeated 2,000-update checkpoint differs from the initial short run;
the duration comparison uses checkpoints within one uninterrupted fit:

| Updates | Worst training condition recall / false positives | Worst test condition recall / false positives |
|---|---|---|
| 2,000 | 72.76% / 5.01% | 67.18% / 19.20% |
| 6,000 | 98.75% / 0.82% | 95.80% / 33.10% |
| 10,000 | 99.47% / 1.43% | 97.64% / 38.28% |

Commands (each run separately):

```bash
timeout 540s bin/prun python tools/card021/bench_full_tree.py prepare --out runs/021_full_tree_covered
timeout 540s bin/prun python tools/card021/bench_full_tree.py fit --out runs/021_full_tree_covered
timeout 480s bin/prun python tools/card021/bench_full_tree.py duration --out runs/021_full_tree_covered
```
