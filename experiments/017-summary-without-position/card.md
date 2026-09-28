---
id: "017"
title: one shared spatial learner
rung: 0
serves: [P1, P4, P12, P21]
status: done
verdict: fail
arch_version: 2
date: 2026-09-27
---

# 017: one shared spatial learner

## 1. Question

Can one learned state, retaining a spatial feature map and nonspatial
context, support both the existing condition recursion and reliable walking?
Before rung 1; serves P1 (features grounded in action success), P4 (keep
useful spatial distinctions), P12 (conditions down to actions), and P21
(walking with bounded computation). The user approved replacing this
card and the representation on 2026-09-27; this preserves the direction.

Cards 012 and 016 isolate walking: the same learned conditions achieve
95% with exact walking, versus 30.2% with run 5's walking and 48.4% with
card 016's. More training did not improve the latter. The old card 017
inventory probe is superseded, not evidence for this architecture.

## 2. What changes

One component: the learned state and the readouts needed to use it.
The condition tree, its evidence rules, and its recursive chooser remain.
The first stage transfers run 5's discovered tree; it does not rediscover it.

```
before: top-down encoder -> conditions; separate ego encoder -> walking
after: ego pixels -> shared encoder -> (spatial map, context)
                                   -> conditions / readiness / walking
```

[Architecture version 2](../../ARCHITECTURE.md) specifies the tensors and
priors. Every 8-pixel patch uses the same encoder. Room features remain a
13 × 13 map. Context combines their mean with features of the inventory
patch, before any mixing between room patches. Thus context is invariant
to permuting room patches; averaging overlapping convolutions was not.
One readiness scorer reads the spatial arrangement within six tiles of the
queried pose, in its orientation, together with context. Its centre patch
is excluded so the actual avatar is not required at a hypothetical pose.
The original eight-neighbor readout lost necessary remote spatial relations;
the exact input audit and the isolated comparison are specified below.

One learner receives all task gradients. No frozen condition model is
consulted during inference. A delayed target copy, when walking training
is enabled, is only a copy of this same learner.

## 3. Dependencies

- Cards 005, 007–008, 010, 012: learned achievement and condition recursion.
- Card 016: whole-room egocentric renderer and fixed centre readout; its
  geometry checks passed. Its walking gate did not fully pass.
- VIN (`1602_02867`): spatial computation and a learned local recurrence.
  This does not establish that our particular readiness field is learnable.
- Fixed-horizon TD (`1909_03906`): a horizon bootstraps from the previous
  horizon. Its convergence proof does not cover our shared recurrent weights.

## 4. Data check

Use the existing key-world collection policy: random play, one third play
starts, six actions including drop, and the full room. Hold wall column 5
out of new learner training. Run 5 already saw that column; report this as
transfer of a learned teacher, not wholly unseen-geometry discovery.
The bounded screen collects 1,000 training and 250 independent test episodes.
Select 512 uniform endpoint frames plus up to 64 teacher-positive frames
per way, per split. Report class counts for each metric; fewer than 20
positives or negatives makes that metric unassessable. Main training,
if all gates pass, uses run 5's 30,000-episode policy.

## 5. Feasibility gate

Run stages in order; no later stage starts after a failed prerequisite.

1. **Structure:** context permutation invariance, map/orientation covariance
   of readiness, inventory isolation, and gradients from each head into the
   same encoder must pass executable tests.
2. **Component upper bound:** fit exact readiness at every valid pose and
   exact conditions, only in this evaluator-labelled arm. On held-out frames,
   readiness recall ≥ 99% and false positives ≤ 1%, separately at the agent,
   other poses and wall column 5; condition recall ≥ 99% and false positives
   ≤ 1% for every accepted node. Budget: 2,000 updates, at most 8 minutes for
   the screen. Insufficient coverage or budget is reported explicitly.
3. **Learned component:** a fresh learner fits only cached run 5 predictions
   at observed poses. Require readiness and conditions ≥ 95% recall and
   ≤ 5% false positives against evaluator truth on the same breakdowns.
   Also report agreement with teacher labels, and the teacher's own errors.
4. **Walking upper bound:** after both component arms pass, attach the
   existing recurrence and fit evaluator distances; ≥ 99% moves closer for
   every way on held-out frames. Profile and record its budget before running.

Trivial readiness/condition baselines: always false has 0% recall and 0%
false positives; always true has 100% for both. Report their class counts.
Walking compares with random moves (about 45% closer in card 016).

## 6. Success criteria and prediction

The main joint run is conditional on all gates, with evaluator labels absent.
Keep the tree fixed to isolate representation transfer and retention.
Train condition/readiness predictions and fixed-horizon walking together;
use the shared learner for all decisions in 500 independent new layouts.

1. **Grounding and retention:** every accepted condition and readiness field
   retains ≥ 95% recall and ≤ 5% false positives on the component breakdowns.
2. **Walking:** ≥ 98% moves closer per way on seen wall columns, ≥ 95% on
   held-out column 5; card 016 scored 61–98% and 69–96%, respectively.
3. **Acting:** ≥ 90% successes, compared with card 016's 48.4%, run 5's
   30.2%, exact walking's 95%, and random play's 0.4%.

Prediction: structural tests pass; locality and transfer of noisy teacher
labels are the component risks. An inventory probe cannot establish walking.
A longer stage needs a concrete budget recorded here before launch; runs
above 30 minutes are handed to the user. No parameter sweep is included.

## 7. Result

| Check | Result | Interpretation |
|---|---|---|
| Structural tests | 4 of 4 passed: context permutation invariance, readiness rotation, inventory isolation, shared gradients | The implemented state has the stated structure; no competence claim |
| Coverage before fitting | 1,053 training and 882 test frames; goal-ready positives at the agent: 23 train / 9 test; key-ready: 15 train / 6 test, against 20 required per metric | Insufficient coverage; no upper-bound fit or learned run started |
| Deeper ways | Some test breakdowns have zero true ready poses, against 20 required | Their recall is unassessable; do not treat absence of examples as success |
| Exact input audit | 31 door-readiness signature groups have opposite labels; 82 of 302 positive poses participate, among 200,500 poses | Radius-one readiness loses information present in the spatial map |
| Wider-readout paired fit | Training readiness 100% versus 50% local-only; test readiness 17.97%, conditions 6.25%, against 99% required | More spatial input permits fitting, but training permits a geometry shortcut |

The initial integration check caught and fixed an evaluator shape mismatch:
card 016's label maps include the inventory row; this learner's spatial map
does not. Coverage was then checked without training. Raw counts and the
superseded pooling probe are preserved in `results.json`.

Validation: the repository suite passes 52 tests, including the four shared
state checks and three checks of masking, class balance and missing classes.

## 8. Decision

**Revise.** Keep the shared map and nonspatial context, but reject local-only
readiness: equal inputs require opposite answers. A wider readout fits the
training pairs but fails geometry transfer. [Card 018](../018-readiness-goal-binding/card.md)
holds geometry fixed while changing goal position to isolate that shortcut.
The full-tree coverage limitation remains; no walking result is claimed.

## Appendix: implementation and commands

### Readiness input audit and bounded comparison (declared before fitting)

`audit_inputs.py` groups queried poses by everything the radius-one readiness
head can see: oriented neighbor tiles, the room's tile histogram and inventory.
Equal signatures guarantee equal input information for any patch encoder.
On 200,500 valid poses, the door way has 31 groups with opposite exact labels;
82 of its 302 positive poses lie in these groups. Direct simulator calls
verify the witnesses. In one, opening the door makes the goal reachable;
in the other, the goal is already reachable on this side of the door.
The shared map retains the distinction; the old readiness readout discards it.

Only readiness's spatial extent changes: radius one versus radius six,
implemented as an oriented masked convolution. Both retain the same shared
patch encoder, context and condition head. This is a component correction
under CHARTER rule 2; it does not change condition recursion or its labels.

Before any fit, construct 128 training and 128 held-out layout pairs with
identical radius-one inputs and opposite true readiness. Training uses wall
column 3; evaluation uses column 4. Column 5 has a one-cell-wide room whose
local wall pattern reveals the side, so it cannot supply the required
matched pairs; this was checked and corrected before fitting. The full-tree
column-5 criterion above is unchanged. All labels here are evaluator
labels for this upper bound, never learned results. Train both arms for
1,000 updates, batch 16 (eight balanced pairs), learning rate 0.0003, seed 17.
Budget is 210 seconds total, with an external four-minute timeout.
Pass this isolated test only if radius-one accuracy is at most 50% (the
exact ceiling), radius-six readiness accuracy is at least 99%, and its shared
condition head accuracy is at least 99% on held-out pairs. Constant answers
score 50%. This diagnostic does not waive or replace the full-tree gates.

```bash
timeout 120s bin/prun python tools/card017/audit_inputs.py
timeout 240s bin/prun python tools/card017/bench_context.py
```

### Full-tree component screen

The shared learner is in `src/worldmodel/spatial_state.py`; the component
runner is `tools/card017/bench_shared.py`. Its checkpoint records whether
oracle labels trained it. The walking forward path is implemented; walking
training and autonomous evaluation wait for the component gate and are not
yet a runnable main experiment.

```bash
bin/prun pytest -q tests/test_spatial_state.py
timeout 540s bin/prun python tools/card017/bench_shared.py --seconds 480
```

`--coverage-only` checks counts without fitting. The runner also checks them
automatically and refuses to fit if either class has fewer than 20 examples
in a required metric. The default screen currently stops for this reason.

The superseded mean-pooling probe remains under `superseded_pooling_probe`
in `results.json`, and its historical script is `tools/card017/bench_pool.py`.
Its 93.9% inventory accuracy is unrelated to this card's pass criteria.
