---
id: "146.1"
title: grouping as settling, with the threshold between the two kinds of pair and a colder descent
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-19
---

# 146.1: grouping as settling, colder

Roadmap step 2 (objects), rung 2. The revision that
[card 146](../146-grouping-as-settling-on-a-known-benchmark/card.md)
decided. Its runs take about a minute each and train nothing, so it falls
under the user's "let's try some experiments" (2026-11-19) and the
ten-minute rule; it was not separately approved. Biological counterpart:
as card 146.

## 1. Question

With the threshold placed between "same object" and "different object"
pairs and a descent cold enough to commit, does settling the energy read
the objects from a trained AKOrN's features as well as clustering that is
told the number of clusters? P7.

## 2. What changes

Two settings of card 146's arm A; the energy, the features (the same three
checkpoints), the data split and the scoring are unchanged.

```
146:    tau in 0.3, 0.5, 0.7, 0.9, 0.95; sharp in 1, 4, 16; couplings scaled by 1/16; beta rises 1 -> 30 over 200 steps
146.1:  tau in 0.7, 0.75, 0.8, 0.85, 0.9; sharp 1;          couplings scaled by 1/4;  beta rises 1 -> 300 over 400 steps
        both: all pairs, or within 2 patches with far pairs repelling at 0.1 or 0.3; K = 8; chosen per seed on the
        first 320 validation images, scored once on the 320 test images
```

Portability check: as card 146. One assumption is added.

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A few images with true objects exist to place the threshold | no | no | no | no | declared: the threshold is read from labelled validation images here; learning the couplings, the next card, has to replace it |

## 3. Dependencies

Card 146: its three AKOrN checkpoints (FG-ARI 88.8 to 89.7 by AKOrN's own
script, against the published 88.6 ± 1.7), its gate, its code. **Declared
gap:** those checkpoints' MBO is 53.7 against the published 56.4 ± 0.9, for
a reason I have not found. MBO is reported and no criterion rests on it.

## 4. Data check

As card 146.

## 5. Feasibility gate

- **Upper bound:** card 146's gate (perfect couplings reach the grid's
  ceiling, 96.4), and its diagnostic on seed 1234's validation images:
  threshold 0.8 with a colder descent scores 89.2 to 90.4 with K = 8 and
  K = 16, against 90.6 for clustering told 4.
- **Trivial baseline:** patch colour clustered into 4, FG-ARI 56.2; arm A as
  declared in card 146, 65.2.

Result of the gate, before the main run: passed (on validation images of
one seed; the test images have not been scored with these settings).

## 6. Success criteria and prediction

Test images, three seeds, means over seeds. Criterion 1 replaces card
146's (the one change of criteria; reason in section 3). Criteria 2 and 3
are card 146's, unchanged.

1. **The features are the published ones on the headline measure.** FG-ARI
   by AKOrN's own script ≥ 85.0. Already measured in card 146: 89.3.
2. **Settling matches the published read-out.** Arm A with K = 8, FG-ARI at
   most 3.0 points below clustering told 4 clusters (89.5).
3. **Without the count.** Arm A with K = 8, FG-ARI at least that of
   clustering cut at a distance threshold chosen on validation images
   (88.9), and merged in at most 10% of images.

Reported, not criteria: K = 4 and K = 16 under the same parameters; MBO;
labels used.

**Prediction.** Criterion 2 passes (88 to 90). Criterion 3 is about even:
the FG-ARI half is within a point either way, and the clustering baselines
themselves merge in 14% of images, so 10% asks for better than the
published read-out. K = 16 stays within 2 points of K = 8.

**Budget.** About two minutes per seed.

```
AKORN=1 tools/card146/run.sh 600 python ../../../tools/card146/arm_a.py s<seed> runs/tetrominoes_akorn_attn_s<seed>/ema_49.pth v2
```

## 7. Result

Run 2026-11-19, about a minute per seed. Test images; seeds 1234, 1235,
1236. Every seed chose threshold 0.8 with couplings between all pairs.

| Read-out of the same features | FG-ARI, three seeds | Mean | MBO | Merged | Labels used |
|---|---|---|---|---|---|
| Clustering told 4 | 89.9, 88.9, 89.8 | 89.5 | 53.7 | 14% | 4.0 |
| Clustering, distance threshold from validation | 89.9, 88.6, 88.3 | 88.9 | 53.8 | 14% | 4.2 |
| Clustering told 8 | 86.1, 84.0, 78.0 | 82.7 | 55.2 | 3% | 8.0 |
| **Settling, K = 8** | 89.2, 87.5, 86.7 | **87.8** | 53.5 | 14% | 5.0 |
| Settling, K = 16, same parameters | 89.5, 87.1, 86.9 | 87.8 | 53.2 | 15% | 5.0 |
| Settling, K = 4, same parameters | 87.8, 87.8, 88.3 | 88.0 | 51.3 | 18% | 4.0 |
| Settling as declared in card 146, K = 8 | 66.1, 65.0, 64.5 | 65.2 | 34.4 | 29% | 6.9 |

| Criterion | Bar | Result | Verdict |
|---|---|---|---|
| 1. The features are the published ones (FG-ARI) | ≥ 85.0 | 89.3 | pass |
| 2. Settling matches the published read-out | at most 3.0 below clustering told 4 (89.5) | 87.8, 1.7 below | **pass** |
| 3. Without the count | at least the threshold clustering (88.9); merged ≤ 10% | 87.8, 1.1 below; merged 14% | **fail** |

**Verdict: fail (2 of 3).**

- **The ceiling costs nothing.** With K at 8 and at 16 the score is the
  same (87.8) and about five labels are used. Clustering told 8 loses 6.8
  points (82.7 against 89.5); settling with a ceiling of 8 loses 1.7.
- **Against the published read-out without the count, settling is 1.1
  points short,** and it merges two of the background and the objects in
  14% of images, the same share as both clustering read-outs, where the bar
  asked for 10%.
- **The remaining error is in the couplings, not the descent**
  (diagnostic, not a criterion). The settled state has lower energy than
  the true grouping in 86 to 99% of images per seed, and in 94 to 99% of
  the images where settling scores below 0.9: the energy prefers the wrong
  answer there. Of the images settling merges (12 to 16%), clustering
  merges most of the same ones (9 to 12% merged by both).
- **The prediction held** for criterion 2 (88 to 90 expected; 87.8, a
  little under) and for K = 16; criterion 3 was called even and failed on
  both halves.

## 8. Decision

**Revise.** Settling an energy reads the objects from these features about
as well as clustering that is told their number (1.7 points below), with
only a ceiling on the number, and what error remains belongs to the
couplings: a cosine with one threshold, placed with labelled images. That
is the part to change next. The read-out is carried forward as it is: the
belief form, all pairs, a descent cold enough to commit. The next card
learns the couplings on this benchmark, with no labelled images to place a
threshold; its design and training signal need the user's go-ahead.
