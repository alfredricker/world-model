---
id: "146"
title: grouping as settling, on a benchmark with published numbers
rung: 2
serves: [P7]
status: done
verdict: uninterpretable
arch_version: og4
date: 2026-11-19
---

# 146: grouping as settling, on a benchmark with published numbers

Roadmap step 2 (objects), rung 2. First card of the direction the user
opened on 2026-11-19: energy modules whose settled states are objects
([the write-up](../../../theory/thousand_brains_energy_formulation.md),
sections 6 and 7). The user agreed the test ladder ("that sounds like a
good plan": a published small benchmark first, then our own sprite world,
then OGBench's renderer with factors swept) and asked for experiments
("let's try some experiments"). This is the ladder's first step. It is a
test bed for one component, as the memory world was (card 136); whole
cards are still judged on OGBench. CHARTER's "Current direction" is not
changed by this card.

Biological counterpart: grouping of image parts by mutual attraction and
repulsion that settles (binding by synchrony in visual cortex, von der
Malsburg; Singer and Gray; disputed), here in the form of a Hopfield
network whose stable states are the groupings.

## 1. Question

Given patch features that are known to hold objects (a trained AKOrN's),
does settling an energy of pairwise attraction and repulsion between
patches find the objects as well as the published read-out, which clusters
the features afterwards and is told how many clusters to make? P7.

## 2. What changes

Nothing in og4. New, in `tools/card146/`: the first two terms of section
7's energy, as a read-out over features computed by someone else's model.

```
published (AKOrN):  features -> average-linkage clustering, told the number of clusters (4 here)
arm A:              features -> couplings J_ij = scale * (cos(f_i, f_j)^sharp - tau), optionally only between
                    patches within 2 of each other with every farther pair repelling
                 -> each patch holds a belief over K = 8 objects; descend
                    E = -1/2 sum J_ij <p_i, p_j> + (1/beta) sum <p_i, log p_i>   with beta rising, noise falling
                 -> a patch's object is its largest belief.  K is a ceiling, not the count.
```

The energy is the Hopfield–Tank form that CRAM learns, with patches as its
cells and object identities as its values. Section 7 wrote the state as a
unit vector per patch (AKOrN's form); the gate below fails that form, so
the main run uses beliefs.

Arm R (reference, not ours): AKOrN's released code (commit `eabbe27`),
attention couplings, one block, 50 epochs, on Tetrominoes: 60,000 images of
32 × 32 pixels, three coloured tetrominoes on black, on an 8 × 8 patch grid.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The image comes as patches with a feature each | yes | yes | yes (cells) | yes | general |
| No more than K things in view | no fixed bound | no fixed bound | no fixed bound | no fixed bound | declared: K is a ceiling set above the count; the cost of a ceiling far above it is reported (K = 16) |
| One still image, nothing moves | no | no | no | no | declared: time enters with the sprite world, the ladder's second step |
| The features already separate the objects | not shown | not shown | not shown | not shown (cards 116–145) | declared: this card tests the settling, not the features; learning the couplings is the next card |
| The background is one more thing to group | yes | yes | yes | yes | general; it is not scored |

## 3. Dependencies

- **AKOrN** (`2410_13821`, LITERATURE): established, with code. Its
  published Tetrominoes numbers are FG-ARI 88.6 ± 1.7 and MBO 56.4 ± 0.9.
  Arm R must reproduce them here before its features are used (criterion 1).
- **The data:** DeepMind's multi-object Tetrominoes, converted without
  TensorFlow by `tools/card146/convert_tetrominoes.py`, following AKOrN's
  converter (first 64 records unused, 320 test, 10,000 validation, 60,000
  training).
- **Scoring:** FG-ARI (agreement of the grouping with the true objects over
  object pixels only; 100 is perfect, 0 is chance) and MBO (mean overlap of
  each true object with its best group), as in AKOrN's `eval_obj.py`. My
  re-implementation gives the same value as their script on a test model
  (66.6 and 66.6).
- **Environment** (git-ignored, rebuilt by hand): `data/ext/akorn` (the
  clone, with `data/tetrominoes` and `runs` linked to this repository's
  `data/tetrominoes` and `runs/146/akorn_runs`); `data/ext/venv146` (a venv
  whose `.pth` file points at the older project environment, plus
  `accelerate ema_pytorch timm einops tensorboard fastcluster tqdm`).
  `tools/card146/run.sh <seconds> …` applies the caps of CHARTER rule 5.

## 4. Data check

Every test image has three objects. On the 8 × 8 grid an image has on
average 33.7 background patches and 10.1 patches per object. Objects are
not small here; small objects are a factor for the sprite world.

## 5. Feasibility gate

`tools/card146/gate.py`, 320 test images, results in `runs/146/gate.json`.
"Merged" is the share of images where two of the background and the three
objects mostly carry one label.

| Check | FG-ARI | MBO | Merged |
|---|---|---|---|
| Ceiling of the 8 × 8 grid: each patch given its main object | 96.4 | 59.3 | 0% |
| Trivial: one group | 0.0 | 9.4 | 100% |
| Trivial: patch colour, average-linkage clustering, 4 clusters | 56.2 | 55.9 | 79% |
| **Upper bound, beliefs:** perfect couplings (+1 same object, −1 different), all pairs, K = 4 | 96.4 | 59.3 | 0% |
| The same, K = 8 | 96.4 | 59.3 | 0% |
| Beliefs, perfect couplings only within 2 patches, K = 8 | 73.4 | 46.5 | 59% |
| The same with every farther pair repelling at 0.3 | 96.0 | 59.2 | 0% |
| Unit vectors (section 7 as written), perfect couplings, all pairs, K = 4 | 68.3 | 45.5 | 52% |
| The same, K = 8 | 85.2 | 53.7 | 16% |

Result of the gate, before the main run: **passed for beliefs, failed for
unit vectors.**

- With perfect couplings the belief form reaches the grid's ceiling, with
  K equal to the count and with K twice it.
- The unit-vector form does not, with the same couplings. A repelling pair
  of unit vectors is best satisfied by pointing opposite ways, and only two
  groups can be opposite: with the grouping term alone (groups of 34, 10, 10 and 10
  patches, couplings of ±1/64), the state with all three objects together
  opposite the background has lower energy (−22.1) than the right one (−10.9). The assignment term has to outweigh that, and
  across the settings tried it did so only in part (21 to 85). The main run
  does not use this form.
- Couplings between near patches only are not enough: two objects that do
  not touch are never told they differ. A repulsion between every far pair
  repairs it (96.0), at the price of splitting the background.
- Reference arm, timing: one epoch of AKOrN's training takes 14 seconds
  here (so 50 epochs about 12 minutes); after one epoch it scores FG-ARI
  66.6, and arm A's script runs end to end on it.

## 6. Success criteria and prediction

Test images, three seeds of arm R (1234, 1235, 1236), means over seeds.
Arm A's parameters (tau in 0.3, 0.5, 0.7, 0.9, 0.95; sharp in 1, 4, 16; all
pairs, or within 2 patches with far repulsion 0.1 or 0.3; scale 1/16; K = 8)
and the clustering threshold are chosen per seed on the first 320
validation images and scored once on the test images.

1. **The reference reproduces.** Arm R, scored by AKOrN's own script:
   FG-ARI ≥ 85.0 and MBO ≥ 54.0 (published 88.6 ± 1.7 and 56.4 ± 0.9; the
   bars are about two of its standard deviations below). If this fails the
   card is uninterpretable: the features are not the published ones.
2. **Settling matches the published read-out.** Arm A with K = 8, FG-ARI at
   most 3.0 points below average-linkage clustering told 4 clusters, on the
   same features. This would show that the grouping can be a settled state
   of an energy instead of a clustering step run afterwards.
3. **Without the count.** Arm A with K = 8, FG-ARI at least that of
   average-linkage clustering cut at a distance threshold chosen on the
   same validation images (the published read-out without the count), and
   merged in at most 10% of images.

Reported, not criteria: arm A with K = 4 and K = 16 under the same
parameters (what a ceiling far above the count costs); clustering told 8.

**Prediction.** Criterion 1 passes (85 to 90). Criterion 2: about even;
average linkage is forgiving of graded similarity, and one threshold tau
has to separate "same" from "different" for every pair. Criterion 3: about
even. On the one-epoch test model arm A scored 61.4 with K = 8 against 66.6
for clustering told 4, and 36.3 with K = 16, so a high ceiling may cost
more than the gate's perfect couplings suggest.

**Budget.** Arm R: three runs of about 12 minutes, one at a time. Arm A:
about a minute per seed. No run over 30 minutes.

```
AKORN=1 tools/card146/run.sh 1500 python train_obj.py --exp_name=tetrominoes_akorn_attn_s<seed> --seed=<seed> \
    --model=akorn --data=tetrominoes --J=attn --L=1 --ch=128 --psize=4 --epochs=50 --c_norm=none
AKORN=1 tools/card146/run.sh 300 python eval_obj.py --model=akorn --data=tetrominoes --model_imsize=32 --J=attn \
    --L=1 --T=8 --ch=128 --psize=4 --c_norm=none --model_path=runs/tetrominoes_akorn_attn_s<seed>/ema_49.pth
AKORN=1 tools/card146/run.sh 600 python ../../../tools/card146/arm_a.py s<seed> runs/tetrominoes_akorn_attn_s<seed>/ema_49.pth
```

## 7. Result

Approved by the user, 2026-11-19 ("approved"), after the gate. Three
reference runs of 50 epochs (about 10 minutes each), then arm A per seed.
Test images; seeds 1234, 1235, 1236; numbers in `results.json`.

| Read-out of the same features | FG-ARI, three seeds | Mean | MBO | Merged | Labels used |
|---|---|---|---|---|---|
| AKOrN's own script (clustering told 4, after PCA) | 89.3, 88.8, 89.7 | 89.3 | 53.7 | | 4 |
| Clustering told 4 (my script) | 89.9, 88.9, 89.8 | 89.5 | 53.7 | 14% | 4.0 |
| Clustering, distance threshold from validation | 89.9, 88.6, 88.3 | 88.9 | 53.8 | 14% | 4.2 |
| Clustering told 8 | 86.1, 84.0, 78.0 | 82.7 | 55.2 | 3% | 8.0 |
| **Arm A: settling, K = 8** | 66.1, 65.0, 64.5 | **65.2** | 34.4 | 29% | 6.9 |
| Arm A, K = 4, same parameters | 66.2, 65.5, 65.6 | 65.8 | 29.2 | 33% | 4.0 |
| Arm A, K = 16, same parameters | 36.0, 35.9, 35.6 | 35.8 | 28.3 | 36% | 14.1 |

| Criterion | Bar | Result | Verdict |
|---|---|---|---|
| 1. The reference reproduces | FG-ARI ≥ 85.0 and MBO ≥ 54.0 (published 88.6 ± 1.7, 56.4 ± 0.9) | FG-ARI 89.3; MBO 53.7 (54.1, 51.4, 55.7) | **fail**, on MBO by 0.3 |
| 2. Settling matches the published read-out | at most 3.0 below clustering told 4 (89.5) | 65.2, 24.3 below | **fail** |
| 3. Without the count | at least the threshold clustering (88.9); merged ≤ 10% | 65.2; merged 29% | **fail** |

**Verdict: uninterpretable, by the card's own rule for criterion 1.** The
headline measure reproduces on every seed (FG-ARI 88.8 to 89.7 against the
published 88.6 ± 1.7), but MBO is 2.7 below the published value and 0.3
below my bar; I do not know why. Criteria 2 and 3 compare two read-outs of
the same features, and arm A as declared fails both by a wide margin
whatever is made of the MBO gap.

**Where arm A's failure sits** (diagnostics, not criteria; seed 1234;
`tools/card146/` and the scratch scripts' output is summarised here):

- The pairwise signal is in the features. Among object patches the cosine
  separates "same object" from "different object" with AUROC 0.979
  (medians 0.90 and 0.69; object against background 0.71).
- The declared threshold grid (0.3, 0.5, 0.7, 0.9, 0.95) has no value
  between those two medians. At 0.5 every pair attracts and the descent
  returns one group (FG-ARI 0.0); at 0.9 most same-object pairs repel.
- The declared descent is too warm. With threshold 0.7 the true grouping
  has lower energy (−6.27) than the state the descent returned (−5.04), and
  so has the clustering's (−6.18): the minimum is in the right place and
  the descent does not reach it. The gate did not show this because its
  perfect couplings were ±1 and the real ones are about ±0.1.
- On validation images, threshold 0.8 with a colder descent (final beta
  300, or couplings four times larger) scores 89.2 to 90.4 with K = 8 and
  with K = 16, against 90.6 for clustering told 4, using 5.2 labels on
  average. The test images have not been scored with these settings.

## 8. Decision

**Revise.** The energy is not shown wrong: its minimum sits near the true
grouping when the threshold is placed between the two kinds of pair, and a
colder descent finds it on validation images with the ceiling at twice and
at four times the count. What failed is two settings fixed in section 6
before any real couplings had been seen, a threshold grid with a hole where
the answer is and a temperature set on perfect couplings. Card
[146.1](../146.1-grouping-as-settling-colder/card.md) repeats arm A on the
same three checkpoints with those two settings changed and the criteria
for arm A unchanged; its reference criterion is FG-ARI alone, with the MBO
gap declared. LESSONS and the graph nodes are written with 146.1, so the
pair is recorded once.
