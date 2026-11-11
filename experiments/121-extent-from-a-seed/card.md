---
id: "121"
title: extent from a seed — a patch where something changed is grown into its object by similarity in a deep block
rung: 2
serves: [P7]
status: draft
verdict:
arch_version: 27
date: 2026-11-10
---

# 121: extent from a seed

Roadmap step 2 (event segmentation), rung 2. Card
[119](../119-variables-from-lasting-change/card.md) found that change
marks only the strip a slide sweeps, not the object: patches that
change together overlapped the drawer 0.17 and the window 0.32. It also
found that the encoder's early blocks know where a change is and its
deep blocks do not. This card tests the other half of that trade: deep
blocks are trained so that patches of one object have similar vectors
even when their pixels differ, so they may turn a seed into an extent.

Biological counterpart: the ventral visual stream's later areas respond
to an object as a whole across changes in its parts' appearance, while
early areas keep precise position (Ungerleider and Mishkin 1982;
DiCarlo, Zoccolan and Rust 2012).

## 1. Question

Given one patch where a condition changed, do the patches similar to it
in a deep block, and connected to it, cover the object that changed?
P7: an object is the unit that a change belongs to.

## 2. What changes

One component: how an object's extent is read, after cards 119 and 120
say where and when something changed. Nothing is trained.

```
119, 120:  block 2 → when and where a change happened → a seed patch
121:       block 12 of the same frame → similarity of every patch to the seed → threshold → connected region
```

- **Similarity:** cosine, after subtracting the frame's mean patch
  vector.
- **Threshold:** Otsu's method on the frame's 196 similarities, which
  splits them into two groups with no labels.
- **Region:** patches above the threshold connected to the seed.
- **Seeds in the main run:** the top patch of card 119's signal at each
  drawer and window slide, and of card 120's signal at each touch.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| An object's patches look alike in a deep block and unlike its surroundings | fails for look-alike neighbours (a cow in a herd, stone beside stone) | fails for adjacent tiles of one kind | fails for same-coloured neighbours | holds for distinct objects on a plain table | declared limit: appearance cannot separate look-alikes (LESSONS, cards 098.8, 104) |
| An object is one connected region | fails behind an occluder | holds | holds | fails behind the arm | declared limit |
| The encoder's deep block is given | pretrained, declared (C1) | the same | the same | the same | general |

## 3. Dependencies

- Card 114's encoder; card 115's masks; card 119's signal and its
  finding about depth; card 120's signal.
- Object extent from the similarity of frozen self-supervised patch
  features is established (Caron et al. 2021, DINO; Siméoni et al.
  2021, LOST, which grows a region from one seed patch). Neither is in
  LITERATURE yet; to be added if the card is kept.

## 4. Data check

400 validation frames of the visual dataset (every 250th). Patches per
object: arm 48, window 20, drawer 17, cube 3, buttons 2–3. A seed alone
already scores IoU 0.34–0.50 on the cube and buttons, and 0.06–0.07 on
the drawer and window.

## 5. Feasibility gate

Run 2026-11-10 (`tools/card121/gate.py`, under a minute). The seed is a
random patch on the true object, so only the growing step is tested.
Mean IoU with the object's true patches:

| Object | Seed alone | Grown, block 2 | Grown, block 12 | k-means, block 12 | Best threshold per case, block 12 |
|---|---|---|---|---|---|
| Drawer | 0.07 | 0.45 | **0.61** | 0.49 | 0.74 |
| Window | 0.06 | 0.34 | 0.49 | 0.56 | 0.65 |
| Arm | 0.02 | 0.33 | 0.44 | 0.49 | 0.54 |
| Cube | 0.41 | 0.04 | 0.12 | 0.14 | 0.77 |
| Button 0 | 0.50 | 0.08 | 0.09 | 0.09 | 0.76 |
| Button 1 | 0.34 | 0.09 | 0.12 | 0.12 | 0.47 |

- **The property is there and mostly grows with depth:** drawer 0.45,
  0.31, 0.51, 0.61 and window 0.34, 0.27, 0.41, 0.49 on blocks 2, 4, 8,
  12.
- **For the large objects it works:** well above the seed and above
  card 119's groups from change (0.17 and 0.32). It beats k-means on
  the drawer and not on the window.
- **For the small objects it fails:** the label-free threshold floods
  into the background, and the result is worse than the seed alone.
  The best threshold per case would reach 0.47–0.77, so the similarity
  holds the information and the threshold rule loses it.
- **The ceiling is modest:** even the best threshold per case reaches
  0.65–0.74 on the drawer and window at this resolution.

**Gate result:** passed for the drawer and window, failed for the cube
and buttons.

## 6. Success criteria and prediction

With seeds from cards 119 and 120, not from the truth:
1. **Drawer and window:** mean IoU at least 0.8 of the true-seed values
   (0.49 and 0.39), and above card 119's groups (0.17 and 0.32).
2. **Seeds on the arm are told apart:** where the seed lies on the arm
   (32% of touches), the grown region overlaps the arm more than any
   other object in at least 80% of cases, so a region can be named
   "self" later by card 117's influence.

The cube and buttons are reported, not judged: the gate shows this
threshold rule cannot pass them.

**Prediction.** Criterion 1 passes for the drawer and is close for the
window. Criterion 2 passes. A seed on the edge of an object, shared
with the background, is the main risk.

**Budget:** encoding the frames at seeds, 1 minute; growing and
scoring, 2 minutes.

## 7. Result

Appended after the run.

## 8. Decision

Appended after the run: keep, revise or stop.
