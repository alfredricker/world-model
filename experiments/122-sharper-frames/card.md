---
id: "122"
title: sharper frames — the same change, touch and extent signals from 256-pixel renders, for the cube and the buttons
rung: 2
serves: [P7, P4]
status: done
verdict: fail
arch_version: 27
date: 2026-11-10
---

# 122: sharper frames

Roadmap step 2 (event segmentation), rung 2. At 64 × 64 the cube covers
3 of the encoder's 196 patches and a button 2 or 3. That has limited
every card on this rung:
- card 115: every detector found the cube 6–15% of the time;
- card 116.2: SAM 2 could not outline objects at 64 pixels;
- card 119: lock flips 53%, cube 13%;
- card 120: a lock flip is seen on the handle, the button itself is the
  top patch in 2% of flips;
- card 121's gate: growing the cube or a button from a seed scores
  0.09–0.12, worse than the seed alone.

The user chose to fix resolution before more work on threshold rules
(2026-11-10, "let's do option 2"), and had already accepted sharper
renders than the benchmark's for card 116.3 (2026-11-03). Each step
runs under 10 minutes, so the card runs on AGENTS' short-run rule, with
its criteria written here before the run.

## 1. Question

Rendered at 256 × 256 and encoded on a 32 × 32 patch grid, do cards
119, 120 and 121's unchanged signals find the button itself and the
cube, and grow their extents? P7 (units), P4.

## 2. What changes

One component: the resolution of the frame the encoder reads. The
signals, windows, floors and block choices are those of cards 119, 120
and 121.

```
before:  64 × 64 frame → 196 × 196 input → 14 × 14 patches (one patch = 4.6 pixels of the 64-pixel frame)
122:     256 × 256 render → 448 × 448 input → 32 × 32 patches (one patch = 2 pixels of the 64-pixel frame)
```

Frames are rendered from the stored joint positions and button states
of OGBench's state dataset, which holds contact (card 120).

**Measures, strict** (card 120's lesson: no slack, and chance measured
under the same rule):
- *button found:* at a lock flip inside a touch, one of the button's
  own patches is in the top 1% of patches of card 120's signal;
- *top patch exact:* the signal's single top patch lies on the object;
- *extent:* card 121's gate, IoU of the region grown from a true seed.

Each is computed the same way on card 120's 64-pixel run, for a like
for like comparison.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The world can be seen at a higher resolution than the benchmark gives | yes, a render setting | no: 64 × 64 sprites are the world | no: the grid is the world | yes, the camera's | declared: departs from OGBench's 64 × 64 (CHARTER's current world); results are not comparable with published pixel baselines |
| An object of interest covers several patches | at close range | yes at 9 × 9 tiles | yes | depends on distance | declared limit |

## 3. Dependencies

- Cards 119, 120 and 121's code and settings; card 114's encoder at
  448-pixel input (as card 116.1 ran it).
- The 256-pixel renders of card 116.3 hold every fifth frame only, so
  they cannot give windows of consecutive frames. Frames are rendered
  again here, every frame, and not stored.

## 4. Data check

Two episodes (2026-11-10): rendering, encoding and signals take 3 s per
episode. Patches per object on the 32 × 32 grid: arm 215, drawer 83,
window 80, button 8 each, cube 5 (at 64 pixels: 48, 17, 20, 2–3, 3).
Transitions and touches are card 120's.

## 5. Feasibility gate

1. **Timing:** 100 episodes per set in about 5 minutes (measured: 3 s
   per episode).
2. **Upper bound:** each strict measure with the arm's patches removed
   by the true mask.
3. **Chance:** each strict measure with the signal's patches shuffled.

Result of the gate: step 1 passed (296–299 s per set); steps 2 and 3
are in section 7's tables.

## 6. Success criteria and prediction

Validation episodes, floors from episodes 900–999:
1. **The button itself:** found (top 1% of patches) at 60% or more of
   lock flips. Compared with the same measure at 64 pixels and with
   chance.
2. **The cube:** card 120's top patch exactly on the cube at 60% or
   more of cube moves (64 pixels: 35%; chance there 2%).
3. **Small extents:** grown from a true seed in block 12, IoU at least
   0.4 for the cube and for each button (64 pixels: 0.09–0.12).

Reported: the drawer and window on every measure, and card 119's
thresholded transitions F1.

**Prediction.** Criteria 1 and 2 pass: a button now has patches that
hold nothing but button, and the cube more than doubles. Criterion 3
probably fails: card 121's gate showed the flood comes from the
threshold rule, which resolution does not change.

**Budget:** 5 minutes per set to render, encode and compute; 3 minutes
to score. No training.

## 7. Result

Run 2026-11-10. Patches per object on the 32 × 32 grid over all frames:
arm 203, window 83, drawer 73, cube 8, button 8.

**Card 120's touch signal, strict.** "Found" is a patch of the object
in the top 1% of patches (2 of 196, 11 of 1,024).

| | 64 px: top patch exact | 256 px: top patch exact | 64 px: found | 256 px: found | 256 px: found, arm masked | 256 px: found by chance |
|---|---|---|---|---|---|---|
| Lock: the button | 2% | 8% | 3% | **27%** | 31% | 8% |
| Lock: the handle | 86% | 88% | 91% | 99% | 98% | 57% |
| Cube | 36% | **61%** | 53% | 81% | 67% | 12% |
| Drawer | 27% | 80% | 50% | 98% | 98% | 66% |
| Window | 52% | 64% | 74% | 95% | 98% | 62% |

**Extent grown from a true seed, block 12** (IoU):

| | Grown, 64 px | Grown, 256 px | Seed alone, 256 px | k-means, 256 px | Best threshold per case, 256 px |
|---|---|---|---|---|---|
| Cube | 0.12 | **0.06** | 0.14 | 0.06 | 0.75 |
| Buttons | 0.09–0.12 | **0.07** | 0.13 | 0.06 | 0.58–0.59 |
| Drawer | 0.61 | 0.54 | 0.02 | 0.46 | 0.66 |
| Window | 0.49 | 0.48 | 0.01 | 0.52 | 0.60 |

1. **The button itself: fail** (27% against 60%; 3% at 64 pixels,
   chance 8%).
2. **The cube: pass** (61% against 60%; 36% at 64 pixels).
3. **Small extents: fail** (0.06–0.07 against 0.4), as predicted: the
   threshold floods at either resolution, while the best threshold per
   case reaches 0.58–0.75.

Reported: card 119's fixed-window signal, thresholded, strict place:
transitions F1 0.63 (64 pixels: 0.61); at its best frame the cube is
the exact top patch 35% of the time (13%). The touch signal's top patch
is on the arm in 23% of touches (32%).

**After the run (diagnostic): why the button fails.** Criterion 1 asked
the button to outrank everything, and at a lock flip the handle changes
colour too, over more patches: 16 handle patches rank above the
button's best on average, against 1.0 on the arm and 1.6 elsewhere.
With the handle's body set aside, a button patch is in the top 1% at
95% of flips (64 pixels: 54%) and is the top patch at 46% (32%). So the
button is seen at 256 pixels. The measure, a single ranking per event,
could not show an event with two separate changed places.

Verdict: **fail** (criterion 2 passed; 1 and 3 failed).

## 8. Decision

**Revise: keep the resolution, change how an event is read.** Sharper
frames help every measure that depends on seeing a small or thin object:
the cube 36% to 61%, the drawer 27% to 80%, the button from invisible to
the second place that changes at a lock flip. They do not help the
extent rule, whose fault is its threshold. Both failures point the same
way: an event is not one top patch but a few separate regions that
change together (the button and the handle; where the cube left and
where it arrived), and at this resolution a region of change may itself
be the object's extent when the whole object changes (a button
recoloured, a cube gone). The next card should read an event as its
connected regions of change, and score those regions as objects, before
any more work on growing by similarity. Cost to carry: 256-pixel
renders leave OGBench's 64 × 64 observation, so planner scores later
will not compare with published pixel baselines unless the agent is
also run at 64.
