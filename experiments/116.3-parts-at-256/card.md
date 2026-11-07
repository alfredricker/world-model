---
id: "116.3"
title: parts from SAM 2 at 256 pixels — slots trained on SAM 2's unlabelled segments of sharper renders, one slot per segment
rung: 2
serves: [P7]
status: draft
verdict:
arch_version: 27
date: 2026-11-03
---

# 116.3: parts from SAM 2 at 256 pixels

Card [116.2](../116.2-slots-from-sam-masks/card.md) stopped at its gate.
On the dataset's 64 × 64 frames, SAM 2 never outlined the large objects
(its best single mask: arm 0.21, window 0.26, drawer 0.39), so it could
not teach the slots. On the same scenes re-rendered at 256 × 256 from
the stored joint positions, SAM 2's best single mask overlaps the
buttons 0.85, the cube 0.73 and the drawer 0.63. The user chose this
route (2026-11-03), accepting that the agent then sees sharper frames
than the benchmark's 64 × 64.

At 256 px SAM 2 still splits the scene into about 31 segments per frame
(95th percentile 37). About 18 lie on the table and walls, 5.5 on the arm
(its links) and 3 on the window. So this card stops asking for one slot
per object. Its slots are **parts**, one per SAM 2 segment. Joining
parts into objects is learned from what moves together. Card 117 does
this for the arm ("the self is what every action moves"), and card
102.2 did it for tiles. Biological counterpart: early vision delivers
surfaces and parts; objects are grouped from them by common fate
(Spelke 1990: infants group by common motion, not by shape).

## 1. Question

Trained on 256-px renders with SAM 2's unlabelled segments as targets,
do slots split the scene into parts that respect each object's outline,
give the cube and each button a part of its own, and keep each part's
identity over time? Rung 2, P7.

## 2. What changes

Against 116.2's design: the input frames (256-px renders, encoded at
448), the slot count, and which segments are supervised. Encoder, slot
attention, the carry over time, the decoder and the three losses are
116.2's.

```
116.2:  64-px frames → DINOv2 at 448 → 8 slots;  SAM 2 on the 64-px frame upsampled
116.3:  256-px renders → DINOv2 at 448 → 32 slots; SAM 2 (16 × 16 points) on the 256-px render
        each patch = 8 × 8 render pixels; its segment = the one covering most of them
        the 32 smallest segments of at least 2 patches are each matched to one slot
```

- **Frames.** Every frame the slots see or are scored on is rendered at
  256 × 256 by OGBench's own camera, from the dataset's joint positions
  and button states: the training frames of 116.1 (every fifth of
  episodes 0–449) and its validation frames. The truth stays 116.1's
  64 × 64 segmentation, so scores compare with 116.1 and 116.2.
- **Slots: 32**, which covers the mean segment count. Where a frame has
  more than 32 segments, the largest are left unsupervised (table and
  wall pieces). Small regions read as figure, large ones as ground, as
  in Rubin's figure–ground.
- **Loss:** 116.2's. Hungarian matching of segments to slots per frame,
  WorldDP's Tversky loss (α 0.99, β 0.01, size weight (1/size)^0.2), plus
  reconstruction and 0.5 × slot contrast over time.
- **Teacher labels:** SAM 2 with 16 × 16 points (0.28 s per frame; 32
  points scored the same and took 1.0 s). It labels whole 4-step training
  segments for 25 minutes, about 5,000 frames, and keeps 10% back to
  watch the mask loss for overfitting. Batches are half labelled.
- **Not the benchmark's frames.** The agent no longer sees what
  `visual-scene` gives its agents, so planner scores later on are not
  like-for-like with GAS's 54.4%. To be declared when planning is scored.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Frames at a resolution the segmenter can read | holds (render size is a setting) | holds (render size is a setting) | holds (64 × 64 cells, flat colours) | holds (real cameras) | perception adapter setting |
| A supplied segmenter outlines parts without prompts | untested (voxel textures) | fails in "everything" mode (card 104) | holds | holds | adapter (C1 allows it); a world where it fails needs its own teacher |
| A fixed number of slots, 32 | no (many creatures, blocks) | partly | partly | partly | declared setting |
| Small regions are figure, large ones ground | partly (sky, terrain) | partly (grass) | partly | holds mostly | declared adapter setting, applied only beyond 32 segments |
| Parts persist between model steps | holds in view | holds | holds | holds | general |

## 3. Dependencies

- 116.1's truth, metrics, scorer and slot model; 116.2's teacher code and
  mask loss (WorldDP, `2606_08775`; Hungarian matching as in DETR, both in
  LITERATURE).
- OGBench's renderer, given the stored states. The 256-px render, read at
  64 × 64, differs from the dataset's frame by 6 of 255 per pixel (card
  116.2's gate).

## 4. Data check

As 116.1: the cube is in view in 94% of frames; each button is about 11
pixels of the 64 × 64 frame, so about 3 patches of the 32 × 32 grid. On
100 truth frames at 256 px, SAM 2 gives each button a segment of its own
in every frame and the cube in about half (116.2's gate follow-up).

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **Render and encode** the 90,090 training and 11,823 validation
   frames at 256 px (0.4 ms per render), then DINOv2 at 448.
2. **Scoring parts.** For each object, the slots with most of their
   pixels on it are taken together, and their union's overlap with the
   object is its **parts score**. If no slot is mostly on the object, the
   score is 0. Grouping by truth is the evaluator's, standing in for the
   learned grouping that comes later. A part that straddles an object
   and the table lowers the score; so does an object with no part.
3. **Teacher and bars,** on 300 truth frames: SAM 2's partition, scored
   as parts. Each object's bar is the smaller of 116.1's (0.7 × the
   32 × 32 ceiling: arm 0.62, cube 0.38, buttons 0.40, drawer 0.54,
   window 0.60) and 0.9 × the teacher's parts score.
4. **Trivial baseline:** k-means with 32 clusters on the new features,
   scored as parts.
5. **Smoke run:** 2 minutes with labelled segments. It checks that the
   mask loss falls and measures steps per second at 32 slots; if under 4,
   the batch is halved.

Result of the gate (2026-11-03, `tools/card116/{render256,parts}.py`,
`runs/116/parts_gate.json`):
1. Rendered in 52 s; encoded in 5 minutes. A render averaged to 64 × 64
   differs from the dataset's frame by 2.8 of 255 per pixel.
2. –4. Scores on the 2,000 truth frames (teacher: the first 300):

   | Parts score | Arm | Cube | Button 0 | Button 1 | Drawer | Window | Mean |
   |---|---|---|---|---|---|---|---|
   | Ceiling, 32 × 32 | 0.88 | 0.54 | 0.58 | 0.57 | 0.77 | 0.86 | 0.70 |
   | SAM 2 teacher (16 points, 0.20 s/frame) | 0.41 | 0.48 | 0.62 | 0.62 | 0.61 | 0.23 | 0.50 |
   | k-means, 32 clusters | 0.72 | 0.21 | 0.00 | 0.00 | 0.47 | 0.51 | 0.32 |
   | **Bar** | 0.37 | 0.38 | 0.40 | 0.40 | 0.54 | 0.20 | |

   - The teacher gives 29 segments per frame (95th percentile 32); a third
     of frames reach 32, where the largest are left out.
   - **The arm's and window's bars test little:** the teacher sets them
     below k-means (0.72 and 0.51). For those two, criterion 1 only shows
     that the slots do not lose the object. The test is on the cube,
     buttons and drawer, and on criterion 2 (mean 0.42 or more).
5. **Smoke runs.** At batch 32 the decoder (32 slots × 1,024 patches) ran
   out of GPU memory, so the batch is 16 (8 labelled, 8 not), at 7.3
   steps per second: about 13,000 steps in 30 minutes. On 228 segments
   labelled in 3 minutes, 8 minutes of training (3,500 steps):
   - the mask loss sat at 0.97 through warm-up, then fell to 0.27 by step
     3,000, with held-out segments at 0.28 (no overfitting yet);
   - scored as parts: buttons 0.49 and 0.50, drawer 0.56, window 0.50,
     arm 0.49, **cube 0.00**; mean 0.42 (k-means 0.32). Each object's
     best slot was the same on only 45–49% of clip steps.

## 6. Success criteria and prediction

On 116.1's held-out frames and clips, with the 64 × 64 truth:
1. **Parts respect objects:** every object's parts score at or above its
   bar.
2. **Better than clustering:** mean parts score at least 0.10 above
   k-means with 32 clusters.
3. **Kept over time:** on the clips, each object's best single slot
   stays the same on at least 90% of the model steps where it is visible.

Meeting all three gives card 117 tracked parts, a few milliseconds per
frame, whose grouping into objects it can learn from what moves together.

**Prediction.** Buttons, cube and drawer pass, as the teacher clears
their bars. The window is the risk: SAM 2 sees through its glass (best
single mask 0.14 at 256 px), so its bar may be set low by the teacher.
Criterion 3 is at risk for parts of the arm, which change shape as it
turns. After the smoke run: the cube (no part in 3,500 steps; SAM 2
gives it a segment in about half of frames) and criterion 3 (45–49%)
are the main risks. Both should gain from 20 times the labelled
segments and 4 times the steps, but a fail on either is likely.

**Budget:** gate about 25 minutes (render 1, encode 5, teacher 5,
k-means 3, smoke 2); labelling 25 minutes; training 30 minutes, capped,
with a longer run handed to the user if not converged; scoring 5
minutes. New data about 95 GB (renders 20, features 75).

## 7. Result

## 8. Decision
