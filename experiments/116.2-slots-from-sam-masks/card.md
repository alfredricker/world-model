---
id: "116.2"
title: slots from SAM 2's masks — the slot model of 116.1, also trained to match SAM 2's unlabelled automatic masks
rung: 2
serves: [P7]
status: done
verdict: uninterpretable
arch_version: 27
date: 2026-11-03
---

# 116.2: slots from SAM 2's masks

The perceptual half of event segmentation (Roadmap step 2: "objects from
a pretrained segmenter"), rung 2. Card [116.1](../116.1-slots-at-448/card.md)
stopped. Slots trained only to reconstruct the frozen DINOv2 patches
separated the large objects from each other but kept the table around
them (52–83% of an object slot's pixels were background), and no slot
took a button. Mean mBO (an object's best overlap with any slot) was
0.21, the same as k-means. SAM 2 finds the small objects that the slots
miss (cube 0.64, buttons 0.40 and 0.49 in 116.1's gate), but it cuts the
large ones into pieces, has no identity over time, and takes 0.74 s per
frame. WorldDP (`2606_08775`) joins the two: slots on DINOv2 patches,
trained with a mask loss against SAM 2's masks, separate this scene's
objects and table at 224 px.

Biological counterpart: early vision groups surfaces by their borders
(border-ownership cells in V2; Zhou, Friedman and von der Heydt 2000).
That grouping feeds object files, which keep an object's identity
through change (Kahneman, Treisman and Gibbs 1992). Here SAM 2 plays the
grouping and the slots play the object files.

## 1. Question

Trained with SAM 2's masks as a second target, without labels or actions,
does the slot model give each object (arm, cube, drawer, window, two
buttons) its own slot in each frame and keep it there over time? Rung 2,
P7 (discover the objects worth tracking).

## 2. What changes

One component, the slot model's training signal: a mask loss is added.
Encoder, slot attention, the transformer that carries slots forward, the
decoder and both of 116.1's losses stay as they were.

```
116.1:  patches → slots → decoder → reconstructed patches + masks
        loss = reconstruction + 0.5 × slot contrast over time
116.2:  the same, + 1.0 × mask loss: each SAM 2 segment of the frame is matched
        to one slot (Hungarian matching), and that slot's mask must cover it
```

- **The teacher's masks carry no labels.** WorldDP fixes in advance which
  slot is the drawer and which the cube. That is a class label, which C1
  rules out. Here SAM 2's automatic mask generator, with no prompts,
  proposes segments in each frame. They become a partition of the 32 × 32
  patch grid: each patch takes the segment covering most of its 2 × 2
  frame pixels. Segments of fewer than 2 patches are left unsupervised.
- **Matching per frame.** Segments are matched one to one with slots by
  the Hungarian algorithm (as DETR does, Carion et al. 2020), at the
  lowest total loss. Slots left unmatched get no mask target. Which slot
  holds an object is decided by the slots, not by the teacher.
- **Mask loss: WorldDP's.** A Tversky loss with α = 0.99 on missed patches
  and β = 0.01 on extra ones. Each segment is weighted by (1/size)^0.2 so
  small objects count, with weights normalised over the batch. Missed
  patches cost most, so each segment's slot must cover all of it. Because
  every patch belongs to some segment, covering the table's segment pushes
  object slots off the table.
- **Slot count.** K = 8 unless the gate finds more than 7 segments in 5%
  or more of frames. In that case K is raised to cover the 95th
  percentile, up to 16, and segments beyond K are left unsupervised,
  smallest first.
- **Training data.** SAM 2 labels whole 4-step training segments, as many
  as fit 10 minutes at the gate's speed, and 10% of them are kept back to
  watch the mask loss for overfitting. Each batch is half labelled
  segments, which get all three losses, and half unlabelled ones, which
  get reconstruction and contrast.
- **At test time** the slots run alone, without SAM 2.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A supplied segmenter's unprompted masks outline the objects | untested (voxel textures) | fails in "everything" mode (card 104) | holds (flat colours) | holds (SAM 2's domain) | perception adapter (C1 allows it); a world where it fails needs its own teacher |
| A fixed number of slots covers the scene | no (many creatures, blocks) | partly | partly | partly | declared setting, as in 116 |
| Objects persist between model steps | holds in view | holds | holds | holds | general |
| No class labels, no actions | holds | holds | holds | holds | general |

## 3. Dependencies

- Card 116.1: the 448-px features (`runs/116/feat448_*.npy`), the truth,
  the metrics, the scorer, the slot model and its settings (decoder width
  256, batch 32).
- SAM 2.1-small, installed and run in 116.1's gate.
- WorldDP's mask loss (`2606_08775`, in LITERATURE). Hungarian matching
  for unordered targets (DETR; LITERATURE's WorldDP row).

## 4. Data check

As 116.1: the cube is in view in 94% of frames, and each button covers
about 11 pixels of the 64 × 64 frame (about 3 patches). The gate counts
how often SAM 2 gives the cube and each button their own segment.

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **Teacher settings.** On 300 truth frames, score SAM 2's partition at
   32 × 32 as segments (FG-ARI, mBO per object) under at most three
   declared settings:
   - (a) 116.1's: 32 × 32 points, 512-px input, smaller masks on top;
   - (b) the same with larger masks on top;
   - (c) 16 × 16 points, if it is at least 3 times faster.

   The best mean mBO is kept. Truth chooses only among these declared
   adapter settings, as 116 chose the grid. The gate also reports
   segments per frame, which sets K.
2. **Bars.** The slots learn the teacher's outlines, so they cannot be
   expected to beat them by much. Each object's bar is the smaller of
   116.1's (0.7 × the 32 × 32 ceiling) and 0.9 × the teacher's score. If
   the teacher's arm is below 116.1's bar because SAM 2 cuts it into
   links, the arm is scored as the union of slots whose pixels are mostly
   arm. Joining the links into one body is card 117's job ("the self is
   what every action moves").
3. **Trivial baseline:** k-means at 32 × 32 (116.1's gate: mean mBO 0.21).
4. **Smoke run:** 2 minutes with labelled segments, to check that the
   mask loss falls and to measure steps per second at the chosen K.

Result of the gate (2026-11-03, `tools/card116/teacher.py`, `runs/116/teacher_gate.json`):
**stopped at step 1. On the 64 × 64 frames SAM 2 does not outline the
large objects, so the bars would fall below k-means.**

| 300 truth frames, at 32 × 32 | Mean mBO | Arm | Cube | Button 0 | Button 1 | Drawer | Window | Segments (mean, 95th pct) | s/frame |
|---|---|---|---|---|---|---|---|---|---|
| (a) 32 points, fine on top | 0.33 | 0.27 | 0.51 | 0.32 | 0.32 | 0.25 | 0.30 | 28, 35 | 0.75 |
| (b) 32 points, coarse on top | 0.23 | 0.28 | 0.40 | 0.05 | 0.06 | 0.29 | 0.28 | 21, 27 | 0.75 |
| (c) 16 points, fine on top | 0.30 | 0.26 | 0.51 | 0.28 | 0.27 | 0.22 | 0.27 | 24, 30 | 0.21 |
| k-means (116.1) | 0.21 | 0.55 | 0.03 | 0.02 | 0.02 | 0.28 | 0.35 | 8 | |

- **Too many segments.** On 60 frames, 19 of (a)'s 28 segments per
  frame lie mostly on the table and walls; the arm is in about 5, the
  drawer 1.3, the window 1.0. K would need 35; the card capped it at 16
  and dropped the smallest segments, which are the cube and buttons.
- **SAM 2 does not see the large objects whole.** Even its best single
  mask, before painting, overlaps the arm 0.21, the window 0.26 and the
  drawer 0.39. The bars would become arm 0.24, drawer 0.23 and window
  0.27, all below k-means.
- **Resolution, in part.** The same 60 frames re-rendered at 256 × 256
  from the stored joint positions (6 of 255 from the dataset's frames
  when read at 64 × 64), SAM 2's best single mask: buttons 0.85 and 0.85,
  cube 0.73, drawer 0.63, but still arm 0.20 and window 0.14. SAM 2 cuts
  the arm into links at any size, and sees through the window's glass.

## 6. Success criteria and prediction

On 116.1's held-out validation frames and clips:
1. **Each object has a slot:** FG-ARI ≥ 0.6, and for every object, mBO
   at or above its bar (step 2 of the gate).
2. **Better than clustering:** mean mBO at least 0.10 above k-means
   (0.31 or more).
3. **Kept over time:** on the clips, each object's best slot stays the
   same on at least 90% of the model steps where it is visible.

Meeting all three gives card 117 objects to measure change on, at a few
milliseconds per frame instead of SAM 2's 0.74 s, with identities SAM 2
does not supply.

**Prediction.**
- The cube and drawer pass. Covering the table's segment pulls the
  object slots off the table, which was 116.1's main failure.
- The buttons are the risk. At about 3 patches each, they need a slot of
  their own every frame, and SAM 2 itself scores only 0.40–0.49 on them.
- The arm takes 2–3 slots, as SAM 2 cuts it into links.
- Criterion 3 rises over 116.1's (cube 77%, button 1 75%), because each
  slot is now pinned to an outline.

**Budget:**
- Gate: about 10 minutes for the teacher settings, 10 for labelling, 2
  for the smoke run.
- Training: 30 minutes, capped. If it has not converged, a longer
  command is handed to the user.
- Scoring: 5 minutes.

## 7. Result

Not trained: the gate showed that the teacher, on 64 × 64 frames, would
set bars below k-means for the arm, drawer and window, so a pass would
show nothing. Verdict: **uninterpretable**.

## 8. Decision

**Revise** (the user, 2026-11-03): card
[116.3](../116.3-parts-at-256/card.md) renders the frames at 256 × 256,
where SAM 2 outlines the buttons, cube and drawer, and takes its slots
as parts, one per SAM 2 segment, since SAM 2 still gives about 31.
