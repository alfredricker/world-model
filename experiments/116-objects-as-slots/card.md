---
id: "116"
title: objects as slots — slot attention on the frozen DINOv2 patches binds the scene's objects
rung: 2
serves: [P7]
status: done
verdict: uninterpretable
arch_version: 27
date: 2026-11-01
---

# 116: objects as slots

The perceptual half of event segmentation (Roadmap step 2, rung 2),
chosen after the segmentation review in LITERATURE's current focus
(2026-11-01). Card 115 showed that events need objects: boundaries
computed over whole patches sat mostly on the arm. Here objects are
learned, not supplied (GOAL C1: "what objects are ... is learned"), by
slot attention over the DINOv2 features card 114 already stored.
Biological counterpart: object files, persistent "this thing" records
that carry an object's identity through change and motion (Kahneman,
Treisman and Gibbs 1992).

## 1. Question

Trained only on `visual-scene-play-v0` frames, without labels or actions,
does a slot model on frozen DINOv2 patches give each object (the arm,
cube, drawer, window and two buttons) its own slot in each frame, and keep
it there over time? Rung 2, P7 (discover the objects worth tracking).

## 2. What changes

A slot model is added beside card 114's predictor. Nothing else changes.

```
card 114:  frame → DINOv2 (frozen) → 196 patch vectors → predictor
116:       196 patch vectors → MLP adapter → slot attention (K = 8 slots of 64 numbers,
           started from the previous model step's slots, carried by a 1-layer transformer)
           → MLP decoder → reconstructed patch vectors; each slot's share of each patch = its mask
```

SlotContrast's design (`2412_14295`, its MOVi-C settings unless stated):
- **Slots:** 8 slots of 64 numbers. The scene has 6 objects and a
  background, and the arm may split (SOLD saw 1–3 slots per arm). Slots
  start from a learned fixed initialisation, with 3 attention iterations
  on the first frame and 2 after.
- **Over time:** a 1-layer transformer (4 heads) carries the slots from
  one model step (5 environment steps) to the next.
- **Losses:**
  - squared error reconstructing the 196 frozen patch vectors, with an
    MLP decoder (DINOSAUR);
  - plus 0.5 × the contrastive loss between slots: each slot is pulled
    toward itself one step later and pushed from every other slot in the
    batch (temperature 0.1).
- **Training:** segments of 4 model steps from training episodes 0–899
  (the stored every-fifth-frame features); batch 64; Adam at 4 × 10⁻⁴
  with 2,500 warm-up steps; gradient clipping 0.05.
- **Two departures from the paper:**
  - 196 tokens per frame where it used 576: we keep the 196 × 196 input
    the predictor uses.
  - No actions: perception is kept action-free, and the arm's
    contingency is card 117's job.
- **Masks for scoring:** each slot's 14 × 14 mask, upsampled to 64 × 64;
  each pixel goes to its highest slot.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A fixed number of slots covers the scene | no (many creatures, blocks) | partly | partly | partly | declared setting (8); variable counts are later work |
| Objects persist between model steps (slots carried forward) | holds in view; lost when out of view (belief's job) | holds | holds | holds | general |
| Frozen self-supervised patch features separate objects | untested (voxel textures) | untested | holds (flat colours) | holds (DINOv2's domain) | adapter (C1 allows it) |
| No actions in perception | holds | holds | holds | holds | general |

## 3. Dependencies

- Card 114's stored DINOv2 features (`runs/114/feat_*.npy`) and its data
  split (episodes 0–899 train; validation and 900–999 held out).
- SlotContrast (`2412_14295`) and the evidence that slot quality tracks
  planning on OGBench Cube (`2608_12078`). Both are in LITERATURE.
- MuJoCo segmentation renders for scoring (card 115's gate checked their
  alignment: 0.057 of 255).
- SAM 2.1 (small) for the reference arm; installed, its checkpoint to be
  downloaded (about 180 MB). C1 allows it.

## 4. Data check

Patch coverage per frame, from card 115's masks (out of 196): arm 48,
window 19, drawer 17, cube 3, buttons 2–3. The cube and buttons span 2–3
patches of 14 × 14. The gate measures what the patch grid can show at
all. It also counts how often each object is visible, and how often the
cube is inside the gripper.

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **Truth.** Pixel-level segmentation renders for 2,000 random
   validation frames, and 100 validation clips of 21 model steps. The six
   objects are scored as foreground (all arm links merged into one); the
   rest is background.
2. **Ceiling.** Give each patch its true majority object and upsample as
   the slot masks are. This is the best any 14 × 14 method can score. It
   gives FG-ARI (agreement on foreground pixels, chance 0) and per-object
   mBO (the best slot's overlap with the object). The bars below are set
   from it.
3. **Baselines:**
   - k-means with 8 clusters on the raw DINOv2 patch vectors (fitted on
     training frames), each cluster taken as a segment;
   - SAM 2.1-small's automatic masks, the supplied reference. Reported,
     not judged.
4. **Smoke run:** 2 minutes, to measure steps per second and check that
   the loss falls.

Result of the gate (2026-11-01, code in `tools/card116/`):
1. **Truth:** rendered. When visible, the cube covers 14 pixels of the
   64 × 64 frame and each button 11 (about 3 × 4 pixels); the cube is in
   view in 94% of frames.
2. **Ceiling of the 14 × 14 grid: too low.**

   | | FG-ARI | Arm | Cube | Button 0 | Button 1 | Drawer | Window |
   |---|---|---|---|---|---|---|---|
   | Ceiling, 14 × 14 | 0.68 | 0.77 | 0.15 | 0.004 | 0.003 | 0.61 | 0.76 |
   | k-means on DINOv2 | 0.68 | 0.64 | 0.02 | 0.02 | 0.02 | 0.20 | 0.33 |

   - No patch is mostly button, so no method on this grid can bind a
     button.
   - k-means already reaches the ceiling's FG-ARI, so criterion 2's
     FG-ARI margin cannot be met. FG-ARI is dominated by the arm, drawer
     and window, which k-means separates.
   - Ceilings at finer grids (600 truth frames):

     | Grid (input) | FG-ARI | Cube | Button 0 | Button 1 |
     |---|---|---|---|---|
     | 24 × 24 (336 px) | 0.81 | 0.40 | 0.32 | 0.15 |
     | 32 × 32 (448 px) | 0.81 | 0.53 | 0.57 | 0.57 |
     | 37 × 37 (518 px) | 0.80 | 0.49 | 0.53 | 0.40 |

     At 448 pixels each patch is exactly 2 × 2 frame pixels; 518 is
     worse, because its patches do not line up with the frame's pixels.
3. Not run: SAM 2 and the slot model. The ceiling decided the card.

## 6. Success criteria and prediction

On held-out validation frames and clips:
1. **Each object has a slot:** FG-ARI ≥ 0.6, and for every object, mBO
   at least 0.7 × its ceiling.
2. **Better than clustering:** FG-ARI and mean mBO each at least 0.10
   above k-means.
3. **Kept over time:** on the clips, each object's best slot stays the
   same on at least 90% of the model steps it is visible.

Meeting all three shows that objects can be learned, with no labels, from
the features the predictor already uses. Card 117 can then measure change
per object.

**Prediction.**
- Criteria 2 and 3 pass for the arm, drawer and window.
- Criterion 1 is the risk, for the cube and the buttons. At 2–3 patches
  their ceiling may be low, and the held cube may join the gripper's
  slot (WorldDP needed a special loss to stop this).
- The arm will take 2 slots.
- **If small objects fail:**
  - with their ceiling well above what the slots reach, the slot model
    is revised;
  - with the ceiling itself low, card 116.1 re-encodes at 336 × 336
    (24 × 24 patches, the paper's setting; about 80 GB and 10 minutes).

**Budget:**
- Truth renders: 2 minutes.
- Ceiling, baselines and SAM 2: under 10 minutes.
- Training: 30 minutes at most, capped. If it hasn't converged, the
  paper's 100k steps are handed to the user as a command (rule 5).
- Scoring: 5 minutes.

## 7. Result

Not run: the gate showed that the 14 × 14 grid cannot show the buttons
(ceiling 0.004) and barely the cube (0.15), and that criterion 2 cannot
be met.

Verdict: **uninterpretable** (the test as set could not be passed).

## 8. Decision

**Revise**, as section 6 named in advance: card 116.1 re-encodes the
frames, at 448 × 448 rather than the 336 named there, because the gate
measured 336 as too coarse for button 1 (0.15). Criterion 2 moves to mean
mBO, since FG-ARI cannot separate a slot model from k-means in this
scene.
