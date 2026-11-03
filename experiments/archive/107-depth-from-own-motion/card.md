---
id: "107"
title: depth from the agent's own motion — a pretrained depth backbone tuned by view synthesis with the known pose
rung: 1
serves: [P7, P2]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 107: depth from the agent's own motion

Approved by the user 2026-10-13 ("alright let's do it"). That was the reply
to a proposal that gave the training's length and the pass bar. The first
card in Craftium, a 3D voxel world built on Luanti (formerly Minetest)
running VoxeLibre, a Minecraft clone. The user chose it as the step between
Crafter and robotics (GOAL: "Mgrid -> Crafter -> Minecraft"). CHARTER's
"Current world" still names Crafter; this card is judged in Craftium by
the user's go-ahead.

## 1. Question

**What the smoke tests found** (2026-10-13, no card):
- **Pretrained trackers lose identity once a creature leaves view.**
  Eight identical cows were each given one perfect prompt.
  - CoTracker3 kept its cow 78% of the time while the cow stayed in view,
    but only 13% after it had left the view and come back.
  - SAM 2.1: 60% and 10%.
- **A belief fixes this, given good positions.** With perfect detections,
  exact depth and the pose, a belief with predicted spread brought back
  the cow's own track on every return within 100 steps (60 of 60). CHARTER
  now lets belief decide "which one".
- **Off-the-shelf depth is not good enough.** Depth Anything 3 (DA3) was
  scored against the game's own voxels.
  - The shape is right: with each frame's scale fixed, error is 12–18%.
  - The scale drifts: the single-frame metric model reads 0.75 ± 0.17 of
    true from frame to frame.
  - Creatures are 26–49% off.
  - Belief identity with this depth: 0.47–0.60, no better than no memory
    (0.49).

**The question.** Suppose the depth model is tuned on the agent's own
frames, using only its known pose, with no labels. Does depth become good
enough for the belief to keep cows' identity? Specifically, are cows that
come back within 100 steps given their own track at least 85% of the time,
against 91–94% with exact depth? Rung 1. Serves P7 (units tracked across
frames) and P2 (belief about things out of view).

## 2. What changes

One component: the perception front end's depth.

```
smoke test:  depth = DA3METRIC-LARGE(frame), as released                  scale 0.75 ± 0.17, creatures 49% off
this card:   the same model; its depth head (the DPT head over the frozen DINO-style backbone) tuned so that
             each frame, moved by its predicted depth and the known pose change, reproduces the frames a few
             steps before and after it (Monodepth2's photometric loss with the pose given, not estimated)
```

**The loss** (Godard et al. 2019, with the pose known):
- **Target and sources:** a target frame t, and sources t − k and t + k.
  Each k is the nearest step (up to 12) at which the camera has moved at
  least 0.25 blocks.
- **Warping:** each target pixel is back-projected with its predicted
  depth, moved by the known pose change, and the source image is sampled
  there.
- **Photometric error:** 0.85 · (1 − SSIM)/2 + 0.15 · L1, taking the
  smaller of the two sources per pixel.
- **Automatic mask:** a pixel counts only where the warped source matches
  better than the unwarped one. This drops whatever moves with the
  camera (the hand, the item bar) and moving creatures, without
  declaring them.
- **Smoothness:** edge-aware smoothness on mean-normalised inverse depth,
  weight 10⁻³.
- **Scale:** the pose is in blocks, so depth is learned in blocks.

**Why the single-frame model, and only its head:**
- **The pose is needed only in training.** At use, depth comes from one
  frame, so it holds when the agent only turns. In the smoke test, 79 of
  1,000 frames had no baseline for the multi-view model.
- **Tuning only the head keeps the backbone's features as they are.** The
  token encoder will later read those same features (GOAL C2).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The agent's pose is known | yes (2D; no depth needed) | no depth needed | odometry | declared (C1) |
| The camera's intrinsics are known | — | — | calibration | declared in Craftium's adapter (field of view 90° vertical) |
| Nearby frames show a mostly still world | — | — | mostly | general: moving pixels drop out by the automatic mask |
| A pretrained geometry backbone | — | — | yes | declared front end; GOAL C1 names outlines and correspondence, not depth (open with the user) |
| The frame is drawn from the pose of the step before | — | — | sensor timing | adapter (Craftium) |

## 3. Dependencies

- **Depth Anything 3** (Lin et al. 2025, ICLR 2026). DA3METRIC-LARGE,
  Apache-2.0, at commit 3d835ec. Metric depth is focal × output / 300.
- **Monodepth2** (Godard et al. 2019; Zhou et al. 2017) for the loss and
  the automatic mask.
- **Craftium** (Malagón et al. 2025, ICML) v0.0.1.
- **Scoring, from the smoke tests, checked by eye:**
  - creature positions from the probe mod's log, projected by the pose;
  - true depth by marching each pixel's ray through the voxel
    observation;
  - the belief check, whose exact-depth result is the reference.
- **Code:** `tools/card107/` (`setup.sh` builds the environments under
  `runs/107/envs`).

## 4. Data check

- **Training:** six roaming clips of 1,200 steps at six places 300–800
  blocks from the test pasture. The agent walks about half the time.
  Frames with a source at least 0.25 blocks away are counted before
  training.
- **Test, never trained on:**
  - the pasture clip (1,000 steps, eight cows, seed 11, voxels) at
    (120, 92);
  - a roaming clip with voxels at (900, 900).

## 5. Feasibility gate

- **Upper bound:**
  - the released model with each frame's scale fixed: 14%, so the shape is
    there;
  - exact depth in the belief: identity score 0.87, returns 91–94%.
- **Trivial baseline:** the released model.
  - terrain: 34% error, scale 0.75 ± 0.17;
  - creatures: 49% off;
  - belief: 0.47–0.55, returns 37–44%.
- **Gate:** the loss must prefer the truth. On the pasture clip's 100
  scored frames, the photometric error at true depth must be below:
  - true depth × 0.8;
  - true depth × 1.25;
  - the released model's depth.

  If not, the loss cannot pull the scale toward the truth, and the main
  run is not made.

## 6. Success criteria and prediction

1. **Terrain**, on both test clips: mean |error| / depth at most 15%, and
   per-frame scale spread (standard deviation) at most 0.05. The released
   model: 34%, ±0.17.
2. **Creatures**, on the pasture clip: mean |error| at most 10%, against
   each creature's near surface (centre depth − 0.4 blocks). The released
   model: 49% against the centre.
3. **Belief**, on the pasture clip:
   - cow returns within 100 steps that get their own track back: at least
     85% (exact depth 91–94%, released 37–44%);
   - identity score: at least 0.80 (exact 0.87).

**Prediction:**
- **Criterion 1 passes.** Blocks are one unit everywhere, so a tuned head
  can learn a stable scale.
- **Criterion 2 is close.** Cows are small, and their depth blurs into the
  ground behind them.
- **Criterion 3 follows criterion 2.**

**Decision rules:**
- **Keep** if 1–3 pass.
- **Revise** if 1 passes and 2 or 3 fails.
- **Stop** if the gate fails, or if the scale spread is not at least
  halved.

**Budget:**
- collection about 5 minutes;
- gate about 2 minutes;
- training one capped run of at most 20 minutes;
- scoring about 5 minutes.

## 7. Result

Run 2026-10-13 ([results.json](results.json); `tools/card107/`). Every run
was capped; training took 6 minutes.

**Data.**
- **Training:** 5,196 target frames with a source, from the six roaming
  clips.
  - Two clips ended early when the agent died. At (−500, −500) it ran off
    a 31-block drop at step 101; the roaming test clip ended at step 509.
  - Craftium's default physics multiplier (×20) lets a held "forward"
    reach 1.3–1.6 blocks per step. A real robot doesn't move like that,
    but neighbouring frames still overlap.
- **Test clips:** true depth covers 65% of pixels on the pasture and 21%
  on the roaming clip. On the roaming clip, tall grass (left out),
  sky, and hills beyond the voxel grid take up most of the view.

**Gate: pass.** The point-sampled photometric error is lowest at true
depth, on both test clips:

| | True | True × 0.8 | True × 1.25 | Released model |
|---|---|---|---|---|
| Pasture (30 frames) | **0.028** | 0.044 | 0.038 | 0.053 |
| Roaming (16 frames) | **0.022** | 0.032 | 0.025 | 0.053 |

**Main run:**

| | Released | Tuned | Bar |
|---|---|---|---|
| 1. Terrain, pasture: mean error / within 25% | 32% / 30% | **17% / 74%** | ≤ 15% |
| 1. Terrain, pasture: scale against true (spread) | 0.74 (±0.16) | **0.99 (±0.10)** | spread ≤ 0.05 |
| 1. Terrain, roaming: mean error / within 25% | 54% / 8% | 44% / 52% | ≤ 15% |
| 1. Terrain, roaming: scale (spread) | 0.45 (±0.36) | 1.02 (±0.70) | spread ≤ 0.05 |
| Terrain, each frame's scale fixed (shape alone), pasture | 13% | 15% | |
| 2. Cows, against the near surface: median / mean size of error | −34% / 37% | −31% / 50% | ≤ 10% |
| 3. Belief: cow returns within 100 steps that get their own track back (exact depth: 100%, 28) | 43% | 50% | ≥ 85% |
| 3. Belief: identity score (exact 0.95) | 0.54 | 0.55 | ≥ 0.80 |

1. **Terrain: fail, but much better.**
   - On the pasture the scale is now right (0.99 against 0.74), and the
     share within 25% rose from 30% to 74%.
   - The roaming clip's scale spread is dominated by frames with few
     scorable pixels.
   - The shape alone did not improve (13% → 15%): tuning fixed the scale,
     not the geometry.
2. **Cows: fail, unchanged.**
   - Neither depth map shows the cows. The head predicts at the
     backbone's patch scale (14 pixels at 504 wide), and a cow 10–20
     pixels across vanishes into the ground around it.
   - Cows stand near the horizon, where ground depth changes fast with
     height in the image, so the blur makes them read 31–34% too near.
   - Tuning cannot fix this: the automatic mask drops moving creatures
     from the loss.
3. **Belief: fail.**
   - With either depth, identity is at the no-memory level.
   - An exploratory variant read each cow's position from the ground depth
     just below its feet. It did no better: 39–46% of returns, positions
     off by 40–49% of their distance. The feet are at the horizon's blur
     too.

## 8. Decision

**Stop**, by the card's rule (2026-10-13): the scale spread was not
halved. On the pasture it fell from 0.163 to 0.098; on the roaming clip it
rose from 0.362 to 0.699. The user's word after the run: "it doesn't need
to be perfect though".

**What stops:** tuning a single-frame depth head by view synthesis, as
the source of creatures' positions for the belief. Two separate failures
block it:
- the backbone's depth cannot see a cow-sized object at this resolution;
- the loss masks out exactly the moving things the belief tracks.

**What stays available:** the tuned head as Craftium's terrain depth. Its
scale is right and its errors are half the released model's.

**What would test the next question:**
- **Resolution:** the same model at twice the resolution (1,008 pixels
  wide), no training, a few minutes. Can it see cows at all?
- **Matching in the learned belief:** use the pose's exact direction to
  each token and only a coarse, uncertain distance along it. How good
  depth must be depends on this matching. The smoke test's image-space
  matchers lost even with exact depth, so that question is still open.
