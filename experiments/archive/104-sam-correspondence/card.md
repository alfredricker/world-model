---
id: "104"
title: SAM 2.1 for correspondence — creatures' identity from a pretrained promptable video segmenter
rung: 1
serves: [P7, P2]
status: done
verdict: stop
arch_version: 27
date: 2026-10-13
---

# 104: SAM 2.1 for correspondence

Approved by the user 2026-10-13 ("Yes", for the overnight session). Drafted at the
user's request ("let's use SAM 2.1 for
correspondence"), after the user amended GOAL C1. A declared, pretrained
perception front end with no language or class labels may now supply
objects' outlines and their correspondence across frames. Series
[103](../103-tracking-with-general-priors/card.md) is closed.

## 1. Question

Suppose SAM 2.1 supplies each creature's mask and carries it from frame to
frame, and the agent supplies only where to point it and which tracks to
keep. Does a creature then keep one identity on Crafter far better than the
best hand-built tracker (103.5, IDF1 0.623 on the learned view), toward the
true-state bound (0.934)? Rung 1. Serves P7 (units tracked across frames) and
P2 (belief about a changing world). Every later Crafter card (the effect
model, 098.4; short-term memory, 102.3) needs this correspondence.

## 2. What changes

One component: correspondence and objects' outlines, which move from 103.5's
sprite search to SAM 2.1. The view's motion (103.4) and earned existence
(103.5) stay.

```
103.5:     a file is a pixel template and mask, found each step by search near where it was;
           born from residual pieces that moved (the pair test); existence earned (IPDA); records by look
this card: a track is SAM 2.1's mask, carried each step by SAM's memory of recent frames
           born where the frame changed in the world's frame (after 103.4's view motion), outside the
             camera's pixels and existing masks: SAM pointed at the piece's centre, its top-scored mask taken
           confirmed once it has moved against the world (at birth: its content found in the last frame at
             a displacement whose source has since changed; or a later move); tentative ones die after 3 steps
           reported while SAM's occlusion score says present; ended after 20 steps absent
           memory older than SAM's window pruned, except the prompted frame
frames:    Crafter rendered at its native 16 px per tile for SAM; the 64 px frames stay for 103.4's placing
```

**Declared:**
- **SAM 2.1** (tiny checkpoint, frozen), under GOAL C1 as amended.
- **The 16 px render.** The adapter draws the view at the sprites' own
  resolution.
- **A mover's step is at most 1.5 times the agent's own step** (the pair
  test's window).
- **One threshold on change.** It is set relative to the frame's own noise
  level (the median absolute difference), and chosen on the development
  worker.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A pretrained segmenter's masks fit the world's objects | yes at 16 px (smoke test, IoU 0.81–0.86 for cows and zombies); no at 7 px | likely (flat colours; untested) | yes (its training domain) | declared (C1) |
| New objects show themselves by changing in the world's frame | movers yes; still creatures not until they move | yes | yes, given ego-motion | general |
| A mover's step is bounded | yes | yes | yes at the frame rate | declared, window 1.5 own steps |
| Things never seen moving need no track | scenery yes; a still cow is missed | yes | partly | general; belief keeps them as scenery (102.3) |
| The view at its native resolution | 16 px per tile | grid cells enlarged | camera resolution | adapter |

## 3. Dependencies

- **SAM 2.1** (Ravi et al. 2024, LITERATURE's current focus). Meta's
  repository at commit 2b90b9f, checkpoints `sam2.1_hiera_*.pt` in
  `runs/sam2/ckpt/`. Smoke tests in `tools/card104/smoke*.py`.
- **103.4's placing** (kept), not refitted: `placing_tau300.npz` from
  103.5, run on the new workers by `tools/card103.5/origins.py`.
- **103.1's camera pixels**, enlarged to 16 px.
- **103.5's files** (memory from workers 0–1, `mem_1034_w01.pkl`) as the
  baseline on the new play, and its IDF1 and extra measures.
- **098.5's collector** (own random stream per creature) for the new play.

## 4. Data check

Crafter is not reproducible across processes: it picks which creature to
despawn by iterating a Python set, whose order depends on memory addresses.
098.5's stored play therefore cannot be re-rendered at 16 px. New play is
collected at both resolutions in one process, by 098.5's collector on seeds
2010–2012:
- worker 10 for development;
- workers 11–12 for the test.

The 64 px play is saved beside 098.5's (`runs/098.5/play_w1x.npz`), so that
103.4's and 103.5's tools read it unchanged. 098.5's test workers had 55,655
creature frames (15% at night). The new workers' counts are checked before
the main run.

## 5. Feasibility gate

On worker 10:
- **Upper bound:** SAM's tracks prompted at each creature's first frame in
  view, using the evaluator's positions (oracle prompts, gate only). This is
  SAM's own correspondence, with discovery perfect.
- **Trivial baseline:** identity by place, 0.009 (103.5).
- **Speed:** frames per second, end to end.

The gate passes if oracle-prompted IDF1 is at least 0.85 and the speed is
at least 20 frames per second.

## 6. Success criteria and prediction

On workers 11–12, with 103.4's learned view motion:
1. **Identity.** IDF1 over cows, zombies, skeletons and arrows at least
   0.80, against 103.5 rerun on the same play (about 0.62 expected) and
   098.6's 0.717.
2. **Tracks on scenery.** At most 1% of reported detections on no creature
   (103.5: 4.3% on the learned view).
3. **Speed.** At least 20 frames per second end to end on one GPU, so that
   an agent's step stays under 50 ms.

**Prediction:**
- **Oracle-prompted IDF1 about 0.9.** In the smoke clip SAM stayed on a
  skeleton and an arrow in every frame they were in view (14 of 14, 7 of 7).
- **IDF1 0.75–0.85 with discovery.** The losses: creatures still since first
  seen (7% of frames in 103.5), thin, fast arrows, and change pieces from
  night noise.
- **Detections on no creature 1–3%,** from masks left on ground after a
  creature leaves or dies.
- **Speed 25–30 frames per second** (the tiny model propagated two objects at
  30 frames per second in the smoke test).

**Decision rules:**
- **Keep** if all three hold.
- **Revise** if IDF1 is at least 0.10 above 103.5's rerun but a criterion
  fails.
- **Stop** this form if IDF1 is less than 0.10 above 103.5's rerun.

**Budget** (each run capped, CHARTER rule 5):

| Step | Time |
|---|---|
| Collecting three workers | about 5 minutes |
| 103.4's placing on three workers | about 3 minutes each |
| 103.5's baseline on workers 11–12 | about 6 minutes each |
| The gate on worker 10 | about 15 minutes |
| The test | about 17 minutes per worker, the two in parallel |

The gate and the test are over 10 minutes, so they need this card approved.
No run exceeds 30 minutes.

## 7. Result

Run 2026-10-13 ([results.json](results.json)). The tracker
(`tools/card104/track.py`) runs SAM 2.1's own track step per object, online,
with its memory pruned to the last 15 frames. It was tuned in oracle mode on
worker 10's first 3,000 frames. The gate ran on frames 3,000–29,999, which
tuning never saw.

**Tuning** (oracle prompts, evaluator's origins, first frames of worker 10):

| Configuration | Frames | IDF1 | On no creature |
|---|---|---|---|
| A point at the tile's centre, prompted once | 1,500 | 0.623 | 11.5% |
| A point on the sprite, prompted once | 1,500 | 0.600 | 17.8% |
| + re-prompted when a track ends; a colour gate; duplicates end | 3,000 | 0.725 | 7.3% |
| Overlaps split by SAM's logit instead of duplicates ending | 3,000 | 0.688 | 6.6% |
| The same, with the small model | 3,000 | 0.696 | 2.5% |
| The same, with SAMURAI's memory selection | 3,000 | 0.679 | 8.6% |
| Box prompts (the tile), with or without the gate | 3,000 | 0.645 / 0.600 | 17–22% |
| Duplicates end, nearest-neighbour upscaling (rerun) | 3,000 | 0.735 | 7.1% |
| **Duplicates end, bicubic upscaling (SAM's own loader)** | 3,000 | **0.780** | 2.9% |

**The gate** (held-out frames, the last configuration):

| Measure | Value | Bar |
|---|---|---|
| Oracle-prompted IDF1 | **0.784** | ≥ 0.85 |
| ID recall / ID precision | 0.733 / 0.844 | |
| ID recall: cow, zombie, skeleton, arrow | 0.72, 0.83, 0.68, 0.72 | |
| Detections on no creature | 2.2% | (test: ≤ 1%) |
| Frames per second, end to end | **36.8** | ≥ 20 |
| GPU memory, peak | 0.74 GB | |

- **Gate: fail** on identity (0.784 against 0.85); speed passes. The test's
  bar (0.80) lies above this upper bound, so the test was not run.
- **The upper bound is generous.** When a creature's track ended, the
  oracle prompted it again under its true identity: 2,574 times, against
  1,117 first prompts. Prompted only once, SAM scored 0.60–0.62, no better
  than 103.5's rerun on workers 11–12 (0.623 and 0.625 with the learned
  view).
- **Where SAM loses a creature,** seen in renders of 3,000 frames: tracks
  older than 10 frames on a creature in view report nothing on 27% of
  frames (364 of 1,350). Three causes:
  - **Night:** the creature cannot be seen at all.
  - **Swaps:** the track moves to a like creature, such as a second cow two
    tiles away.
  - **Shrinking masks:** the mask covers only part of the sprite and
    shrinks frame by frame.

  The view's motion does not matter: SAM calls a creature in view absent
  on 12% of frames when the view moved and 10% when it did not.
- **Where SAM is good:** its masks, while it has them, lie on creatures.
  Detections on no creature were 2.2%, against 4.3% for 103.5. And it is
  fast.
- **Prediction:** oracle about 0.9, wrong (0.78). Speed 25–30 frames per
  second, beaten (37).

## 8. Decision

**Stop** (2026-10-13), SAM 2.1's propagation as Crafter's source of
correspondence. The gate failed, so the card cannot be passed as written.
The stop line was 0.10 above 103.5's rerun, about 0.72. Even the oracle's
upper bound clears it by only 0.06, and only by restoring each creature's
true identity every time SAM lost it. Discovery cannot restore identity.

What SAM lacks here is what tells one cow from another: where each one is
expected to be next. Its memory matches appearance, and Crafter's cows look
alike. SAM is worth keeping for outlines when prompted: masks on creatures,
2.2% elsewhere, 37 frames per second. Correspondence between like objects
is prediction's job. That fits the series' rule that perception may have
several parts but prediction has one: the transition model predicts where
each object goes, and association follows the prediction. That is a
question for series 105's objects card
([105.3](../105.3-objects-and-history/card.md)), whose input changes with
this decision.
