---
id: "115"
title: boundaries from surprise — System 1's unpredicted changes mark when conditions change
rung: 2
serves: [P7, P4]
status: done
verdict: uninterpretable
arch_version: 27
date: 2026-11-01
---

# 115: boundaries from surprise

The first card on CHARTER Roadmap step 2, event segmentation (the user,
2026-11-01). Conditions are the stable stretches between event
boundaries. Event segmentation theory (Zacks et al. 2007) says the mind
tracks features and perceives a boundary when a salient one changes
unpredictably. This card tests the temporal half with parts we already
have: card 114's predictor (kept by 114.1) supplies the prediction, and
its failures supply the boundaries. Objects (SAM 2) and the conditions
themselves come in the next cards.

Rung 2 of the ladder (inserted by the user, 2026-11-01): segment events.
This card is its first test, the boundaries; the conditions between them
come next.

## 1. Question

Does card 114's predictor, run on real frames, show its largest errors
where the scene's conditions change? That is, when a lock flips, the
drawer or window starts or stops sliding, or the cube starts or stops
moving, and not when the arm merely moves as commanded. Serves P7
(discover the events worth tracking) and P4 (keep the distinctions that
change outcomes).

## 2. What changes

Nothing is trained. A boundary detector is added on top of the trained
predictor.

```
card 114:  3 real frames + actions → predicted next frame's 196 patch vectors
115:       surprise(t, patch) = squared error of that prediction against the real frame,
           over that patch's usual error → frame score → peaks above a threshold = boundaries
```

- **Surprise.** For every held-out frame t, the predictor takes the real
  frames t − 10, t − 5 and t with their actions and predicts t + 5. A
  patch's surprise is its mean squared error over the 384 numbers. It is
  divided by that patch's usual error: its median and spread over
  training episodes 900–999. This is surprise against expected noise
  (Yu and Dayan 2005, LITERATURE).
- **Frame score:** the mean of the 4 most surprising patches, since an
  object covers a few patches. It is dated t + 5, the end of the
  interval in which the change happened.
- **Boundaries:** local peaks of the frame score (highest within ±5
  steps) above a threshold. The threshold is chosen on held-out training
  episodes 900–999 to maximise F1. Validation episodes are the test.
  Chosen this way, the true events set one number; the score at a
  label-free threshold (the top 3% of frames) is also reported.
- **Matching:** each true event is matched to at most one boundary within
  ±5 steps (one model step), and each boundary to at most one event.
- **Localisation:** the patch with the highest surprise at a matched
  boundary, compared with MuJoCo's segmentation render of the changing
  object.
- **True events** (scoring only, from the simulator's state):
  - lock flips;
  - starts and stops of the drawer, the window and the cube (speed above
    2 mm per step for at least 3 steps; gaps under 3 steps merged).
  - The arm's own motion, including the gripper closing, is not an
    event. It is predicted from the commands, and boundaries near it are
    reported separately.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A forward model's failures mark events | holds | holds | holds (frame changes) | holds | general (the theory's claim) |
| Each patch has a usual error (a fixed camera) | no: the view moves; per tracked object instead | no: the view follows the agent | yes | fixed cameras yes, wrist cameras no | declared; per-object normalisation comes with objects (next card) |
| Changes the agent causes but did not predict are events, as are the world's own changes | holds (mobs, day) | holds | holds | holds | general |
| Boundaries are resolved to one model step (5 environment steps) | a setting | a setting | a setting | a setting | declared setting |

## 3. Dependencies

- Card 114's predictor and frozen DINOv2 features (kept by card 114.1),
  and the held-out features already encoded (`runs/114/feat_val.npy`,
  `feat_tail.npy`, every frame).
- OGBench's simulator, to render segmentation masks for scoring
  (rendering checked 2026-10-31).
- Event segmentation theory (Zacks et al. 2007). The precedent for
  surprise from a representation-space predictor is Garrido et al. 2025
  (`2502_11831`). Both are in LITERATURE's current focus.

## 4. Data check

True events in the held-out episodes, counted from the simulator's state
(2026-11-01):

| Event | Validation (test) | Episodes 900–999 (threshold) |
|---|---|---|
| Lock flips | 691 | 718 |
| Drawer starts / stops | 314 / 314 | 322 / 322 |
| Window starts / stops | 371 / 371 | 373 / 373 |
| Cube starts / stops | 599 / 599 | 609 / 609 |
| Gripper closes (arm, not an event) | 2,103 | 2,161 |

- That is about 3,260 events per 100,100 frames, one every 31 steps.
- A drawer moves for a median of 9 steps and a window for 11, so a
  slide's start and stop are less than two model steps apart. One-to-one
  matching makes a detector find both.
- The cube moves for a median of 42 steps, mostly while carried.

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **Masks line up.** MuJoCo renders each frame's RGB and segmentation
   from the stored joint positions and button states. On 1,000 held-out
   frames, the RGB render differs from the dataset's frame by under 2 of
   255 on average, so the masks describe the frames the model saw.
2. **Upper bound: change with the arm masked out.** The frame-to-frame
   change of the patch vectors (no prediction), with patches on the arm
   removed by the true mask. It has the same normalisation, frame score,
   peaks and threshold rule. If it cannot reach the bar below (F1 0.6),
   the bar is unreachable from these features, and the bar is revised
   before the main run.
3. **Baselines:**
   - **Change without prediction:** the same detector on frame-to-frame
     change, with no mask. It sees the arm's motion as change.
   - **Random boundaries** at the rate the surprise detector chooses.

Result of the gate (2026-11-01, code in `tools/card115/`):
1. **Masks line up: pass.** Rendered frames differ from the dataset's by
   0.057 of 255 on average (largest 0.106), on 2,002 frames. On average
   the arm covers 48 of the 196 patches, the cube 3, a button 2–3, the
   drawer 17 and the window 19.
2. **Upper bound: fails the bar.** Change with the arm masked out
   reaches F1 0.43 (precision 0.50, recall 0.38), against the bar of 0.6.
   By this section's rule, the bar had to be revised before the main
   run. The scorer computed every detector in one pass, so the main
   numbers below exist, but they are not judged against the bar.
3. **Baselines.** Change without prediction: F1 0.41. Random boundaries
   (7,552, as many as the surprise detector chose): F1 0.33. With an
   event every 31 steps and ±5 steps of tolerance, a third of random
   frames fall near some event, so timing alone barely separates the
   detectors.

## 6. Success criteria and prediction

On validation episodes, threshold from episodes 900–999:
1. **Boundaries:** F1 ≥ 0.6 against the true events.
2. **Prediction matters:** F1 at least 0.15 above change without
   prediction (each at its own best threshold). This is the theory's
   claim: an unpredicted change, not any change, marks an event.
3. **Localisation:** at matched boundaries, the most surprising patch
   lies on the changing object, or one patch from it, on at least 80%.
   For a lock flip, the object is the button or its handle; both change
   colour.

Meeting all three shows that System 1's failures, measured against its
usual error, find the moments and places where conditions change, with
no labels. These are the boundaries the conditions card needs.

**Prediction.**
- Criterion 2 passes: the arm's motion dominates plain change and is
  predicted from the commands.
- Criterion 3 passes for slides and buttons. The cube is often hidden by
  the gripper.
- Criterion 1 is the risk, for a reason that runs through the theory: a
  good predictor is not surprised by what it predicts. Card 114's model
  forecasts presses at 95% and pulls at 96–100% from a grasp. So well-
  predicted changes may pass unflagged: lock flips, and slides stopping
  at the end of their travel. Starts after an uncertain contact should
  be flagged. Expected F1 0.45–0.65, recall lowest on stops and lock
  flips.
- **If criterion 1 fails while 2 and 3 pass,** the next card adds the
  other half of the theory: a boundary also where the predicted change
  is large (a predicted event is still an event). That reads System 1's
  forecast of change, not only its error.

**Budget:** surprise on 200,200 frames, about 2 minutes; segmentation
renders, about 3 minutes; scoring, 1 minute. No training.

## 7. Result

Validation episodes, threshold from episodes 900–999 (3,257 events):

| Detector | F1 | Precision | Recall | Boundaries |
|---|---|---|---|---|
| Surprise (this card) | 0.40 | 0.29 | 0.66 | 7,552 |
| Change without prediction | 0.41 | 0.41 | 0.42 | 3,388 |
| Change, arm masked (bound) | 0.43 | 0.50 | 0.38 | 2,448 |
| Random | 0.33 | | | 7,552 |

- **The criteria, for the record** (not judged, since the gate failed):
  1. F1 0.40, against 0.6.
  2. 0.02 below plain change, against +0.15.
  3. Localised 69%, against 80%.
- **Where surprise does well:**
  - Lock flips: recall 83%, the top patch on the button or handle 85%.
    The prediction that well-predicted flips would go unflagged was
    wrong.
  - Window: recall 68–73%, localised 88–90%.
  - Cube starts: 59% recall, against 2% for plain change. The predictor
    discounts some of the arm's motion, but not enough to raise
    precision.
- **Where it fails:** precision. 30% of unmatched boundaries are within
  5 steps of the gripper closing or opening; most of the rest sit on the
  arm.
- **After the run (diagnostics, `tools/card115/diagnose.py`):** matching
  in time and place, where a boundary counts only if its top patch is on
  the changing object:

  | Detector | F1, time only | F1, time and place |
  |---|---|---|
  | Random (random frame and patch) | 0.33 | 0.05 |
  | Surprise | 0.40 | 0.30 |
  | Surprise, arm masked | 0.40 | 0.30 |
  | Change | 0.41 | 0.31 |
  | Change, arm masked | 0.43 | 0.40 |

  - In time and place, surprise is far above chance, so it does find
    condition changes.
  - But it is no better than plain change, even with the arm masked.
    This predictor's errors do not single out what it failed to
    predict.
  - Cube events are found 6–15% of the time by every detector. The cube
    covers 3 patches and is held inside the gripper.

Verdict: **uninterpretable.** The upper bound did not reach the bar, so
the test as set could not be passed. The diagnostic matters more: at
this resolution, surprise from card 114's predictor marks events no
better than plain change.

## 8. Decision

**Revise.** Two changes, for the user to choose:
- **The test:** match in time and place (random 0.05), with the bar set
  from an upper bound measured first.
- **The signal: discount the agent's own body.** Most false boundaries
  are on the arm, the one thing the agent moves. The brain attenuates
  the sensations it causes itself, by predicting them from a copy of the
  motor command (Blakemore, Wolpert and Frith 1998: a self-produced
  tickle is felt less). The self can be learned as the region whose
  change the actions predict. Events are then sought in the rest of the
  scene, scored per object.
- Both need objects, Roadmap step 2's perceptual half. So the next card
  is the perceptual segmenter: SAM 2 masks on 64 × 64 frames, scored
  against MuJoCo's masks, with common fate as the fallback. Card 115.1
  then measures surprise per object, the self discounted, matched in
  time and place.
