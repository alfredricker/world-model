---
id: "114"
title: lock effects from pixels — a latent transition model on frozen patch features predicts what pulls and presses do
rung: 1
serves: [P6, P4]
status: done
verdict: uninterpretable
arch_version: 27
date: 2026-10-31
---

# 114: lock effects from pixels

The first card on the learned stack (card
[113](../113-post-mortem-learned-stack/card.md), signed 2026-10-31).
Roadmap step 1, System 1. Drafted at the end of the thinking session; it
runs only once the user approves it.

## 1. Question

Suppose a transition model is trained on frozen DINOv2 patch features of
OGBench `visual-scene-play-v0`, from frames and actions only. Given a real
frame and the actions that follow, does it predict:
- that pulling the drawer or the window opens it when it is unlocked and
  leaves it shut when it is locked;
- that pressing a button changes that lock?

This is rung 1 (P6: different actions get different, correct predictions
of the effect, at the horizon where it shows). It also serves P4: the lock
is a few red or white pixels that change the outcome, and the model must
keep them.

## 2. What changes

Version 27 is the MiniGrid agent; nothing of it is used. This card builds
the first part of the new stack, and only that part.

```
before (Crafter, series 105):  frame → hand-built tokens → transformer → next tokens
now:   64×64 frame → upsample to 196×196 → DINOv2 ViT-S/14, frozen
       → 196 patch vectors (384 numbers each)
       → predictor (DINO-WM's ViT, about 19M parameters) with the next 5 actions
       → the 196 patch vectors 5 steps later
```

- **Encoder:** DINOv2 ViT-S/14, frozen, at DINO-WM's 196 × 196 input. The
  frames are 64 × 64, so they are upsampled about three times.
- **Predictor:** DINO-WM's design.
  - A ViT of depth 6 over the patches of the last H ≤ 3 model steps, with
    a causal mask over steps.
  - A model step is 5 environment steps (DINO-WM's frameskip). The 5
    actions are joined into one 25-number vector, embedded by an MLP and
    appended to every patch.
  - Trained by teacher-forced squared error on the next step's patches.
- **Rollout for scoring:** open loop from a real frame, given the real
  actions, for k = 1–4 model steps (5–20 environment steps). The effect
  is scored at the step where it shows (C6). k is a measuring device, not
  the target.
- **Read-out, for scoring only:** probes trained on real frames' patches
  to the evaluator's state (drawer and window positions, both button
  states, the cube's position), then applied to predicted patches. A probe
  is a measuring device here, not the capability (C4); behavioural use
  comes with skills (rung 2).
- **Not given:** the arm's own state. GOAL C1 allows it, and OGBench
  stores it, but OGBench's published agents use the frame alone. Leaving
  it out keeps their numbers comparable. The arm is drawn see-through, so
  the frame shows it.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A frozen pretrained image encoder gives the state as patch vectors | holds (any RGB frame) | holds | holds (screen as image) | holds | perception adapter (rule 9 allows it) |
| A fixed third-person camera | no (first-person) | no (the view follows the agent) | yes | often | adapter; the predictor reads only patch vectors |
| A fixed number of patches per frame | yes, after resizing | yes | yes | yes | general |
| Actions as a continuous vector, embedded | discrete there: one-hot | discrete: one-hot | discrete or clicks | yes | general (one embedding per action form) |
| 5 environment steps per model step | a setting | a setting | a setting | a setting | declared setting |
| The world changes only through the agent | no (mobs, day) | no | yes in most games | no | not assumed: every patch is predicted |
| A goal | | | | | not used in this card |

## 3. Dependencies

- **DINOv2** (Oquab et al. 2024), self-supervised with no labels;
  allowed by GOAL C1 as amended on 2026-10-31. Established method.
- **DINO-WM's predictor** (`dino-wm`). An established method on six
  environments, cited in LITERATURE (rule 7(b)), though never on OGBench
  or at 64 × 64.
- **OGBench `visual-scene-play-v0`** and its `info` fields (`ogbench`),
  read for scoring only.

## 4. Data check

From the code, before any download:
- The drawer oracle never checks the lock, so failed pulls are in the
  data. By the play policy's task weights, about a quarter of drawer
  attempts meet a locked drawer.
- Buttons are chosen as often as the cube; each press flips one lock.

The gate counts both, by the evaluator's state on the 100 validation
episodes:
- **Pulls:** the gripper within 3 cm of the handle and the commanded
  motion along the slide for at least 5 steps. These are defined by what
  the input shows, never by whether the drawer moved (LESSONS: split by
  input, not by outcome).
- **Presses:** `button_states` flips.

Each kind (drawer and window pulls, locked and unlocked; presses of each
button) needs at least 100 held-out events. If the validation episodes
fall short, 100 training episodes are held out as well.

## 5. Feasibility gate

Run before the main run, each step under 10 minutes and capped (rule 5):
1. **The data check above.**
2. **Can the encoder see the lock?** Linear probes on real frames'
   patches, trained on training episodes and tested on validation ones.
   - Button states: at least 0.98 accuracy, against the majority class
     (counted in step 1).
   - Drawer and window positions: R² at least 0.95.
   - If the lock is not readable from frozen patches, the encoder fails.
     The card stops, and LeWorldModel with patch tokens is the next card
     (LITERATURE).
3. **Upper bound:** the same predictor on OGBench's state observations
   (`scene-play-v0`), scored on the same criteria. It shows the test can
   be passed.
4. **Trivial baselines:**
   - **"No change":** predicts every locked pull right and every unlocked
     pull and every press wrong.
   - **"Lock-blind":** the average outcome of a pull, regardless of the
     lock, so it opens every pull. It gets every unlocked pull right and
     every locked one wrong.
   - Each scores about 0.5 when the two lock states are weighted equally,
     so the bar below cannot be met by either.

Result of the gate (2026-10-31, code in `tools/card114/`):
1. **Data check: passes, with training episodes 900–999 held out.** The
   validation episodes alone had 57 locked drawer pulls, short of 100, so
   training episodes 900–999 were held out as well, as planned. The model
   trains on episodes 0–899. Held-out counts (validation + 900–999):
   - Pulls: drawer locked 138, unlocked 555; window locked 195, unlocked
     651. No locked pull moved 4 cm; 91% of unlocked drawer pulls and 100%
     of unlocked window pulls did (the rest stop short).
   - Presses: 658 of button 0, 688 of button 1.
   - Locked pulls are 17% of drawer attempts, not the quarter estimated
     from the code.
   - Two details fixed by the data before any model was trained:
     - The oracle alternates between opening and shutting, so a pull is a
       command along the slide in either direction, with at least 4 cm of
       room that way.
     - The gripper must be closed (opening > 0.5) as well as within 3 cm
       of the handle.
   - The state upper bound has its own held-out events on `scene-play-v0`:
     drawer locked 107, window locked 159.
2. **The encoder sees the lock: passes.** Linear probes on all 196 patch
   vectors, trained on episodes 0–899 and tested on every fifth validation
   frame:
   - Button states: 1.000 and 1.000 accuracy, against majority classes of
     0.550 and 0.576.
   - Drawer position R² 0.977 (error 0.95 cm); window R² 0.974 (1.36 cm).
   - The cube is harder (R² 0.83–0.88, error up to 4.9 cm); reported only.
   - Encoding took 2.9 minutes for 380,280 frames.
3. **Upper bound: passes.** The same predictor on the 40-number state of
   `scene-play-v0` (3 tokens per window instead of 588), trained 10
   minutes, read by a least-squares map:
   - Pulls: locked drawer 100% right, unlocked drawer 96.5%; locked
     window 100%, unlocked window 100%.
   - Presses: 92.9% flip the lock (button 0 94.5%, button 1 91.2%); 99.99%
     of quiet model steps show no false flip.
   - It overfits (training error near 0, held-out error flat from
     5 minutes on), but it shows the test can be passed.
4. **Baselines: pass.** "No change" is right on every locked pull and on
   none of the unlocked ones that moved (drawer 9%, window 0%), and on no
   press. "Lock-blind" is the reverse. Weighting the two lock states
   equally, each scores 0.45–0.55.

The predictor has 13.5M parameters (DINO-WM's has about 19M; the width
is 384 with a linear map in and out, so one design serves both inputs).
A 90-second smoke run was lock-blind: it opened 76% of locked drawer
pulls and 98% of locked window pulls, and flipped no lock.

## 6. Success criteria and prediction

Scored on held-out validation episodes, at the model step where the true
effect shows (first k with the true drawer or window moved 4 cm, or the
lock flipped; k ≤ 4).
1. **Pulls given the lock.** "The drawer (window) opens by at least 4 cm"
   is read from the predicted state by the probe and compared with the
   truth. Right on at least 90% of locked pulls and at least 90% of
   unlocked pulls, for drawer and window alike. Both baselines get 0% on
   one of the two.
2. **Presses.** The predicted lock state flips on at least 90% of presses
   (no change: 0%), and does not flip on at least 95% of model steps
   without a press.
3. **Near the upper bound.** On criteria 1 and 2, within 5 points of the
   state-based predictor.

Meeting all three shows that a model trained on pixels and actions alone
has the lock's effect, the first condition the new ladder needs. The
cube's motion is reported, not judged.

**Prediction.** The gate's probes pass: a red or white handle covers
several 14-pixel patches after upsampling. Criterion 2 passes. Criterion 1
is the risk, because a pull's outcome is decided by a few pixels while the
arm's motion dominates the squared error. LESSONS' warning applies:
average error drops rare certain changes (105.7). If so, the next card
changes the loss (card 105.8's likelihood form is the candidate), not the
encoder.

**Budget:**
- Download 2 GB; encode about 200,000 frames (every fifth) in under 5
  minutes; about 30 GB of fp16 features, memory-mapped.
- Predictor training up to 30 minutes, capped. If it has not converged,
  the longer run is handed to the user as a command (rule 5).
- Scoring under 5 minutes.

## 7. Result

The patch predictor trained for 25 minutes (32,059 steps, 6 epochs;
held-out error 0.20, still falling slowly). Held-out events, validation
plus training episodes 900–999:

| Measure | Patches | State upper bound | No change | Lock-blind | Actions alone | Bar |
|---|---|---|---|---|---|---|
| Drawer, locked pulls right | 100% (138) | 100% | 100% | 0% | 95% | ≥ 90% |
| Drawer, unlocked pulls right | 96.4% (555) | 96.5% | 9% | 91% | 96% | ≥ 90% |
| Window, locked pulls right | 97.4% (195) | 100% | 100% | 0% | 98.5% | ≥ 90% |
| Window, unlocked pulls right | 100% (651) | 100% | 0% | 100% | 100% | ≥ 90% |
| Presses: lock flips (1,346) | 95.3% | 92.9% | 0% | – | – | ≥ 90% |
| Quiet model steps without a false flip | 99.98% | 99.99% | 100% | – | – | ≥ 95% |

- **As written, all three criteria pass.** Every rate clears 90% (95% for
  false flips) and lies within 2.6 points of the upper bound.
- **The prediction was wrong.** Pulls were not the weak point: the smoke
  run's lock-blindness went away with training, and squared error did not
  drop the rare outcome.
- **But the test can be passed without seeing the lock.** The play policy
  steers toward a planned target, so on a locked slide the gripper is held
  back and the next commands stay saturated. A classifier given only the
  six action chunks the model sees (gradient-boosted trees, trained on
  episodes 0–899) is right on 95% of locked drawer pulls and 98.5% of
  locked window pulls. The first five actions alone give 69% and 97%. The
  gate had no action-only baseline, so it missed this.
- So the result does not show whether the model reads the red or white
  lock or the actions. Presses are less affected (the lock's new colour
  must be produced), but the same doubt applies.
- Reported, not judged: the cube's predicted position is 4.4–4.6 cm off
  at k = 1–4. The probe itself errs by up to 4.9 cm, so this measures the
  probe more than the model.

Verdict: **uninterpretable.** The criteria pass, but an untested
baseline also passes them.

## 8. Decision

**Revise.** The model, encoder and loss stay. The test changes: card
114.1 breaks the link between actions and outcome.

- **Counterfactual lock:** from each held-out pull, set the simulator to
  the stored state with the lock flipped, render the history frames, and
  replay the same recorded actions open loop to get the true outcome. An
  unlocked pull's actions then meet a locked slide, and the reverse. A
  model that reads the lock follows the colour; one that reads the
  actions follows the actions.
- **Gate additions:** the action-only baseline above must fall below the
  bar on the counterfactual events; and replaying recorded actions from a
  stored state must reproduce the recorded slide motion.
- Rendering works on this machine with `MUJOCO_GL=egl`, libglvnd on
  `LD_LIBRARY_PATH` and `__EGL_VENDOR_LIBRARY_DIRS` set (checked
  2026-10-31).
- No retraining: the trained predictor (`runs/114/pred_patch.pt`) is
  scored again.
