---
id: "114.1"
title: counterfactual lock — does the pixel model follow the lock's colour or the actions?
rung: 1
serves: [P6, P4]
status: done
verdict: pass
arch_version: 27
date: 2026-10-31
---

# 114.1: counterfactual lock

Card [114](../114-lock-effects-from-pixels/card.md) passed its criteria
but was uninterpretable. In OGBench's play data the actions after a pull
show whether the slide moved: the controller steers toward a target, so
on a locked slide the gripper is held back and keeps being sent forward,
while on an unlocked one it arrives and turns back. Actions alone were
right on 95% of locked drawer pulls and 98.5% of locked window pulls. This
card scores the same trained model on pulls where the lock is changed and
the actions are not.

## 1. Question

When only the lock changes (same arm, same actions, red handle instead
of white or the reverse), does card 114's predictor change its forecast
of whether the drawer or window opens? Rung 1 (P6: the effect of an
action, predicted correctly under each condition); P4 (a few red or
white pixels decide the outcome).

## 2. What changes

Nothing in the model. The trained predictor (`runs/114/pred_patch.pt`),
the frozen DINOv2 encoder and the probe are reused unchanged, with no
retraining. Only the test events change.

```
card 114:  recorded frames (lock as it was) + recorded actions → forecast vs recorded outcome
114.1:     simulator set to the recorded state with the lock flipped
           → rendered frames (lock flipped) + the same recorded actions
           → forecast vs the simulator's outcome for those actions
```

- **Counterfactual events.** For each held-out pull of card 114 (start
  frame t):
  1. Set OGBench's scene to the stored joint positions and velocities at
     t − 10, with that slide's lock flipped (`set_state` changes the
     colour of the button and handle, and the slide's damping).
  2. Replay the 30 recorded actions from t − 10 to t + 19 in the
     simulator.
  3. Render the frames at t − 10, t − 5 and t as the model's history.
  4. Record the slide's position at t + 5, …, t + 20 as the truth.
- An unlocked pull becomes **newly locked**: the actions of a pull that
  opened meet a slide that cannot move. A locked pull becomes **newly
  unlocked**: the actions of a stalled pull meet a free slide.
- **Factual replays** use the same procedure without the flip. They check
  that rendering and replay reproduce card 114's inputs and outcomes.
- **Scoring** is card 114's (`tools/card114/score.py`): first step where
  the true slide has moved 4 cm, or no move predicted at any k ≤ 4. The
  model is causal over steps, so its forecast at step k uses only the
  actions up to that step.
- **Presses are not retested.** To predict a flip, the model must already
  read the button's current colour from the frame, because the actions do
  not say which way the lock goes.

### Portability check (CHARTER rule 9)

Card 114's table holds for the model. The new assumption is about the
test only:

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The simulator can be set to a stored state, changed, and re-rendered | partly (world edits by mod) | yes (pickled state) | no in general | no; the agent's own varied tries play this role | scoring device only, never seen by the model |

## 3. Dependencies

- Card 114's trained predictor, probe and held-out events (gate passed;
  probes read the lock at 100%).
- OGBench's scene simulator (`ogbench`), the same one that generated the
  data, used for the truth.
- Rendering with MuJoCo's EGL backend, checked on this machine on
  2026-10-31: libglvnd on `LD_LIBRARY_PATH` and
  `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`.

## 4. Data check

The events are card 114's held-out pulls (validation episodes and
training episodes 900–999): drawer 138 locked and 555 unlocked, window
195 locked and 651 unlocked. Each gives one counterfactual event. The
events that test the question are those whose **true outcome changes**
when the lock flips. Every newly locked pull should stay shut. A newly
unlocked pull opens only if the stalled pull's actions push far enough.
The gate counts both, and each of the four kinds (drawer or window,
newly locked or newly unlocked) needs at least 100 changed-outcome
events. If newly unlocked drawer pulls fall short (138 at most), the
card says so and judges that kind on what there is.

## 5. Feasibility gate

Each step under 10 minutes and capped (rule 5).
1. **Replay fidelity.** Factual replays reproduce the recorded slide
   position within 1 cm at t + 5, …, t + 20 on at least 95% of events.
   Rendered frames match the dataset's frames (mean absolute difference
   under 2 of 255), or their patch features give the probe the same
   lock and slide readings. The model's score on factual replays is
   within 3 points of card 114's, so any failure below comes from the
   lock, not the rendering.
2. **Counts** of changed-outcome events, as in section 4.
3. **The test breaks the shortcut.** Card 114's action-only classifier
   (`tools/card114/actions_only.py`) on the changed-outcome events
   should score near 0%, because the actions point to the old outcome.
   If it scores 50% or more, the flip has not separated actions from
   outcome, and the card stops.
4. **The test can be passed.** A lock-reading rule (the probe's lock
   state on the real start frame: locked means "stays shut", unlocked
   means "opens") scores at least 90% on each kind. This is the upper
   bound.
5. **Comparison, not a bound.** Card 114's state-input predictor is
   scored on counterfactual events built the same way from
   `scene-play-v0`. Its input holds the lock as a clean one-hot number,
   but it learned from the same correlated play, so it may take the same
   shortcut. It tells which of two causes a failure has (section 6).

Result of the gate (2026-11-01, code in `tools/card114.1/`):
1. **Replay fidelity: one sub-check missed its bar.**
   - **Slide position within 1 cm** at every step: 89.3% of factual
     replays, against the 95% asked. The misses are in how far a moving
     slide travels (median error 1.7 cm), not whether it moves: none is on
     a locked pull.
   - **Outcome (opened or not):** agrees with the record on 98.4%.
   - **Frames:** rendered frames differ from the dataset's by 0.28 of 255
     on average (largest 2.1).
   - **Model on factual replays:** within 1.3 points of card 114 on every
     kind.
   - The state replays are alike (90.6% within 1 cm, 99.1% same outcome).
   - **Judged not to bear on the result:** a counterfactual's truth is the
     simulator's own replay, not the recording, so drift from the record
     does not touch it. Flagged for the user.
2. **Counts: pass.** Changed-outcome events: drawer newly locked 514,
   newly unlocked 127; window newly locked 650, newly unlocked 194.
3. **The shortcut is broken: pass.** The action-only classifier is right
   on 0–4.7% of changed-outcome events.
4. **The test can be passed: pass.** The lock-reading rule is right on
   100% of each kind.
5. **State comparison:** passes too (next section).

## 6. Success criteria and prediction

1. **Counterfactual pulls.** On changed-outcome events, the forecast is
   right on at least 90% of each kind: drawer newly locked, drawer newly
   unlocked, window newly locked, window newly unlocked. The action-only
   classifier gets near 0% here, and the lock-reading rule at least 90%.

Meeting it shows that the pixel model's forecast of a pull depends on the
lock it sees, the first condition the new ladder needs.

**If it fails,** the state comparison says why:
- **State model passes, pixel model fails:** the cue is in the input but
  the pixel pathway does not carry it to the forecast. The next card
  changes the loss (card 105.8's likelihood form).
- **Both fail:** correlated play teaches the shortcut whatever the
  input. The next card changes the data: the agent varies its own tries,
  pulling the same way whatever the colour. The counterfactual replays
  are themselves such data. Biology does the same: preschoolers explore
  more when the evidence is confounded, which is how they separate causes
  (Schulz and Bonawitz 2007; Gopnik and Sobel 2000, the "blicket
  detector"). This fits CHARTER's conditions from memory and experiment
  (card 111).

**Prediction.** The lock is in every history frame, while the actions
that give the outcome away come later (chunks 2–4). We expect the model
to use both cues, and to fail criterion 1 on **newly locked** pulls. They
are scored as "no move at any k ≤ 4", and the later actions that turn
back say "it opened". We expect 50–80% there and above 90% on newly
unlocked pulls, with the state model failing the same way. If so, the
cause is the data.

**Budget:**
- Replays: about 1,540 pulls × 2 (factual and counterfactual) × 30
  simulator steps, plus 3 rendered frames each, for both datasets: about
  5 minutes.
- Encoding 9,000 frames and scoring: about 1 minute.
- No training.

## 7. Result

Changed-outcome events (the counterfactual's true outcome differs from
its factual twin), held-out pulls of card 114:

| Kind | n | Pixel model | State model | Actions only | Lock-reading rule |
|---|---|---|---|---|---|
| Drawer, newly locked | 514 | 100% | 99.8% (532) | 4.7% | 100% |
| Drawer, newly unlocked | 127 | 96.9% | 97.9% (95) | 4.7% | 100% |
| Window, newly locked | 650 | 99.5% | 98.6% (641) | 0% | 100% |
| Window, newly unlocked | 194 | 100% | 100% (159) | 1.5% | 100% |

- **Criterion 1 passes:** every kind is at least 96.9%, against 90%.
  When the lock changes and the actions do not, the forecast follows the
  lock. For example, the actions of a pull that opened, given a red
  handle, are forecast to leave the drawer shut on 100% of events.
- **The prediction was wrong.** The model does not lean on the actions,
  and nothing failed, so neither the loss nor the data needs changing.
  The play data does let each pull be read from the lock: the colour is
  in every history frame, while the actions give the outcome away only
  later.
- The state model behaves the same, so the clean lock number and the red
  or white pixels serve the forecast equally.
- Together with card 114 (presses flip the lock at 95%; false flips on
  0.02% of quiet steps), rung 1's test is met: from pixels and actions
  alone, the model predicts that a pull opens only when unlocked, and
  that a press changes the lock.

Verdict: **pass**, with the replay-fidelity sub-check missed as stated
in the gate.

## 8. Decision

**Keep.** Card 114's predictor on frozen DINOv2 patches is System 1 for
the learned stack; rung 1 is met. Next steps, for the user:
- **The architecture record.** ARCHITECTURE.md still describes MiniGrid
  version 27. Writing this predictor in as version 28 starts the
  portability tests (Crafter, Craftium) that AGENTS asks for before
  adoption, with measures this card's successor defines.
- **Card 115** (from card 113): HIQL on `visual-scene-play-v0`, the
  baseline for rungs 2 and 4.
