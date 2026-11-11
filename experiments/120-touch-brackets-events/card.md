---
id: "120"
title: touch brackets events — what differs between just before a touch and just after it is what the touch changed
rung: 2
serves: [P7, P4, P6]
status: done
verdict: pass
arch_version: 27
date: 2026-11-10
---

# 120: touch brackets events

Roadmap step 2 (event segmentation), rung 2. Card
[119](../119-variables-from-lasting-change/card.md) found condition
changes from lasting change per patch, but only 53% of lock flips and
13% of cube moves: both are a few patches wide and under the arm when
they change. The user (2026-11-10): "When I flip a switch or press a
button I feel a specific sensation." The simulator has that sensation
and the project has not used it. OGBench's proprioception includes
`gripper_contact`, the force on one finger pad scaled to 0–1. GOAL C1
admits proprioception, and CHARTER lists "the arm's own state, if
OGBench provides it" as still to be checked.

Biological counterpart: touch and vision are bound by timing. A felt
contact tells the brain when its own action met the world, and what is
seen to change at that moment is attributed to the action (Blakemore,
Wolpert and Frith 1998 for self-caused sensation; infants learn objects'
properties by handling them).

## 1. Question

If the agent compares the scene just before a touch begins with the
scene just after it ends, does the difference mark what the touch
changed, including lock flips and cube moves that card 119 misses? P7
(units from interaction), P4, and P6 (the touch is the agent's own
action meeting the world).

## 2. What changes

One component: where card 119's two windows are placed. Nothing is
trained.

```
119:  before = 10 frames ending 5 before τ, after = 10 frames starting 5 after τ, at every frame τ
120:  before = 10 frames ending 2 before a touch begins, after = 10 frames starting 2 after it ends
```

- **A touch:** a run of frames with contact above 0.05, gaps under 3
  frames merged.
- **Signal per touch and patch:** card 119's ratio (squared distance
  between the two windows' mean vectors over their variance plus a
  floor), on DINOv2 block 2 (card 119's choice).
- **Why it should help:** the arm has left by the time the "after"
  window is read, so a button it covered while pressing is visible
  again, in its new colour. A carried cube is seen gone from where it
  was and present where it was put.
- **Scoring:** for each true transition inside a touch, is one of the
  touch's 4 highest patches on its object (one patch of slack)? For
  each touch, is its top patch on an object that changed? Touches with
  no true transition should score low: reported as the area under the
  curve separating them from the rest.
- **Compared on the same episodes:** card 119's signal as it stands;
  the same windows placed at random frames (touch adds nothing);
  touch alone, with a random patch (timing without place).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The body reports contact | no touch sense; the nearest is the game's dig or hit feedback | none | none | yes: force or tactile sensors, motor current | declared: where there is no touch, card 119's signal stands alone |
| What matters changes while the agent touches it | no: creatures, night, growth change untouched | no, the same | mostly; some worlds animate | mostly; other agents do not | declared limit: touch finds the agent's own effects only |
| A patch shows the same place over time | no | no | yes | fixed cameras only | declared, as in card 119 |
| Windows of 10 frames, 2 frames clear of the touch | a setting | a setting | a setting | a setting | declared setting |

## 3. Dependencies

- Card 119's signal and scoring code; card 114's encoder; card 115's
  renderer for masks (its colour render matches the dataset's frames to
  0.06 of 255).
- `gripper_contact` is stored only in OGBench's state dataset
  `scene-play-v0` (observation index 18), not in `visual-scene-play-v0`.
  The two hold different episodes from the same generator. So this
  card's frames are rendered from the state dataset's stored joint
  positions and button states, 64 × 64, as card 115 rendered masks.
  Getting contact for the visual dataset would need the physics
  replayed, and is not attempted.

## 4. Data check

Run 2026-11-10 on `scene-play-v0`'s 100 validation episodes (state
only, a few seconds).

- **Touch is rare and specific.** Contact is above 0.05 in 28% of
  frames. 93% of those frames lie inside a true transition (± 5
  frames); mean contact outside transitions is 0.02.
- **Touch covers the transitions.** A touch overlaps 99.9% of lock
  flips, 97% of cube moves, 93% of window slides and 85% of drawer
  slides.
- **One touch is nearly one event.** 2,160 touches, median 7 frames,
  median 32 frames apart. 82% contain exactly one transition, 6% two
  (mostly the cube with the drawer), 11% none.
- **A slide can outlast its touch.** A touch holds the whole lock flip
  in 99.9% of cases and the whole cube move in 93%, but the whole
  drawer slide in 73% and the whole window slide in 41%.
- **Caveat on the data.** This is scripted play, where the arm touches
  only in order to manipulate. An agent that bumps into things would
  have many more touches without events.
- **Resistance did not work as tried.** The gap between the effector's
  motion and a linear prediction from the last three commands and the
  last motion was no more common inside transitions than outside (49%
  against 47% of frames). The prediction's use of the last motion
  probably absorbs the resistance; not pursued here.

## 5. Feasibility gate

Each step capped at 10 minutes (rule 5).
1. **Render** frames and masks for the state dataset's validation
   episodes and its training episodes 900–999, and check that card
   119's block-2 signal scores there as it did on the visual dataset
   (transitions F1 0.63 ± 0.05), so the two datasets are comparable.
2. **Upper bound:** the touch-bracketed signal with patches on the arm
   removed by the true mask in both windows.
3. **Baselines:** random frames with the same windows; touch with a
   random patch.

Result of the gate (2026-11-10): rendering took 65–71 s per set.
1. Card 119's signal on this dataset: transitions F1 0.66 (0.63 on the
   visual dataset), so the two are comparable.
2. Upper bound, arm masked: 96% of transitions found.
3. Random time 45%; touch with random patches **56%**. The second is far
   above the 10% this card predicted: with 4 patches, one patch of
   slack, and the whole drawer or window counted for a lock flip, the
   matching rule is lenient. The criteria were not changed; section 7
   adds a strict reading.

## 6. Success criteria and prediction

On validation episodes, per transition inside a touch:
1. **Lock flips:** found in place for at least 80% (card 119: 53%).
2. **Cube moves:** found in place for at least 60% (card 119 on block
   2: 13%).
3. **Touch adds place, not only time:** overall share found at least
   0.15 above "touch with a random patch".

Drawer and window slides, the share of touches whose top patch is on
the arm, and the separation of touches without a transition are
reported.

Meeting all three shows that a single contact number, with no location
of its own, lets vision say what the agent's action changed.

**Prediction.** Criterion 1 passes: the button is clear of the arm in
both windows. Criterion 2 is the risk: the cube is 3 patches, and where
it was put may still be under the opening gripper 2 frames after
release. Criterion 3 passes easily, since a random patch is right
about 10% of the time.

**Budget:** rendering 200,200 frames with masks, about 6 minutes at
card 115's speed; encoding block 2, 1 minute; scoring, 2 minutes.

## 7. Result

Run 2026-11-10 (approved by the user the same day), 2,110 touches on
validation episodes (50 dropped at episode ends), 1,936 transitions
inside a touch.

| Found in place (top 4, one patch of slack) | Lock | Cube | Drawer | Window | All |
|---|---|---|---|---|---|
| Before and after the touch (this card) | **99%** | **94%** | 98% | 97% | **97%** |
| The same, arm masked (bound) | 99% | 88% | 99% | 98% | 96% |
| Card 119's signal at its best frame, no threshold | 98% | 54% | 100% | 100% | 86% |
| Windows at a random time | 46% | 46% | 49% | 38% | 45% |
| Touch, 4 random patches | 69% | 35% | 63% | 58% | 56% |

1. **Lock flips: pass** (99% against 80%).
2. **Cube moves: pass** (94% against 60%).
3. **Place, not only time: pass** (97% against 56%, +0.41).

Touches with no transition score lower than the rest (area under the
curve 0.89). The top patch is on the arm in 32% of touches.

**Strict reading, after the run** (a diagnostic, not a criterion): the
single top patch, exactly on the object.

| Top patch exactly on | Lock: the button | Lock: the handle's body | Cube | Drawer | Window |
|---|---|---|---|---|---|
| This card | 2% | 84% | 35% | 25% | 51% |
| Card 119, best frame | 0% | | 13% | 82% | 98% |
| Chance | 1% | | 2% | 12% | 10% |

- **A lock flip is seen on the handle, not the button.** The handle
  changes colour too and is larger. The button's best patch ranks 9th
  at the median and is in the top 4 for 7% of flips. So criterion 1
  passed through the handle; at this resolution the button itself is
  not found by either signal.
- **The cube is this card's real gain:** in the top 4 exactly on the
  cube for 74% of moves, top patch 35% against 13%.
- **Slides are found worse than by card 119's fixed windows** on the
  strict reading. A slide often outlasts its touch (data check), so
  the "after" window is read while the drawer or window still moves.
- **Card 119's lock and cube misses were partly its threshold.** With
  no threshold it locates 98% of lock flips (through the handle).

Verdict: **pass**, on a matching rule more lenient than intended.

## 8. Decision

**Keep,** as a second source beside card 119's signal, not a
replacement. Touch says when the agent's own action met the world, with
no threshold, and comparing the scene before and after it locates what
moved under the hand: the cube, which fixed windows miss. Fixed windows
stay for slides and for changes the agent does not touch. Limits to
carry: contact exists only where the body reports it (portability
table); this play data touches only to manipulate; and the button
itself is still not found at 64 × 64. A later card that sets bars on
"found in place" must first measure what random patches score under
its matching rule.
