---
id: "126"
title: a dense layer trained on what a touch does — gate only: no room above memory without training
rung: 2
serves: [P7, P4, P1]
status: draft
verdict:
arch_version: 27
date: 2026-11-10
---

# 126: a dense layer trained on what a touch does

Roadmap step 2 (event segmentation), rung 2. The user (2026-11-10):
train dense per-patch features over several encoder depths by
consequences, "not just next token or pixel prediction, we need an
abstraction layer beyond that". Cards 125–125.2 reached the ceiling
on every lock test with frozen features, so this card was to be judged
on what they do not cover: the cube, and compact records. Only the
gate was run. It shows no room for a trained layer on the measures
available in this scene, so sections 6–8 are open and the decision is
the user's.

Biological counterpart: as card 124 (object knowledge organised by
what can be done to things; Goodale and Milner 1992).

## 1. Question

Does a small code per patch, trained to predict what a touch changes
(whether, where, and the settled state after), keep what the lock
tests need and predict the cube's outcomes better than memory alone?
P7, P4, P1.

## 2. What changes

One component, not built: a learned layer between the frozen encoder
and the records of card 125.

```
125:  record = frozen features (768 numbers per patch) before the touch
126:  record = a code (about 16 numbers per patch) mixed from several depths,
      trained so that the codes and the gripper's motion predict the touch's lasting change
```

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A touch is described by the effector's start and end | the target is at the view's centre | the facing tile | the clicked cell | holds | proprioception |
| The lasting-change signal is a usable teacher | needs patches placed in the world | the same | holds | fixed cameras only | **weak here for small objects** (gate) |

## 3. Dependencies

Cards 122 (change signal), 124 (stored features), 125–125.2 (records
and tests).

## 4. Data check

Memory: episodes 600–699 (2,008 touches). Test: 500–599 (2,032
touches); 86% have a true change. Touches with exactly one true
change: cube 466, drawer 245, window 279, button presses 361 and 322.

## 5. Feasibility gate

Run 2026-11-10 (`tools/card126/gate.py`, capped, under a minute). The
task: score every patch for "will have changed lastingly after this
touch", from the scene before and the gripper's position at contact
and release. Truth, for scoring only: the patches of each object that
truly moved (before and after) or of a button that flipped.

Three arms, none trained:
- **the teacher:** the touch's own observed change signal (card 122's,
  block 2). A layer trained on that signal cannot be expected to beat
  it;
- **recall by motion:** the mean change signal of the 8 remembered
  touches with the nearest gripper start and end. It never looks at
  the scene;
- **nothing:** a constant map.

Average precision of the patch scores, by what truly changed:

| | Cube | Drawer | Window | Button presses |
|---|---|---|---|---|
| Nothing | 0.015 | 0.10 | 0.08 | 0.007 |
| Recall by motion | 0.41 | 0.71 | 0.46 | 0.05–0.09 |
| The teacher | 0.44 | 0.67 | 0.43 | 0.08–0.11 |

Whether a touch changes anything (area under the ROC curve): recall by
motion 0.90, the teacher 0.92.

**Gate result: not passed.** Two findings.
- **The upper bound sits on the trivial baseline.** Where a touch
  changes the scene is given almost entirely by where the gripper
  goes; recall by motion equals the teacher without seeing the scene.
  The one thing the scene adds, whether a pull is locked, is what card
  125 already reads.
- **The teacher is weak.** Against the true patches the change signal
  reaches 0.44 on the cube and about 0.1 on a button (the handle and
  the arm outrank it). A layer trained on it inherits that.

Also from card 125.2's gate (`transfer_gate.py`): a lock's state read
on one button carries to the other on frozen features at 98–100%, so
that test of abstraction has no room either.

## 6. Success criteria and prediction

Not written: the gate did not pass.

## 7. Result

None.

## 8. Decision

Open, for the user.
