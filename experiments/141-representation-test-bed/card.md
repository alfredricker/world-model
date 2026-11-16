---
id: "141"
title: a representation test bed — every question the agent asks of a frame, with the exact answer, inside and outside what its training signal covered
rung: 3
serves: [P1, P4, P5]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 141: a representation test bed

Roadmap step 2 (the perceptual side), rung 3. First card of the
representation week (the user, 2026-11-13). The memory week assumed
perfect perception; this week assumes perfect memory and asks what a
frame can be made to say. Like [card 132](../132-memory-test-bed/card.md)
and [card 136](../136-memory-world/card.md), this card changes
nothing in the agent: it builds the test and measures the current
readings on it.

The problem ranked first in the graph is
[a reading knows only what its training covered](../../graph/problem/perception-from-narrow-signal.md).
The cube's position is read to 0.43 cm inside the range of
remembered grasps and 9.2 cm outside it (card 128); with the cube in
the open drawer the reading is valid in 0.4% of frames (card 131); a
window left 4 to 10 cm open is read as closed (card 130).

Biological counterpart: the ventral visual stream is shaped by
general statistics long before any one task, and acting on objects
then tunes it; a monkey that has only grasped fruit from a table
still sees the fruit in a box. A reading that exists only where the
hand has been is the opposite arrangement.

## 1. Question

For each question the agent asks of a frame, how far is its current
reading from the best answer the same frozen features allow, and
how much of that distance appears only outside what the reading's
training signal covered? P1 (grounded meaning), P4 (task-relevant
abstraction), P5 (knowing where a prediction applies).

## 2. What changes

Nothing in the agent. One new test.

```
frames     the frame before each touch of recorded play, as stored frozen DINOv2 patch
           features (runs/125/before_<set>.npy); 8,230 frames to learn from, 6,170 to test
truth      cube position, drawer and window opening, two locks (scores only)
questions  1 same or different: for a pair of frames from different episodes, does
             variable k differ? This is what the planner asks of (now, goal).
           2 value: where is the cube; how far open is each slide; is each lock on
           3 in view: can the cube be located at all (it is hidden in the closed drawer)
cells      cube: on the table / in the open drawer / in the closed drawer
           slides: pairs differing by under 1 cm / 1 to 4 / 4 to 10 / over 10
           locks: same / different
covered    each reading is also trained with part of a variable's range withheld from
           its training signal (a third of the table for the cube; openings of 4 to 10 cm
           for a slide) and scored inside and outside that part
```

Arms:

- **Current readings (og4):** the aim read-out trained on the
  agent's own contact points, with its validity threshold, and the
  appearance reading (10 patches per place compared between two
  frames).
- **Labelled probe (upper bound):** a small attention read-out over
  the same frozen features, trained on the true state. It says what
  the features hold; the agent may not use it.
- **Base rate (trivial baseline):** always "same"; the mean value.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A true state exists to score against | from the engine | from the engine | from the grid | needs instruments | scores only, never an input |
| The scene's variables are known and few | no | no | no | no | declared for the test; the readings under test are not told them |
| A frozen pretrained encoder gives the features | holds | weak on pixel art | weak | holds | adopted (card 114.1) |
| One fixed camera | no, the view moves | no | yes | often no | declared; a moving view is a later cell of this bed |
| Frames come from recorded play | own play instead | own play | own play | holds | general: any stored frames with truth |

## 3. Dependencies

- Stored features and truth: cards 122, 125 and 132 (`runs/132/ev_<set>.npz`).
- The current readings: cards 128 to 129.2 (`runs/129/readout_q.pt`,
  `tools/card129.1/`).
- A probe on frozen features is an established method (linear and
  attentive probing of self-supervised encoders; DINOv2 in
  LITERATURE.md).

## 4. Data check

Done 2026-11-13 from the stored truth (under a minute).

| Cell | Frames to learn from | Frames to test |
|---|---|---|
| Cube on the table | 6,587 | 4,793 |
| Cube in the open drawer | 1,048 | 985 |
| Cube in the closed drawer | 595 | 392 |
| Drawer open under 1 cm / 1 to 4 / 4 to 10 / over 10 | 2,259 / 1,186 / 1,490 / 3,295 | 1,499 / 1,050 / 1,005 / 2,616 |
| Window open under 1 cm / 1 to 4 / 4 to 10 / over 10 | 2,206 / 848 / 1,100 / 4,076 | 1,699 / 648 / 795 / 3,028 |
| Each lock on | 52% and 55% | 57% and 55% |

Every cell has several hundred frames, so every pair cell has
thousands of pairs.

One limit found: recorded play does not leave the cube anywhere new.
A test cube on the table lies a median of 0.19 cm from a cube
position in the learning frames, and only 5 of 4,793 are over 3 cm
away. So "outside the covered range" cannot be read off the stored
frames as they are; it is made by withholding a part of the range
from a reading's training signal, as section 2 says.

## 5. Feasibility gate

This card is the gate for the week: its result is the pair of bounds
per cell.

- **Upper bound:** the labelled probe.
- **Trivial baseline:** the base rate.
- The gate passes if the probe is at least 20 points of balanced
  accuracy (the mean of the accuracy on each of the two outcomes;
  50% is chance) above the base rate in every same-or-different
  cell. A cell where it is not is dropped from the bed and reported:
  the frozen features do not hold that answer, which is a finding
  about the encoder.

Result of the gate (2026-11-14): passed in every cell. The probe is
32 to 49 points above the base rate (50 in every cell, by
construction); the lowest cell is window pairs differing by 1 to
4 cm (82.3).

## 6. Success criteria and prediction

Proposed, to be fixed with the user.

1. **The test can be passed:** the labelled probe scores at least
   95% balanced accuracy on same or different in every kept cell,
   locates the cube to a median of 1 cm on the table and in the open
   drawer, and reads a slide's opening to a median of 1 cm.
2. **The test shows the known faults:** the current readings are at
   least 15 points under the probe in the two cells where the agent
   fails today: the cube in the open drawer, and slides differing by
   4 to 10 cm.
3. **Coverage is what the test measures:** with a third of the table
   withheld from the aim read-out's training signal, its median
   error is at least 5 times larger in the withheld part than in the
   rest (card 128: 9.2 against 0.43 cm), while the probe trained
   with the same part withheld is reported beside it.

Reported without a bar: every arm in every cell, and the "in view"
question. These decide what the week's cards must beat.

**Prediction.** 1 passes, with the 1 to 4 cm slide cell the doubtful
one. 2 passes. 3 passes for the read-out; the open point is whether
the probe also fails in the withheld part. If it does, labels are
not the cure and the fault is in how a position is read from
patches, not in which signal trains it.

**Budget.** No rendering. Two probes and two retrained read-outs,
each a few minutes on the GPU; every run capped and under 10
minutes.

## 7. Result

Run 2026-11-14, `tools/card141/` (`current.py`, `probe.py`,
`cover.py`, `score.py`; each run under 7 minutes),
`runs/141/bed.json`, numbers in [results.json](results.json). 6,170
test frames, 12,340 pairs. Balanced accuracy on "does this variable
differ between the two frames" (50 is chance and the base rate).

| Cell (pairs same / different) | Current readings | Labelled probe |
|---|---|---|
| Cube, both on the table (1,153 / 6,494) | 99.4 | 99.0 |
| Cube, one in the open drawer (350 / 2,417) | 70.6 | 99.0 |
| Cube, one hidden in the closed drawer (152 / 1,242) | 61.0 | 82.4 |
| Drawer differs by 1 to 4 cm (4,267 / 1,801) | 69.8 | 87.4 |
| Drawer differs by 4 to 10 cm (4,267 / 2,227) | 81.8 | 98.7 |
| Drawer differs by over 10 cm (4,267 / 4,045) | 93.1 | 98.9 |
| Window differs by 1 to 4 cm (3,962 / 1,712) | 59.6 | 82.3 |
| Window differs by 4 to 10 cm (3,962 / 2,120) | 76.6 | 99.2 |
| Window differs by over 10 cm (3,962 / 4,546) | 96.3 | 99.3 |
| Drawer's lock (6,737 / 5,603) | 89.0 | 96.7 |
| Window's lock (6,752 / 5,588) | 92.0 | 92.3 |

Values, median error against the truth:

| Question | Current reading | Labelled probe | Base rate (the mean) |
|---|---|---|---|
| Cube on the table | 0.60 cm | 0.18 cm | 8.2 cm |
| Cube in the open drawer | 27.7 cm | 0.39 cm | |
| Drawer opening | no value | 0.12 cm | 6.9 cm |
| Window opening | no value | 0.12 cm | 8.6 cm |
| Lock on or off | 94 to 96% | 98.3 and 96.1% | 55% |

With card 131's "in view" threshold added to the aim (not part of
og4), the cube cells read 96.5, 92.2 and 80.9: the open drawer is
then told apart from the table, but the cube in it is still not
located.

Part of the range withheld from training, median error:

| Reading | Kept part | Withheld part |
|---|---|---|
| Aim read-out, cube on the table, upper third withheld | 0.64 cm | 2.18 cm (0.88, 1.83 and 4.36 cm at 0 to 2, 2 to 5 and 5 to 12 cm past the edge) |
| Labelled probe, the same third withheld | 0.19 cm | 1.91 cm |
| Labelled probe, drawer 4 to 10 cm open withheld | 0.10 to 0.16 cm | 0.85 cm (0.14 when not withheld) |
| Labelled probe, window 4 to 10 cm open withheld | 0.07 to 0.17 cm | 1.71 cm (0.16 when not withheld) |
| Appearance reading, remembered pairs with the slide 4 to 10 cm open dropped | | pairs differing by 4 to 10 cm: drawer 81.8 → 78.9, window 76.6 → 75.1 |

1. **The test can be passed: fail.** The values pass (cube 0.18 and
   0.39 cm, slides 0.12 cm, against a bar of 1 cm). Same or
   different is under 95 in four of eleven cells: both slides at 1
   to 4 cm (87.4, 82.3), the hidden cube (82.4) and the window's
   lock (92.3).
2. **The test shows the known faults: pass.** The current readings
   are 28 points under the probe with the cube in the open drawer,
   and 17 (drawer) and 23 (window) under it for slides differing by
   4 to 10 cm.
3. **Coverage is what the test measures: fail on the number.** The
   read-out's error in the withheld third is 3.4 times that in the
   kept part (bar 5 times). It grows with distance past the edge and
   reaches 6.8 times beyond 5 cm.

What the run showed beyond the criteria:

- **Labels are not the cure.** The probe trained on the true cube
  position fails in the withheld third as the read-out does (1.91
  against 2.18 cm), and its slide error grows 6 to 10 times in a
  withheld range. A position read by a trained map from patches is
  right only where that map was trained, whatever signal trains it.
- **Comparison by appearance barely depends on coverage.** Dropping
  the remembered pairs in the withheld range costs the appearance
  reading 1.5 to 2.9 points. It is coarse everywhere instead: 60 to
  82% below 10 cm.
- **The frozen features hold what the agent lacks.** The cube in the
  open drawer can be located to 0.39 cm, and a slide's opening read
  to 0.12 cm, from the features og4 already uses.

Two checks afterwards on why criterion 1 fell short (not declared;
`tools/card141/posthoc.py`):

- The 1 to 4 cm cells are low because "same" (under 1 cm) and
  "different" (from 1 cm) touch with no gap. With a gap (same under
  0.5 cm, different 2 to 4 cm) the probe scores 97.8 (drawer) and
  95.1 (window).
- The probe's "can the cube be located" answer is unsteady. This fit
  called the cube locatable in 29% of 168 frames with the drawer
  shut (under 1 cm); the second fit (the withheld one) did so in 2%
  and scores 94.4 in the hidden-cube cell. One fit per arm was run,
  so the spread is not measured.

## 8. Decision

**Revise.** The bed is kept with two repairs before it is used as a
bar: a gap between "same" and "different" in the smallest slide
cell, and the probe run over several seeds with its "locatable"
answer made steady. Next card, for the user to approve: a reading
that compares instead of mapping. The cube's position, and a
slide's opening, would be answered by recall over stored frames
whose outcome is known from the agent's own touches, matched on the
patches that the touch changed, so that an unseen position is
reached by interpolating between remembered ones and not by a map
that ends where training ended.
