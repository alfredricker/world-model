---
id: "125"
title: conditions by contrast over remembered touches — what differs in the scene before touches that worked and touches that did not
rung: 2
serves: [P6, P4, P7, P1]
status: done
verdict: fail
arch_version: 27
date: 2026-11-10
---

# 125: conditions by contrast over remembered touches

Roadmap step 2 (event segmentation), rung 2. Cards 116–124 tried to
get one token per object first and failed each time on outlines, while
the parts that need no outline worked: touch brackets the agent's
events (card 120), the change signal says where an event happened
(card 122), and a per-patch predictor follows the lock (card 114.1).
The user (2026-11-10) agreed to make the event the unit: a memory of
touches, and conditions read from it by contrast.

Biological counterpart: people judge a cause by contrasting outcomes
with and without a factor, within cases that are otherwise alike
(Cheng and Novick 1990, focal sets; card 111 used the same idea on
symbols). Here the cases alike are touches at the same place.

## 1. Question

From a memory of its own touches, with no object outlines and no
labels, can the agent find which part of the scene decides whether a
touch has an effect, and which other touch sets that part? P6
(conditions), P4 (memory), P7 and P1.

## 2. What changes

One component: how a condition is read. Nothing is trained.

```
124:  frames → 12 object tokens (trained) → transition → does the lock change the prediction?
125:  each touch → a record: gripper position, patches before, which patches changed
      records at the same place, split by "changed" / "did not" → the patches whose
      before-features differ are the candidate condition
```

- **Record:** the gripper's position at contact (proprioception, given
  by GOAL C1); blocks 2 and 12 of the frozen encoder per patch, averaged
  over 3 frames ending 2 frames before contact (256 pixels, 32 × 32
  patches); card 122's lasting-change map across the touch.
- **Place:** k-means (10) over the gripper positions in memory.
- **Outcome:** the touch's change score (mean of the 4 highest
  patches), split in two by Otsu's threshold on its logarithm.
- **Contrast:** per patch, the squared difference between the two
  outcome groups' mean before-features over their pooled variance,
  summed over the 768 numbers. The 10 highest patches are the
  candidate condition. The same contrast with outcomes shuffled gives
  the level expected by chance.
- **Chain (main run only):** the touches whose change map covers the
  candidate patches are the events that set the condition.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A contact signal marks the agent's events | none; an action's use would stand in | the same | a click or key press | holds | touch from the state dataset (card 120) |
| The agent knows where its effector is | pose given, the target is at the view's centre | the facing tile | the clicked cell | holds | proprioception |
| A fixed camera, so a patch is a place | no; needs patches placed in the world | no (scrolling view) | yes | fixed cameras only | declared since card 119 |
| An outcome is "changed" or "did not" | holds for most acts | holds | holds | holds | a two-way split; graded outcomes not handled |

## 3. Dependencies

- Cards 120 and 122 (touch brackets, 256-pixel change maps), card 114's
  encoder. Card 111 for contrast within alike cases.
- Preconditions as classifiers over the states where a skill succeeds:
  Konidaris et al. 2018 (LITERATURE), who leave image preconditions
  open. Contrast over memory on patch features is not an established
  method; the gate below is the evidence it can work here.

## 4. Data check

OGBench's state dataset. Memory: episodes 900–999, 2,088 touches. Test:
the 100 validation episodes, 2,110 touches. 70% of touches are followed
by a lasting change under the outcome rule. The 10 places hold 23 to
449 touches each.

## 5. Feasibility gate

Run 2026-11-10 (`memory.py`, 1 minute per set; `gate.py`, under a
minute; both capped).

**Where the contrast points.** Truth is used only to name the places
and score the patches. A button covers 0.8% of the patches.

| Place (named by truth) | Touches in memory | Contrast against shuffled | Of the 10 candidate patches, on the locking button |
|---|---|---|---|
| Drawer handle (two places) | 229, 146 | 108 times, 70 times | 8 of 10, 8 of 10 (button 1) |
| Window handle | 362 | 16 times | 8 of 10 (button 2) |
| Cube, table (five places) | 117–137 | 1.0–1.9 times | 0 |
| Buttons (two places) | 449, 380 | 49 times, 20 times | 0; arm and drawer patches |

- In every handle place the contrast finds the button that locks that
  slide, and it finds nothing where the lock does not matter.
- No candidate patch lay on the handle itself, which is painted too:
  the arm covers it just before a pull.
- At the buttons the contrast is large and lies mostly on the arm. A
  press that does not register differs by the arm's pose. The method
  has no notion of the self; card 117 is where that belongs.

**Prediction on validation touches**, balanced accuracy for "did the
slide move":

| Place | Candidate patches | All patches | Gripper position | True lock (upper bound) |
|---|---|---|---|---|
| Drawer handle, first | 0.854 | 0.806 | 0.655 | 0.854 |
| Drawer handle, second | 0.727 | 0.717 | 0.768 | 0.727 |
| Window handle | 0.691 | 0.501 | 0.543 | 0.691 |

The candidate patches equal the true lock in all three places. The
upper bound is below 1 because not every touch near a handle is a
pull; in the second place the gripper's position says more than the
lock does.

**Gate result:** passed. Limits: one setting of each choice (10
places, 10 patches), and the validation episodes have now been seen,
so the main run uses episodes not yet touched.

## 6. Success criteria and prediction

Main run on fresh episodes of the state dataset: memory 800–899, test
700–799, each rendered by card 122's script (5 minutes per set, with a
new flag for the first episode). Settings fixed as in the gate.

1. **Found:** in every place where a slide moves in at least 30% of
   touches (truth, for scoring only), at least 6 of the 10 candidate
   patches lie on the button that locks it, and the contrast is at
   least 5 times the shuffled level. In every place where no slide
   moves and no button flips, it is under 5 times.
2. **Predicts:** in those slide places, the candidate patches predict
   "the slide moved" on test touches within 0.03 of the true lock's
   balanced accuracy.
3. **Chain:** of the memory's touches whose change map has at least 3
   of a slide place's 10 candidate patches in its top 1%, at least 80%
   are presses of the button that locks that slide (truth), and they
   lie in one place.

**Prediction.** 1 and 2 repeat the gate on new episodes and should
pass. 3 is new and is the risk: card 122 found the button in the top
1% of patches at only 27% of flips, because the handle outranks it, so
the 80% may hold while few presses are caught. The share of presses
caught is reported.

**Not in this card:** a condition whose outcome is graded (where the
cube lands); the self (button places); a learned layer. Card 126
follows: a per-patch layer over several depths, trained on what a
touch does (its outcome and after-state from the scene before it, not
the next frame), and judged by this card's tests.

**Budget.** About 15 minutes in steps under 10 minutes, capped (rule
5). `runs/125/before_*.npy` is 3.3 GB per set.

## 7. Result

Main run 2026-11-10 on episodes not seen before: memory 800–899
(2,028 touches), test 700–799 (2,028 touches). `main.py`, criteria
written in it before the run. Two details the card left open were
fixed there: a quiet place is one where each slide moves in under 5%
of touches and a button flips in under 5%; "one place" means at least
80% in a single place.

**1. Found: pass.** Four slide places and three quiet places.

| Place (named by truth) | Touches in memory | Contrast against shuffled (bar 5) | Candidate patches on the locking button (bar 6) |
|---|---|---|---|
| Drawer handle | 179 | 33 times | 8 of 10 |
| Drawer handle | 217 | 88 times | 8 of 10 |
| Window handle | 154 | 16 times | 8 of 10 |
| Window handle | 251 | 12 times | 8 of 10 |
| Quiet (three) | 140–203 | 0.8, 1.7, 2.5 times | not counted |

A button's usual place is 8 patches, so 8 of 10 is the whole button.

**2. Predicts: pass.** Balanced accuracy for "the slide moved" on test
touches; the bar is within 0.03 of the true lock.

| Place | Candidate patches | All patches | Gripper position | True lock |
|---|---|---|---|---|
| Drawer handle | 0.775 | 0.710 | 0.676 | 0.775 |
| Drawer handle | 0.833 | 0.780 | 0.663 | 0.833 |
| Window handle | 0.819 | 0.630 | 0.651 | 0.819 |
| Window handle | 0.667 | 0.468 | 0.601 | 0.667 |

**3. Chain: fail,** as the prediction feared. The rule (3 of the 10
candidate patches among a touch's 10 most changed) caught 27 touches
for the drawer, 48% of them presses of its button (bar 80%), and 8% of
that button's 155 presses. For the window it caught 1 touch of 162
presses. The handle and the arm outrank the button in the change map.

**Diagnostics after the run, not judged** (`chain_diag.py`,
`chain_state.py`).
- The change signal read at the candidate patches ranks presses above
  other touches (area under the ROC curve 0.81 drawer, 0.94–0.95
  window), but no threshold separates them cleanly.
- Read as a change of state: the candidate patches' before-features
  over all touches are split in two by 2-means. The two states match
  the true lock in 91–92% of touches. A touch sets the condition if
  the state before the next touch differs. That catches 94–96% of the
  presses, and 98–100% of the touches caught lie in one place, the
  button's. Only 47–49% of them are presses. A likely reason, not
  checked: just before a touch at the button the arm covers it, so the
  state there is misread.

**Also seen:** at the two button places the contrast is large (14 and
69 times) and lies on the arm, as in the gate. These places were
outside the criteria.

## 8. Decision

**Revise.** Kept: contrast over remembered touches finds each lock's
button and predicts as well as the true lock, with no training,
outlines or labels (criteria 1 and 2, on fresh episodes). Revised: the
chain is read as a change of the condition's state between touches,
not from the ranking of the change map; the diagnostic supports it
(the right place in 98–100%), and its precision needs the arm set
aside. That and the arm appearing as a condition at the buttons both
point at a model of the self (card 117). Not tested: other numbers of
places or candidate patches; graded outcomes such as where the cube
lands.
