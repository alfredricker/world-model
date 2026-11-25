---
id: "145"
title: a state that changes only when the scene does — one learned vector that must explain every frame given the arm's pose, and may change only through gates that cost to open
rung: 2
serves: [P7, P4, P2, P1]
status: done
verdict: fail
arch_version: og4
date: 2026-11-17
---

# 145: a state that changes only when the scene does

Roadmap step 2 (event segmentation), rung 2, which is still open.
First card of a week on one ability (the user, 2026-11-17: "focus on
fundamental abilities rather than specifics"): a learned state of the
scene, with events as the moments it changes.

og4 has no such state. It has clusters of where the hand made
contact, ten image patches per cluster, and one trained position
read-out for the cube; it reads them only while the arm rests out of
the way, and a contact sensor tells it when an event starts and ends.
Three things wait on a real state: knowing when an effect ended (a
drawer slides on after the sensor reports release), believing in what
is out of view, and acting under feedback while the arm is in the
picture.

Earlier attempts and what each lacked: slots that reconstruct the
frame kept the background and missed small objects (cards 116–116.1);
slots trained on the next frame with a cost on how much each token
changes moved the cube into the arm's token, because the arm changes
every step and hiding the cube there was free
([card 124](../124-tokens-from-consequences/card.md)). Each ran for 30
minutes at most.

Biological counterpart: cortical activity holds a stable pattern
within an event and shifts at its boundary (Baldassano et al. 2017;
Zacks et al. 2007), and the brain predicts the sensations its own
movement causes from a copy of the motor command and sets them aside
(Blakemore, Wolpert and Frith 1998).

## 1. Question

If one vector must explain every frame's features given the arm's own
pose, and may change only through gates that cost to open, does it
become a state of the scene: unchanged while only the arm moves,
changed exactly when something in the scene changes, with each scene
variable in dimensions of its own, readable at every step? Rung 2.
P7 (units and events without labels), P4 (keep what changes
outcomes), P2 (what is believed now, including behind the arm), P1.

## 2. What changes

One component, new: a state layer between the frozen encoder and
everything that reads the scene. Nothing in og4's planner or skills
changes in this card; the state is judged by itself first.

```
og4:   frame at rest → patches of each place, a position read-out; events from the contact sensor
145:   every model step t (5 frames):
       x_t   frozen patch features (1,024 patches, each cut to 64 principal components)
       p_t   the arm's own pose: joint angles, gripper position, yaw and opening (no contact sensor)
       z_t   the state, 32 numbers
       candidate, gate = encoder(x_t, p_t, z_{t-1})        attention over the patches
       z_t   = z_{t-1} + gate × (candidate − z_{t-1})      gate = max(0, tanh(·)) per dimension
       x̂_t  = decoder(z_t, p_t)                            every patch from the state and the pose
       loss  = squared error of x̂_t  +  λ × the number of gates open
```

- **The self is given, not found.** The decoder draws the arm from the
  pose, so the state has no reason to hold the arm and the arm's
  motion needs no gate. This is what card 124 lacked.
- **The gate** is GateL0RD's (LITERATURE): its opening is counted, not
  its size, so a variable switches or stays.
- **Variables are read afterwards**, not built in: dimensions whose
  gates open together are one variable. No slot count is declared.
- **Events** are the steps where a gate opens.

Arms, same data and budget: (A) as above; (B) λ = 0, no cost on
change; (C) the decoder not given the pose.

Not in this card: predicting what an action will change before it is
seen (the next card: a transition on this state, whose gates' causes
are conditions); the cube shut in the drawer for hundreds of steps;
any use in the agent.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The agent's own pose is given | holds (C1) | holds | no body: the state explains the whole frame | holds | general (C1) |
| The scene changes in few variables per step, the self aside | holds for the agent's acts; nightfall changes all | the same | mostly | holds | general, with scene-wide change a declared exception |
| A patch is a place: the decoder draws each patch of one fixed view | no: the view moves, so the camera's pose joins p_t | view follows the agent | holds | fixed cameras only | declared; the pose input is the route to a moving view |
| A frozen self-supervised encoder gives the features to explain | holds | weak on pixel art | weak | holds | adopted (card 114.1) |

## 3. Dependencies

- Frozen DINOv2 patches, blocks 2 and 12, at 32 × 32 (cards 119, 122).
- GateL0RD (`2110_15949`) and its use for events (`2206_02042`,
  THICK), in LITERATURE. Explaining frozen features, not pixels:
  DINO-WM. A recurrent state under a reconstruction loss with an L0
  gate, on image features with the pose given to the decoder, is not
  an established method in a setting like this; that is what the card
  tests.
- The questions asked of a state, and their cells: card 141.

## 4. Data check

Done 2026-11-17 (`tools/card145/check_*.py`, `runs/145/`), from
stored truth; 450 training episodes, 90,000 model steps.

| Question | Answer |
|---|---|
| In how many steps does no scene variable change / one / two | 65.3% / 33.3% / 1.4% (the cube with the drawer); never three |
| How often each changes | cube 22.8% of steps (stretches of 9 steps: it is carried), drawer 4.5%, window 5.3%, each lock 1.7% |
| How long a variable rests between changes | 16 to 28 steps (median) |
| How often the gripper moves | 99.7% of steps |
| Is the contact sensor a good event signal | overall yes (F1 0.82 against "something changes this step"); but it covers only 61% of the drawer's steps of change and 47% of the window's |
| Do 64 components per patch keep the scene | yes: card 141's labelled probe reads the cube to 0.19 cm (full features 0.18), slides to 0.11–0.13 cm (0.12), locks 98% (96–98%) |

So the premise holds in the data: the arm always moves and the scene
rarely does. A state that follows the scene exactly needs a change in
at most 35% of steps.

Training set: 450 episodes of the state dataset rendered at 256
pixels, every fifth frame, encoded and cut to 64 components (about
12 GB, written with pauses); 100 validation episodes for scoring.

## 5. Feasibility gate

Each step capped and under 10 minutes.

1. **The frame can be explained from a small state.** The decoder
   alone, trained on the true state (cube pose, two openings, two
   locks) and the pose. On the patches that change across an event it
   must remove at least half of the error left by a decoder given the
   pose only (the trivial baseline).
2. **The questions can be answered at any step.** Card 141's labelled
   probe, trained and tested on frames from every step, arm in view,
   scores at least 0.95 (area under the curve) on "does this variable
   differ" in the cells of criterion 2. A cell where it does not is
   reported and its bar set from the probe.
3. **Baselines for events:** the contact sensor (above) and plain
   change of features, scored as criterion 1.
4. **Pilot:** 10-minute fits at three values of λ, for throughput and
   to find the range where gates neither all open nor all shut. Not
   judged.

Result of the gate (2026-11-17): passed, with the pilot inconclusive.

1. **A small state explains the frame: passed.** Two decoders, 9
   minutes each. On the 20 patches an event changes, the decoder
   given the true state leaves this share of the pose-only decoder's
   error: cube 0.49, drawer 0.23, window 0.44, locks 0.46 and 0.35
   (bar 0.5; the cube passes narrowly). Over whole frames: 0.19
   against 0.57, where 1 is always answering the mean frame.
2. **The questions can be answered at any step: passed.** The
   labelled probe scores 0.994 to 1.0 in every cell (cube 0.994 over
   3,312 same and 3,012 different pairs). Random pairs from different
   episodes held only 18 with the cube in the same spot, so for the
   cube each step is also paired with a step of another episode
   where the cube lies within 1 cm. At any step the probe reads the
   cube to 0.38 cm (0.18 at rest), and is over 2 cm off in 10% of
   steps.
3. **Baselines for events, with one step of slack.** Contact sensor:
   precision 0.93, recall 0.80 (drawer) and 0.81 (window). Section 6
   quoted the sensor without slack (0.79, 0.59, 0.46) beside bars
   that allow slack, so criterion 1's precision bar of 0.8 is below
   what the sensor already does. The bars stay as approved; the
   result is reported against the sensor with slack as well. Plain
   change of features is no better than always saying "change"
   (precision 0.53 against 0.55).
4. **Pilot: inconclusive.** Training runs at 5.5 steps a second, so
   8.5 minutes shared between two fits gave 1,400 steps each, where
   the model has only learnt to draw the arm (error 0.51). Gates
   were open in 73% of steps at λ = 0.003. One dimension already
   follows the drawer (opening read to 0.28 cm from 100 labels).
   λ is therefore set in the first pass, starting at 0.003.

Two details of the build, both inside section 2's form: the encoder
is also given the difference between the frame and what the old
state predicts for it, and the reconstruction is scored on a random
quarter of the patches per step (memory).

## 6. Success criteria and prediction

Arm A, validation episodes, every step, two seeds; the mean.

1. **Events:** the steps with a gate open are the steps where the
   scene changes. Precision at least 0.8, and recall at least 0.9 for
   each of the drawer's and the window's steps of change (contact
   sensor: precision 0.79, recall 0.61 and 0.47), with one step of
   slack.
2. **Values:** for two steps from different episodes, the distance
   between the states, within the dimensions of one variable, says
   whether that variable differs: area under the curve at least 0.95
   for the cube on the table (under 1 cm against over 3 cm), each
   slide (under 1 cm against 4 to 10 cm) and each lock. Dimensions
   are assigned to variables on training episodes.
3. **Variables:** each of the five variables has dimensions of its
   own. At least 80% of a variable's steps of change open one of its
   dimensions, and at least 80% of those dimensions' openings fall on
   its steps of change.

Reported without a bar: arms B and C; the smaller cells (slides
differing by 1 to 4 cm, the cube in the open drawer); a value read
from the state with 100 labels; the state while the arm hides the
cube; when a slide's gates close against when it truly stops.

**Prediction.** Criterion 1 is the likeliest: a slide moving changes
many patches. 2 passes for slides and locks only if the loss does not
drown small regions; the buttons are 3 patches of 1,024, and a
weighting toward patches that change is the first repair if it does.
3 is the uncertain one: nothing forces one variable per dimension
except the count of open gates, and the cube, carried for 9 steps at
a time, may be held as "in the hand" (two openings) or tracked (nine).
Either is a finding. Arm C should need a gate at almost every step.

**Budget.** Building the data: about 10 minutes in three capped
steps. Gate: five runs under 10 minutes. First pass: 30 minutes per
arm, launched here once approved. If arm A's gates behave, the long
runs (2 hours, two seeds) are handed to the user as commands.

## 7. Result

### First pass (2026-11-17), one seed, 30 minutes a run; not the verdict

`tools/card145/` (`train.py`, `run_arm.sh`, `evaluate.py`,
`diag_state.py`, `diag_weight.py`, `diag_size.py`), `runs/145/`.
100 validation episodes, every step.

| Run | Gates open per step | What a read-out trained on the state finds (20,100 labelled states; a diagnostic) |
|---|---|---|
| A, λ = 0.003 from the start | 0.15 | drawer 0.11 cm; window, locks, cube: nothing |
| B, λ = 0 | 15.5 | drawer 0.06 cm, drawer's lock 99.9%, window 1.5 cm, window's lock 55%, cube nothing |
| A after B: B's weights, λ raised to 0.003 over 10 minutes | 1.25 | drawer 0.07 cm, drawer's lock 100%, window 0.38 cm, window's lock 77%, cube nothing |

For comparison: always answering the mean is 6.4 cm (drawer), 9.1 cm
(window) and 9.9 cm (cube); the pose alone guesses a lock 56% of the
time. Reconstruction error (1 = the mean frame; the pose alone 0.57;
the true-state decoder after 9 minutes 0.19): 0.28, 0.26, 0.13.

The third run against the criteria (60 minutes of training in all):

| Criterion | Measure | Third run | Bar | Contact sensor |
|---|---|---|---|---|
| 1 Events | precision of "a gate is open" | 0.61 | 0.8 | 0.93 |
| | recall, drawer / window | 1.00 / 0.97 | 0.9 | 0.80 / 0.81 |
| 2 Values (area under the curve) | drawer, differing by 4 to 10 cm | 0.999 | 0.95 | |
| | window, differing by 4 to 10 cm | 0.958 | 0.95 | |
| | drawer's lock / window's lock | 0.991 / none | 0.95 | |
| | cube on the table | 0.49 | 0.95 | |
| 3 Variables | recall of own dimensions: drawer, window, drawer's lock | 1.00, 0.95, 1.00 | 0.8 | |
| | precision of own dimensions | 0.27, 0.35, 0.31 | 0.8 | |

What the first pass shows:

- **The state learns the scene in the order of how much of the
  picture each variable is.** Measured with the gate's true-state
  decoder, swapping a variable's value costs this share of the
  pose-only error: drawer 0.83, window 0.45, cube 0.12, each lock
  0.03. The drawer came first, then its lock (painted on its handle),
  then the window; the window's lock is arriving; the cube is not
  there after 60 minutes.
- **A cost on change from the start stops the learning.** A closed
  gate passes no gradient, so what is not yet learnt when the gates
  shut is never learnt (run A: the drawer only). Learning first with
  no cost and raising it afterwards works.
- **The values are steady and jump at events; the gates are not yet
  clean.** Where a variable's dimensions open on its own steps of
  change the state moves 2.3 (drawer), 1.6 (window), 0.8 (lock); at
  their other openings 0.04, 0.05, 0.01. The size of the state's
  change marks the drawer's steps of change with an area under the
  curve of 0.995, the lock's with 1.0, the window's with 0.83. The
  openings were still falling when the run ended (1.5 to 1.25 per
  step over its last 8 minutes).
- **One dimension follows "the cube in the open drawer"** (0.96 in
  that cell), where the cube is large in the picture and next to a
  variable already learnt.
- **Training settings found on the way** (inside section 2's form):
  the state is carried through whole episodes in chunks of 64 steps,
  so a gate is free only at an episode's first step; with segments of
  32 steps each starting free, the model was handed the scene every
  segment.
- A GPU driver fault (kernel Xid 13) stopped one run after 5 minutes;
  runs now save every 5 minutes and resume.
- Evaluation detail: a group of dimensions is named after a variable
  if its openings fall on that variable's steps 1.5 times more often
  than chance.

Arm C (no pose) has not been run. The long runs are the user's to
launch (section 6's budget):

```
tools/card145/run_arm.sh A_long_s0 120 --lam 0.003 --carry 1 --T 64 --B 8 --free 40 --ramp 20 --seed 0
tools/card145/run_arm.sh A_long_s1 120 --lam 0.003 --carry 1 --T 64 --B 8 --free 40 --ramp 20 --seed 1
tools/card145/run_arm.sh C_s0 60 --lam 0.003 --carry 1 --T 64 --B 8 --free 20 --ramp 10 --nopose 1
```

Each is capped (16 GB, 8 cores, its time plus 5 minutes), resumes
after a fault, and writes `runs/145/<name>.pt`.

### Long run, seed 0 (2026-11-17), 120 minutes, 28,346 steps; one seed, not the verdict

Validation episodes, every step. "60 min" is the first pass's third run.

| Measure | Bar | 60 min | 120 min |
|---|---|---|---|
| Reconstruction error (1 = the mean frame) | none | 0.132 | 0.100 |
| Steps with a gate open (a true change: 35.1%) | none | 43.6% | 44.2% |
| Events, precision (contact sensor 0.93) | 0.8 | 0.61 | 0.62 |
| Events, recall: drawer / window | 0.9 | 1.00 / 0.97 | 1.00 / 0.99 |
| Events, recall: locks / cube | none | 1.00, 0.77 / 0.45 | 1.00, 1.00 / 0.50 |
| Values, AUROC: drawer / window (4–10 cm apart) | 0.95 | 0.999 / 0.958 | 0.996 / 0.954 |
| Values, AUROC: drawer's lock / window's lock | 0.95 | 0.991 / no dimensions | 0.998 / 0.993 |
| Values, AUROC: cube on the table | 0.95 | 0.49 | 0.50 |
| Own dimensions, recall: drawer / window / locks | 0.8 | 1.00 / 0.95 / 1.00, none | 1.00 / 0.99 / 1.00, 1.00 |
| Own dimensions, precision: same order | 0.8 | 0.27 / 0.35 / 0.31, none | 0.38 / 0.33 / 0.49, 0.33 |
| Network probe on the state: drawer / window, cm | none | 0.07 / 0.38 | 0.16 / 0.36 |
| Probe: locks right / cube, cm (guessing 9.9) | none | 100%, 77% / 8 | 99.8%, 99.6% / 7.7 |

- **Twice the training brought in the window's lock** (77% → 99.6%
  read from the state; a dimension of its own, 22), the smallest
  variable in the picture. Four of five variables are now in the state,
  each in its own dimensions.
- **It did not bring in the cube** (AUROC 0.50, probe 7.7 cm against
  5.9–7.8 from the arm's pose alone), and **did not reduce the false
  openings** (precision 0.61 → 0.62; a gate is open in 36.5% of steps
  where only the arm moves). More training of this form is unlikely to
  fix either.
- **The false openings stay tiny.** The state moves by 1.6–3.7 at a
  variable's own steps of change against 0.03–0.08 at its other
  openings. Reading an event as "the state moved by more than a
  threshold" (set on 50 validation episodes, scored on the other 50)
  gives precision 0.98 with recall 0.98 for the drawer and 0.90 for the
  window, against the gates' 0.63 and the sensor's 0.93. One threshold
  on the whole state misses the locks (recall 0.04 and 0.54), whose
  jumps are smaller. This is a diagnostic, not criterion 1's measure.

Seed 1 and arm C were not run (see the decision).

**Verdict: fail, on one seed of the two planned.** Criterion 1 fails on
precision (0.62 against 0.8), criterion 2 on the cube (0.50 against
0.95, which is chance), criterion 3 on the precision of each variable's
own dimensions (0.33 to 0.49 against 0.8). The second seed was planned to
confirm a pass; it cannot turn these into one, since twice the training
moved none of the three.

## 8. Decision

**Stop** (the user, 2026-11-24). The fault is in the training signal, not
in the amount of training. The state was trained to let a decoder redraw
every patch, and that error is summed over patches, so a variable is
learned in the order of how much of the picture it fills: swapping the
drawer changes 0.83 of the error the pose leaves, the window 0.45, the
cube 0.12. The cube, which most tasks are about, was never learned.
GOAL's C6 already says frames are evidence and not the thing predicted.
The user's rule from this card: no decoder as a representation's training
signal; a representation is trained by consequences, that is by what the
agent's actions change ([LESSONS](../../LESSONS.md)).

Kept as findings: a cost on change from the start stops learning, so
learn first and raise the cost afterwards; the size of the state's change
marks the drawer's and window's events better than the gates do (precision
0.98 against 0.63).
