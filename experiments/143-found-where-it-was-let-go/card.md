---
id: "143"
title: found where it was let go — the read-out that finds an object is also trained on where the hand released it, so the object is found in places it was put but never picked up from
rung: 3
serves: [P1, P7, P4]
status: done
verdict: fail
arch_version: og4
date: 2026-11-15
---

# 143: found where it was let go

Roadmap step 2 (the perceptual side), rung 3. Third card of the
representation week, on the "finding" step that
[card 142](../142-find-then-place/card.md) set aside (its check 3).

og4 reads where an object is with a read-out over frozen patch
features: a score per patch says which patches hold the object, and
their weighted location gives a position. Its only training signal
is the contact point: a frame from before a touch, labelled with
where the gripper then closed. So the read-out has seen the object
only in places it was later picked up from.

With the cube in the open drawer the reading is 27.7 cm off, while a
probe trained on the true position reads it to 0.39 cm from the same
features ([card 141](../141-representation-test-bed/card.md)). The
data check below gives the reason: in 400 episodes of recorded play
the cube is put into the drawer 119 times and taken out once. The
read-out has 2 training frames with the cube in the open drawer.

Biological counterpart: vision for reaching is calibrated by the
animal's own movements, not by being shown answers (Held and Hein
1963). Putting a thing down teaches where it is as surely as picking
it up does: the hand knew where it was at both moments.

## 1. Question

If the read-out is also trained on frames from just after a touch,
labelled with the point where the hand released the object, does the
agent find the cube in the open drawer, at no cost on the table, and
can it then put the cube there when asked? Rung 3. P1 (grounded
meaning: an object is where the hand can meet it), P7 (learned
units), P4 (the reading holds when the surroundings change).

## 2. What changes

One component: the training set of the aim read-out. The network,
the features, the loss and the schedule are og4's.

```
og4:  frame before a touch (hand at least 15 cm away)  →  the contact point of that touch
143:  the same, plus
      frame after a touch (hand at least 15 cm away, before the next touch)
                                                       →  the release point of that touch
      only for touches at a group whose aim reading is switched on (og4's own test on
      memory, card 129.1), asked with that group, and only if a change was seen across
      the touch
```

The restriction matters and is the agent's own, with no object
named. A release point labels the object only if the object was in
the hand: when the drawer carried the cube, the hand's release point
is 9.6 cm from the cube (data check). That case is a relation
between two variables and is not this card.

Arms: og4 (baseline); og4 plus release labels.

Not in this card, named so that nothing is hidden:
- **Slides.** The same idea fails for them as the data stand: the
  contact sensor stops reporting a median of 7 frames (drawer) and
  25 frames (window) before the slide stops, so the sensed release
  point reads the opening no better than its mean (6.4 and 8.5 cm
  against 7.4 and 8.6). With the true moment the slide stops, the
  hand reads it to 1.3 and 0.3 cm. The next card is therefore "when
  does a hold end", from the agent's own signals.
- **The cube hidden in the closed drawer** ("not in view" as a
  value), and a label kept beyond the next touch. Both need belief.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The agent knows where its effector is at release | pose given; the block is placed at the aimed cell | holds (the faced cell) | no body: the acted-on cell instead | holds (proprioception) | general |
| An object stays where it was released until the next touch | no (items fall, mobs move) | mostly | holds | no in general (rolling, falling) | declared; the frame is drawn soon after release, and a wrong label is one the read-out's own held-out error exposes |
| A touch's start and end are sensed | no contact sense; the action itself brackets | the action brackets | the action brackets | holds with touch or force sensing | declared (card 120); for slides it fails here, see above |
| One fixed camera | no | no | yes | often no | declared, as og4 |

## 3. Dependencies

- The read-out, its groups and the test that switches the aim
  reading on: cards 128 to 129.2 (`tools/card129/`,
  `tools/card129.1/`).
- The test bed: card 141 (`tools/card141/`); only its cube cells are
  used, which need neither of the two repairs card 142 lists.
- The planner and cube skill for criterion 3: og4 (card 130).
  The put-away problems are new and are written for this card from
  `tools/card128/plan_cube.py`.
- No new method. Card 142's owed LITERATURE entry (spatial soft
  arg-max) covers the read-out's form.

## 4. Data check

Done 2026-11-15 from stored truth and proprioception, no rendering
(`tools/card143/check.py`, `check_hold.py`, seconds each;
`runs/143/`). Which object a touch moved was taken from the truth
here; the run uses the agent's own groups.

| Where the cube is in the labelled frame | Contact labels (og4), 400 episodes | Release labels, hand began on the cube, 300 episodes | Release point against the true cube |
|---|---|---|---|
| On the table | 1,798 | 1,271 | 0.32 cm median, 97% within 1 cm |
| In the open drawer | 2 | 88 | 0.43 cm median, 81% within 1 cm |

In 200 held-out episodes the release point is as good (0.34 and
0.32 cm). The read-out trains on 240 episodes, so it gains about 70
frames with the cube in the open drawer. That is few beside the
table's thousand, and the gate asks whether it is enough.

Release labels to exclude, and why the rule in section 2 excludes
them: 113 touches moved the cube into view in the open drawer
without starting on it (the drawer moved in 103 of them); their
release point is 9.6 cm from the cube.

## 5. Feasibility gate

- **Upper bound:** the 143 arm with the true cube position as the
  target on exactly the added frames. It says whether about 70
  put-ins are enough for this read-out. Bar: the cube in the open
  drawer read to a median of at most 1.0 cm on the bed's 985 test
  frames (the probe, with 1,048 labelled frames: 0.39 cm).
- **Trivial baseline:** og4, 27.7 cm; it must stay above 10 cm.
- **The agent's own selection:** at least 50 added frames have the
  cube in the open drawer, and at least 85% of all added labels lie
  within 1 cm of the true cube.
- **Live baseline:** og4 on the put-away problems of criterion 3.

Result of the gate (2026-11-16): passed, after the fallback the
prediction named.

- **The agent's own selection: passed.** 1,101 releases selected, of
  which 96.4% lie within 1 cm of the true cube; 84 have the cube in
  the open drawer (66 in the training episodes), read to 0.43 cm.
- **Upper bound: missed with one frame per release (1.80 cm), passed
  with up to four (0.93 cm; bar 1.0).** Four frames are drawn from
  the same interval after each release, so they add arm poses, not
  cube positions (`build.py --per 4`, 4,394 frames). The main run
  uses four.
- **Trivial baseline:** og4, 27.7 cm.
- **Live baseline:** og4 puts the cube in the drawer in 0 of 50
  put-away problems (it reads the goal's cube 27.4 cm off).
- A warning from the upper bound: even trained on true positions, the
  open-drawer pair cell reaches 92.8, under criterion 1's bar of 95.

## 6. Success criteria and prediction

Mean of 3 seeds of the read-out. Proposed, to be fixed with the user.

1. **Found where it was put:** on the bed's test frames with the
   cube in the open drawer, median error at most 1.0 cm (og4 27.7;
   probe 0.39), and the bed's "cube, one in the open drawer" cell at
   least 95 (og4 70.6; probe 99.0; 50 is chance).
2. **No cost on the table:** median error at most 0.75 cm (og4
   0.60), and the "cube, both on the table" cell at least 98.5 (og4
   99.4).
3. **Used in behaviour (GOAL C4):** 50 self-made put-away problems
   (cube at random on the table, drawer open at the start and in the
   goal, the goal frame showing the cube in the drawer): the cube
   ends inside the drawer in at least 70% (card 131's bar; og4
   expected near 0%, measured at the gate). One seed.

Reported without a bar: the "hidden cube" cell; the error by distance
from the nearest release label; the read-out's largest score in each
cell (card 142 found it halved in the open drawer).

**Prediction.** 1 passes if the upper bound does; the doubt is 70
frames against a thousand, and the answer if it fails is to draw
several frames per release, not a new mechanism. 2 passes. 3 is the
uncertain one: reading the goal is fixed, but the remembered put-in
is replayed with an offset, and a cube dropped on the drawer's rim
fails.

**Budget.** About 1,400 new frames to render and encode (a few
minutes). Four fits of about 3 minutes (og4 exists; upper bound; 3
seeds). Two live runs of 50 problems, about 3 minutes each. Every
run capped and under 10 minutes.

## 7. Result

Run 2026-11-16, `tools/card143/` (`build.py`, `fit.py`, `score.py`,
`put.py`, `centre.py`; every run capped and under 10 minutes),
numbers in [results.json](results.json). The read-out is the mean of
3 seeds; the bed has 985 test frames with the cube in the open
drawer and 4,793 with it on the table.

| Measure | og4 | og4 + release labels | Upper bound (true positions on the same frames) | Bar |
|---|---|---|---|---|
| Cube in the open drawer, median error | 27.7 cm | 1.70 cm (1.41, 1.51, 2.17) | 0.93 cm | at most 1.0 |
| Pair cell "one in the open drawer" | 70.6 | 89.4 (91.9, 92.9, 83.3) | 92.8 | at least 95 |
| Cube on the table, median error | 0.60 cm | 0.45 cm | 0.28 cm | at most 0.75 |
| Pair cell "both on the table" | 99.4 | 99.3 | 97.7 | at least 98.5 |
| Put-away problems, cube ends in the drawer (50, one seed) | 0% | 68% | | at least 70% |

1. **Found where it was put: fail.** The error falls from 27.7 to
   1.70 cm and the pair cell rises from 70.6 to 89.4, but the bars
   were 1.0 cm and 95.
2. **No cost on the table: pass.** The table reading improves (0.45
   against 0.60 cm): each release is a second label for a table
   position.
3. **Used in behaviour: fail by one problem** (34 of 50; the bar
   needs 35).

What the run showed beyond the criteria:

- **The cube is now found; what is left is placing it.** Within 1 cm
  of a release label the error in the drawer is 0.81 to 0.98 cm; 1
  to 3 cm away, 2.3 to 3.8 cm; over 3 cm away, 7.6 to 15.8 cm. This
  is card 141's fault (a trained map ends where its training ended),
  now inside the drawer, where 66 put-ins are the whole of training.
- **One seed in three finds the cube weakly.** Seed 2's largest score
  in the open drawer is under zero (seeds 0 and 1: 6.1 and 2.7;
  table about 6), and it is the seed at 2.17 cm.
- **og4 remembers no move that ends in the drawer.** It takes a
  move's end from the next time the hand finds the object, and the
  cube is almost never picked out of the drawer. With one addition
  (not declared, so not a criterion): a move the hand never follows
  up ends where the hand let go. Then 100 more moves are remembered
  and the cube ends in the drawer in 90% of problems (84% within
  4 cm of the goal's spot). The same addition with og4's read-out
  gives 0%, so the read-out is what was missing; the moves are what
  uses it.
- **Reading the centre of the weights through a second-order map**
  (card 142's form), fitted on the agent's own 78 labels above table
  height, gives 1.18 cm in the open drawer against the read-out's
  1.41 for the same seed, and 0.88 against 0.46 cm on the table.
- **The agent does not know it is done:** it stops by itself with the
  cube inside in 2 to 3 of 50 problems; the rest end on the
  benchmark's flag or the step limit.

Two things differ from the card as approved, both stated above: four
frames per release instead of one (the gate), and the moves that end
where the hand let go (reported beside criterion 3, not as it).

The run used the Python environment in
`~/.cache/pypoetry/virtualenvs/worldmodel-iVGTe5yi-py3.13`: the
project's `.venv`, rebuilt on 2026-11-12, has no ogbench, mujoco or
scikit-learn.

## 8. Decision

**Revise.** Two of three criteria miss their bars, so the card does
not pass, and no version is adopted from it. The release labels are
kept as the training set for the cards that follow: they are right
to 0.4 cm by the agent's own selection, they cost nothing on the
table, and they take the open drawer from unreadable to a reading
whose remaining error is of a known kind. Next,
[card 142](../142-find-then-place/card.md) on top of them: its
subject is exactly the error that grows with distance from a label.
After it, for the user: moves that end where the hand let go, as a
declared change of their own; and "when does a hold end", for the
slides.
