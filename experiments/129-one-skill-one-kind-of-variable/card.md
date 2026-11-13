---
id: "129"
title: one skill and one kind of variable — every touch a shifted replay aimed by one read-out of where a hand can close
rung: 3
serves: [P12, P4, P21, P1]
status: done
verdict: fail
arch_version: og2
date: 2026-11-11
---

# 129: one skill and one kind of variable

Roadmap step 3 (skills), rung 3. Version og2
([card 128](../128-a-touch-moved-to-where-the-cube-is/card.md)) added
the cube as a special case: a second skill path, a second kind of
variable, a read-out that assumes one movable object, and a line that
says "move the cube first". The user (2026-11-11): "we're trying to
architect for edge cases where our solutions should be general
principles". This card replaces the special case with one mechanism
before tasks 4 and 5 are tried.

Biological counterpart: an affordance map. Parietal and premotor areas
represent, for the scene in view, the places where a grasp is possible
(Cisek 2007; Gibson 1979); a fixed handle and a loose cube are aimed
at by the same system.

## 1. Question

Can one read-out, one skill and one kind of variable do what og2 does
with two of each, at no loss on tasks 1–3? P12, P4, P21, P1.

## 2. What changes

One component: how a touch is aimed and how the scene is compared with
the goal. Conditions (cards 125–125.2) and the planner's ordering by
conditions are unchanged.

```
og2:  buttons, handles: variable = a fixed set of patches, compared by before/after match;
                        skill = exact replay
      cube:             variable = one position from a read-out trained on cube touches only;
                        skill = shifted replay; "move the cube first"
129:  every touch:      targets = the peaks of one read-out, "where could a hand close with an
                        effect", trained on every touch across which a change was seen;
                        variable = a target's position, now and in the goal frame;
                        skill = shifted replay: the shift is (the target nearest the remembered
                        contact point − that contact point); near zero for what never moves;
                        no target near it = the touch cannot be done now
```

- **Read-out.** og2's network (per patch a score and an offset), with
  a loss that allows several answers: the softmax weights are a
  mixture over patches, and the loss is the likelihood of the contact
  point under that mixture. A frame with a cube, two buttons and two
  handles then gets a peak on each, since any of them may be touched
  next. No object is named anywhere.
- **Differs from the goal:** a target now with no target in the goal
  frame within the threshold (og2's, from the lengths of remembered
  moves), and the reverse.
- **Recall:** the remembered touch whose contact point is nearest the
  unmatched target now and whose release point is nearest the
  unmatched target in the goal.
- **Removed:** og1's effect patches and its before/after matching,
  og2's cube-only read-out and cube path, the "cube first" line.
- **Kept and still separate:** a condition is a state read from
  appearance (a lock is not a position). That is a real second kind of
  value, not a special case of the first.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| What a touch changes can be summed up by where the effector closed and opened | holds for digging, placing | holds | a click has one point; no release | holds for grasps and pushes | exact for the cube (0.4 cm); loose for handles, see section 4 |
| A fixed camera, so a patch maps to a position | no | no | yes | fixed cameras only | declared since card 119 |
| Targets are few enough to be separate peaks | many alike blocks; peaks merge | the same | holds | cluttered scenes: no | five targets |
| The goal is given as a frame | no | no | no | sometimes | OGBench's goal image |

## 3. Dependencies

- Cards 127 and 128 (planner, skill, read-out), 125–125.2 (conditions).
- Mixture density outputs for many-answer regression: Bishop 1994.
  Affordance maps learned from a robot's own grasps: Zeng et al. 2018
  (per-pixel grasp success). Using the peaks as the planner's
  variables is this card's test.

## 4. Data check

Memory set m400 has 1,453 touches across which a change was seen;
three sets give about 4,400, one frame each (about 7 GB of features,
written in paced chunks, held in memory for training).

Checked 2026-11-11: does the gripper's contact point follow a slide's
position? Only loosely.

| | Slope of gripper position on slide position (1 = follows exactly) | Scatter around it, median |
|---|---|---|
| Drawer, at contact / at release | 0.40 / 0.68 | 2.6 cm / 1.6 cm |
| Window, at contact / at release | 0.12 / 0.46 | 1.8 cm / 2.4 cm |
| Cube, at contact (card 128) | 1.0 | 0.4 cm |

So a handle's position is a blurred variable: the gripper meets a
slide at different points and often before it starts moving. An open
and a closed slide (16–20 cm apart) should still be told apart; small
differences will not be. This is the card's main risk, and og1's
appearance match is the fallback it would return to.

## 5. Feasibility gate

Step 1 run 2026-11-11 (`tools/card129/readout.py`, `readout_q.py`;
each run under 3 minutes, capped). 4,067 touches with a seen change
from three sets; trained on episodes 0–79 of each (3,267), scored on
episodes 80–99 (800 frames). Marks set before the run: a target within
1 cm of the cube in at least 90% of frames; within 3 cm of where the
gripper then closed in at least 90% for buttons and for handles.

| Read-out | Cube within 1 cm (bar 90%) | Buttons within 3 cm (bar 90%) | Handles within 3 cm (bar 90%) |
|---|---|---|---|
| As designed: one mixture over all targets | 40% | 84% | 78% |
| Asked about one place at a time | 72% (own place); 34–42% when the cube is in another place's region | 99.6% | 87.5% |
| Asked about one group of places | 72% in any frame, median 0.65 cm | 99.3% | 65.5% |
| og2's cube-only read-out (reference) | about 90%, median 0.49 cm | not covered | not covered |
| The place's or group's mean contact point (baseline), median | 8.5 cm | 0.5 cm | 1.5 cm (place), 3.0 cm (group) |

**Not passed** by any of the three. Step 2 (the skill aimed by the
read-out) was not run.
- **The mixture** fails because the targets share one softmax: the
  target about to be touched takes the weight (the arm's heading gives
  it away) and the others fall under the peak threshold.
- **Asking per place** removes the competition, but the table's places
  are arbitrary cuts of where the cube can be.
- **Groups** (found with no truth): two places are joined when
  remembered moves run both ways between them. Memory gives exactly
  the three table places (the cube), the two drawer places, the two
  window places, and each button place alone. One more link runs one
  way only: table to open drawer, 7–14% of table touches, none back.
- **Handles are the real failure,** as section 4 feared: where the
  gripper meets a slide is not settled by the frame (the read-out
  beats the group's mean by 0.9 cm of 3.0). A position cannot be the
  slide's variable.

## 6. Success criteria and prediction

Agreed with the user 2026-11-11 ("approved"), before the gate was run:
1. Benchmark tasks 1, 2 and 3, the same 50 episodes each: each within
   10 points of og2 (84%, 96%, 86%). A loss inside that margin is
   accepted for the general form (AGENTS, exception for general
   forms) and goes into the known limits.
2. Random cube problems at least 88% (og2: 98%).
3. The code has one skill path and one comparison with the goal; no
   line names the cube, and none fixes the order of variables.

**Prediction.** The cube and the buttons carry over. The slides are
the risk (section 4): a window left a few centimetres open, already
og1's commonest failure, may get worse.

**Not in this card:** tasks 4 and 5. They follow, with no
task-specific rule allowed: a target out of view keeps its last
position, and whether it can be seen is a condition found by contrast.

**Budget.** Rendering and encoding about 4,400 frames (5 minutes);
training about 5 minutes with features in memory; live runs under 2
minutes each.

## 7. Result

The gate was not passed, so the main run was not made and criteria
1–3 are not judged.

## 8. Decision

**Revise.** "One kind of variable, a position" is wrong for this
scene: it fits the cube (0.65 cm), is trivially right for a button
(it never moves), and fails for a slide, whose state shows in
appearance and not in where the hand meets it. Kept from the gate:
groups of places from two-way moves, which replace every mention of
"the cube" and merge og1's doubled slide places with no truth; and one
read-out asked per group, which covers the cube and the buttons in one
network. Proposed revision (129.1, for the user to approve): every
group has both readings, appearance over its changed patches and an
aim from the read-out, and each is used where memory shows it
informative (patches changed at least 5 times the general level; the
aim beating the group's mean contact point), with one shifted skill
whose shift is zero where the aim is not informative. Not tested: more
training or data for the cube's 72%; the skill aimed by this read-out.
