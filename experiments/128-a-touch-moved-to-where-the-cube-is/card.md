---
id: "128"
title: a touch moved to where the cube is — a remembered grasp shifted by where the frame shows the cube now and in the goal
rung: 3
serves: [P12, P4, P21, P1]
status: done
verdict: pass
arch_version: og2
date: 2026-11-11
---

# 128: a touch moved to where the cube is

Roadmap step 3 (skills), rung 3. Version og1
([card 127](../127-plans-from-remembered-touches/card.md)) replays a
remembered touch exactly, which works for buttons and handles because
they never move. The cube does. The user (2026-11-11) agreed the cube
is next: it is the first test of a skill that adapts, not replays.

Biological counterpart: reaching is aimed by sight in hand-centred
coordinates, a map learned from seeing one's own hand act (Held and
Hein 1963; parietal reach areas). The remembered movement is reused
and its end points are moved to where the eyes put the target.

## 1. Question

Can the agent read from a frame where the cube is, in the gripper's
own coordinates, well enough to shift a remembered grasp onto it and a
remembered release onto the goal's position, and so solve the
benchmark's task 3? P12 (chains), P4 (memory), P21, P1.

## 2. What changes

One component: the skill for places that have no fixed variable.

```
og1:  redo touch i: follow its recorded gripper path as it was
128:  redo touch i, shifted: add (where the cube is now − where it was in touch i) up to contact,
      and (where the goal shows it − where touch i left it) from release on, blended in between
      "where the cube is" = a read-out from the frame's patch features, trained on the agent's own
      touches: the scene before a cube touch → where the gripper then closed
```

- **The read-out** is the only trained part: per patch, a score from
  its features; the position is the score-weighted mean of a learned
  position per patch (a soft arg-max). Its target is the gripper's
  position at contact, which the agent knows from proprioception; no
  cube label is used. The input frame is taken while the gripper is
  still at least 15 cm from that position, so the read-out cannot
  answer by finding the arm.
- **Recall:** the remembered move whose start is nearest the cube now
  and whose end is nearest the goal's cube.
- **The planner** is version og1's; the cube is one more variable whose
  value is a position, not one of two states.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A touched object sits where the effector closed | the target is at the view's centre, so the shift is in view coordinates | the facing tile | the clicked cell | holds for grasps | within 1 mm in memory |
| A fixed camera, so a patch maps to a position | no | no | yes | fixed cameras only | declared since card 119 |
| A remembered movement stays valid when shifted | holds for approach paths | holds | holds | within the arm's reach | the table's free area |
| The goal shows the object | no goal frame | no | no | sometimes | task 3 yes; tasks 4 and 5 see section 4 |

## 3. Dependencies

- Card 127 (skill, planner, memory), cards 120 and 122 (touches).
- Key points from soft arg-max over features: Levine et al. 2016
  (spatial softmax). Training it from the agent's own contact position
  is this card's test; it is not an established result.

## 4. Data check

Memory (episodes 400–499) holds 442 touches that moved the cube more
than 5 cm; the gripper is centred on the cube at contact to within
1 mm. Six more sets of 100 episodes have their touches listed and can
be added for the read-out (about 3,000 cube touches).
- **Task 3:** cube 20 cm along the table; unlock and close the window,
  open the drawer.
- **Tasks 4 and 5:** the goal has the cube inside the closed drawer.
  If it cannot be seen in the goal frame (not yet checked), the agent
  has to infer where it went, which nothing here does. They are not
  attempted in this card.

## 5. Feasibility gate

**Part 1, run 2026-11-11** (`tools/card128/gate.py`, capped, under 2
minutes): the shifted skill with the truth supplying the positions.
Live `scene-v0`, 30 trials each; success is the cube within 4 cm of
its goal.

| Arm | Task 3 | Random start and goal on the table |
|---|---|---|
| Replay the nearest remembered move, no shift | 90% | 47% |
| Shifted | 87% | 100% |
| Shifted, gripper turned to the cube's yaw | 100% | 97% |
| Shifted, cube's position wrong by 1 cm | 93% | 77% |
| Shifted, cube's position wrong by 2 cm | 37% | 47% |

The shift works. Task 3's start and goal happen to lie near remembered
moves (3 cm off on average), so replay alone nearly solves it; random
problems show the shift's worth. The read-out must be right to about
1 cm. A patch is roughly 2–3 cm of table, so that needs a position
finer than one patch.

**Part 2, run 2026-11-11** (`tools/card128/readout.py`): the read-out,
trained on 895 cube touches of three sets of 100 episodes (episodes
0–79 of each), tested on the 235 touches of episodes 80–99. Error on
the table, in cm, median (90th percentile):

| | Held-out touches | Live, arm at rest, cube inside the remembered range | Live, cube outside it |
|---|---|---|---|
| Mean position (trivial baseline) | 8.5 (12.2) | | |
| Soft arg-max only | 2.1 (4.0) | 3.4 over both ranges | |
| Soft arg-max with a per-patch offset | 0.49 (0.88) | 0.43 (0.96) | 9.2 (13.8) |
| Gripper at contact against the true cube (the target's own error) | 0.41 (0.77) | | |

Passed inside the range of remembered grasps (y from −9 to 18 cm);
outside it the read-out does not extrapolate. Training took 8 minutes
(slow random reads from disk), close to the 10-minute rule.

## 6. Success criteria and prediction

Agreed with the user 2026-11-11 ("yes"), before part 2 was run.
1. Read-out: median error at most 1 cm on held-out cube touches.
2. Random table problems, positions read from the frames: at least
   70% (truth: 97–100%; replay: 47%).
3. Benchmark task 3, run once, full planner: at least 60%.

**Prediction.** 1 is the risk: sub-patch precision from 8-pixel
patches.

## 7. Result

Run 2026-11-11 (`tools/card128/plan_cube.py`: card 127's planner with
the cube as one more variable; each run under 2 minutes). Decisions
made while building, with no truth: a remembered move's ends are the
gripper's positions at contact and release; a move is real, and the
cube differs from the goal, above Otsu's threshold on the moves'
lengths (2.4 cm); the cube is moved before anything else; the
gripper is not turned to the cube's angle (nothing reads it).

| | Result | Bar |
|---|---|---|
| 1. Read-out, held-out touches, median error | 0.49 cm | at most 1 cm |
| 2. Random table problems (50 fresh; 20 earlier, all solved) | 98% | at least 70% |
| The same, recalled move replayed with no shift | 58% | |
| 3. Benchmark task 3, run once (50 episodes, one seed) | 86% | at least 60% |

All three pass. Notes:
- In criterion 2 the episode ends, by the benchmark's rule, the moment
  the cube is within 4 cm of its goal, which is while it is still
  held. Task 3 is the test that it stays: the cube ended within 4 cm
  in all 50 episodes.
- Task 3's seven failures are card 127's: five stopped with a slide
  read as at its goal, two never opened the drawer in 750 steps.

**All five tasks with this one agent** (50 episodes each, one seed;
tasks 1 and 2 are second runs, as a check that the cube changed
nothing; 4 and 5 were not attempted in design and have no bar):

| Task | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|
| Success | 84% | 96% | 86% | 0% | 0% | 53.2% |

Card 127 had 92% and 94% on tasks 1 and 2; with 50 episodes the
differences are within chance. The published 54.4% (GAS) uses
64-pixel frames alone; this agent uses 256-pixel frames,
proprioception and the contact sensor, so the two means are not a
fair comparison. In tasks 4 and 5 the agent stops without ever
acting on the cube's goal.

## 8. Decision

**Keep.** A remembered grasp, shifted by a read-out the agent trained
on its own touches (0.49 cm), moves the cube where the goal frame
shows it: 98% on random table problems against 58% for replay, and
86% on task 3. Limits: the read-out fails outside the range of
remembered grasps (9 cm off); one object, on the table only; the
cube's angle is not read; one seed. Open: tasks 4 and 5, where the
goal has the cube in a closed drawer. Adopted as version og2 (the
user, 2026-11-11).
