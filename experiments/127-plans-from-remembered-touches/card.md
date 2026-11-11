---
id: "127"
title: plans from remembered touches — backward over conditions, each step a remembered touch done again, on OGBench's scene tasks 1 and 2
rung: 3
serves: [P21, P12, P6, P4]
status: done
verdict: pass
arch_version: 27
date: 2026-11-11
---

# 127: plans from remembered touches

Roadmap steps 3 and 7 in their smallest form (skills; planning over
conditions), on what cards 125–125.2 read from memory: which patches
decide whether a pull works, their states, and the place to touch to
change them. The user (2026-11-10): "Let's try out the planning
algorithm on OGBench." Card 126 (a trained layer) waits for a harder
scene.

Biological counterpart: the hippocampus replays remembered episodes
when a route is planned, forward and backward from the goal (Pfeiffer
and Foster 2013); a remembered action is reused, not recomputed.

## 1. Question

Given a goal frame, can the agent choose and order remembered touches
so that each one's condition holds when it is done, and reach the
goal in OGBench's scene? Rung 3. P21 (work backward over conditions,
no rollouts), P12 (chains), P6, P4.

## 2. What changes

Two parts are new; both read only the memory of cards 125–125.2.

```
skill:    "redo remembered touch i": reach its approach point (lift, move over, descend),
          then follow its recorded effector path. Closed loop on proprioception.
planner:  for each variable (a place's usually-changed patches) compare now with the goal;
          for a variable that differs, recall the touch whose before matches now and whose
          after matches the goal there;
          if that place's outcome depends on a condition that now reads "nothing changes",
          first redo a touch from the condition's cause place (card 125.2);
          a condition is not restored while a pending step needs it.
          After each touch the scene is read again and the plan is remade.
```

- **A touch's after-state** is the steady reading before the next
  touch of its episode, so the arm is out of it.
- **A place's variable** is the 10 patches its touches change most,
  relative to touches in general, between steady readings.
- **Reading the live scene:** the arm goes to a rest pose (the
  remembered gripper position farthest from every place in use) and
  holds still before each reading.
- **No search over imagined futures:** each step is one recall, and
  the order comes from the conditions (P21).
- **Not in this card:** the cube. Picking and placing needs a
  remembered touch adapted to where the cube is now. Tasks 3–5 need
  it, so only tasks 1 and 2 are attempted.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A remembered touch can be done again from its recorded effector path | no: the target moves with the view; needs a path relative to the target | the same | a click repeats exactly | fixed base and fixed fixtures only | holds for buttons and handles; fails for the cube |
| An action moves the effector by a learnable amount | holds | holds | holds | holds | a linear fit on memory |
| The goal is given as a frame | no | no | no | sometimes | OGBench's goal image |
| The agent can rest and look before acting | holds | creatures do not wait | holds | holds | declared |

## 3. Dependencies

- Cards 125–125.2 (conditions, states, cause places), 122 (signals).
- Backward search over remembered states: Eysenbach et al. 2019;
  effects inside the next precondition: Konidaris et al. 2018
  (LITERATURE). Replaying stored trajectories as skills is old in
  robotics (teach and repeat); choosing them by recall of before and
  after states under learned conditions is this card's test.
- Published bar: GAS 54.4% and HIQL 47.9% on `visual-scene-play`,
  means over all five tasks from 64 × 64 pixels alone. **Not
  comparable as it stands:** this agent uses 256-pixel frames,
  proprioception, a memory built with the contact sensor, and attempts
  two tasks of five.

## 4. Data check

Memory: episodes 400–499 (2,106 touches): 361 and 348 presses of the
two buttons, 51 drawer openings from closed, 102 window openings, 72
drawer closings. A touch redone takes about 38 steps; an episode
allows 750.
- Task 1: both unlocked, both closed → open both (2 touches).
- Task 2: both locked, both open → both closed and locked again
  (unlock, close, lock, twice: 6 touches).

## 5. Feasibility gate

Run 2026-11-10 (`tools/card127/reenact.py`, capped, 2 minutes). The
skill alone, in the live `scene-v0`, from a task's start. The touch to
redo was chosen with the truth here; the planner must choose it by
recall.

| Touch redone | Remembered | Success, 30 trials |
|---|---|---|
| Press button 1 (drawer's lock) | 361 | 100% |
| Press button 2 (window's lock) | 348 | 100% |
| The same presses from task 2's start | | 100%, 100% |
| Open the drawer | 51 | 83% (60% and 70% in two runs of 20) |
| Open the window | 102 | 87% |
| Chain: unlock, then close the drawer (task 2's start) | | 97% |

The skill works without training. Pulls fail in 13–40% of single
tries, so the planner's rereading and retrying matters. Not shown:
that recall picks the right touch, and that states are read correctly
from the live scene (the arm's rest pose may differ from play data).

## 6. Success criteria and prediction

Agreed with the user 2026-11-11 ("yes"), before the planner was built.
1. **Development only on self-made problems;** the benchmark's tasks
   are run once, at the end.
2. **Benchmark:** success on task 1 at least 80% and on task 2 at
   least 60%, 50 episodes each, 750 steps.
3. **The conditions do the work:** with the condition check removed
   (pull at once), task 2 falls below 20%.

**Prediction.** Task 1 passes. Task 2 is the test of the chain and the
risk is reading the live scene.

**Budget.** No training. Each run under 10 minutes, capped.

## 7. Result

Run 2026-11-11. `know.py` reads the planner's knowledge from memory,
`plan.py` runs it live in `visual-scene-v0` at 256 pixels, `diag.py`
scores the log. Each run took under a minute.

**What the agent knows, read with no truth** (`runs/127/know.json`):
of 10 places, four have a variable (their touches change their 10
patches at least 5 times as much as touches in general) and the same
four have a condition with a cause place elsewhere. Scored afterwards:
two are the window handle with the window's button as cause place, two
the drawer handle with the drawer's button. The button places and the
cube places get neither.

**Development,** on self-made problems: each lock, and each slide's
start and goal (open or closed), drawn at random through the
environment's own task option, the cube unmoved. This differs from the
draft, which said validation episodes. Four versions, 30 problems each
with one seed, then 50 fresh ones:

| Version | Success |
|---|---|
| First build: a variable differs if its distance to the goal is over a threshold | 67% |
| Rest pose moved from the medoid of steady touches (it was in front of the window) to the remembered position farthest from every place in use | 77% |
| A variable differs if (now, goal) is nearer a remembered change than a remembered non-change | 93% |
| Recall only touches across which a change was seen (card 125's outcome) | 97% |
| The last version on 50 fresh problems | 92% |

**Benchmark, run once** (50 episodes each, one seed):

| | Success | Bar | Mean steps when solved |
|---|---|---|---|
| Task 1 (open both) | 92% (46 of 50) | 80% | 160 |
| Task 2 (unlock, close both, lock again) | 94% (47 of 50) | 60% | 318 |
| Task 2, condition check removed | 0% (0 of 50) | under 20% | none |

All three criteria pass. With 50 episodes each rate is uncertain by
about 7 points.

**What worked and what did not** (`diag.py`, truth for scoring only):
- Lock states were read right in 98–100% of readings, live and in the
  goal frame, although the goal frame's arm is at a random pose.
- A slide away from its goal was noticed in 86–100% of readings. A
  drawer at its goal was called different in 17–22% (window 0–4%).
- Closing pulls worked in 82–97% of tries. Opening the drawer worked
  in 35% of tries on task 1; the planner retried, and three of task
  1's four failures are drawers that never opened in 750 steps.
- All three failures on task 2 are a window left 5–9 cm from closed
  (the benchmark allows 4) that the planner read as closed and stopped.
- Found while developing: the contact sensor misses some pulls. A
  brief contact is recorded, then the drawer opens with no contact
  sensed, so the touch's after-state shows a change that redoing the
  touch cannot produce. Requiring a change seen across the touch
  itself removed these.

**Against the published bar.** Not a fair comparison, and not a win:
GAS's 54.4% is a mean over five tasks from 64-pixel frames alone.
Counting the three cube tasks as zero, this agent's five-task mean is
37%. It also uses 256-pixel frames, proprioception and a memory built
with the contact sensor.

## 8. Decision

**Keep.** A planner that works backward over conditions read from 100
episodes of play, each step a remembered touch done again, solves the
scene's two cube-free benchmark tasks at 92% and 94% with no training
and no search over imagined futures; without the condition check the
six-touch task falls to 0%. Limits: one seed and one memory; no cube
(tasks 3–5); the rest pose rule and the 10-patch variables were chosen
on self-made problems in this one scene; opening the drawer is
unreliable per try; the goal is a frame; the fixed camera and the
contact sensor are assumed. Not adopted into ARCHITECTURE here: that
is a new version and the user's call.
