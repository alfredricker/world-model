---
id: "132"
title: memory test bed — on true visible state, which outcomes need memory, and the ceiling and floor for recalling them
rung: 3
serves: [P4, P12, P21]
status: done
verdict: fail
arch_version: og4
date: 2026-11-12
---

# 132: memory test bed

Roadmap step 3, rung 3. The user (2026-11-12): small cards patching
big problems lead to dead ends; isolate one component and perfect it
over about a week, and "it is okay to assume that the components it
depends on have been perfected". The component is memory, short and
long term. This card builds the test and its bounds; cards 133–136
each change one mechanism against it. It replaces card 131.

Biological counterpart: the eye sees sharply only where it looks, and
working memory holds the rest (change blindness, Rensink et al. 1997);
the hippocampus stores single episodes that later cue recall.

## 1. Question

With perception taken as perfect, how much of predicting a touch's
outcome depends on what the agent no longer sees, and how far apart
are recall with a perfect key and recall with no memory? P4 (memory).

## 2. What changes

Nothing in the agent. A test bed over stored play, one event per
touch, using the simulator's true state for what is in view.

```
view A (camera)   every variable's true value, except the cube: absent while it is inside
                  the closed drawer
view B (narrow)   only the variable the hand touches, before and after the touch
task              before each touch: given where the hand will touch, this episode's views so
                  far, and a store of earlier episodes, predict what the touch will show
prediction        recall: the 5 nearest stored events by a key, majority outcome
                  (the same in every arm; only the key differs)
scored events     A: a closed drawer is reopened: is the cube inside?
                  B: a drawer or window is pulled: does it move? (depends on a lock, in view
                     only when its button or that slide was touched)
arms              floor: key = the touch and the present view
                  ceiling: key = the touch and the true hidden value (cube inside; the lock)
                  last seen: key = the touch and each variable's last value in view
                  (version 27's belief rule), plus yesterday's "points left behind" on A
```

The truth of what is out of view is never an input, outside the
ceiling arm. What the agent has not seen in this episode it does not
know.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near (symbols) | yes | no | declared for the memory week (the user); removed when the winner returns to pixels |
| The world comes as a few named variables | no | no | no | no | declared; the test bed only, the mechanisms tested later must take vectors |
| A view covers part of the world (narrow view) | holds (first person) | holds (local map) | whole grid in view | holds | general; here imposed, since the camera sees the whole scene |
| One touch is one event | needs event boundaries | the same | one action | holds with contact sensing | card 120 |

## 3. Dependencies

- Card 120 (a touch brackets an event); the stored event streams of
  700 episodes (`tools/card132/build.py`).
- Recall by nearest stored cases: versions 20–27 (MiniGrid line),
  Model-Free Episodic Control (graph, `paper/model-free-episodic-control`).
- Known before this card (2026-11-11, on pixels, no card): a network
  predicting the next frame told a full reopened drawer from an empty
  one at 0.66 (chance 0.5, real frames 0.998); recall keyed on the
  touch alone 0.40; with "points left behind" 0.67 (graph, results
  `132-*`).

## 4. Data check

Run 2026-11-12 (`tools/card132/check.py`, under a minute, capped).
Store: 400 episodes, 7,830 touches. Test: 300 episodes, 5,870 touches.

| View | Scored event | Store | Test | How long out of view |
|---|---|---|---|---|
| A | Reopened drawer, cube inside | 40 | 40 | median 2 touches, at most 19 |
| A | Reopened drawer, empty | 205 | 158 | |
| B | Pull of a drawer or window | 3,325 | 2,541 | |
| B | of which did not move | 323 | 222 | |
| B | of which a lock was flipped since that slide was last touched | 921 | 697 | |

Under view B, across all test touches with a known target (5,580):
the target had not been seen yet in the episode at 25%, was unchanged
since last seen at 69%, and had changed while out of view at 5%
(11% for the cube, which the drawer carries).

- **Stored play is a shallow memory test.** The same object is often
  touched again at once (median gap 0–2 touches), and a hidden cube
  is revealed after 2 touches. Long delays have to come from the
  agent's own practice, in a later card.
- **The three shares are three different jobs:** not yet seen (a
  prior over likely states, the user's ingredient 4), unchanged
  (persistence), changed out of view (one variable's effect on
  another's remembered value).
- 40 reveals with the cube make that score coarse: one event is 2.5
  points.

## 5. Feasibility gate

This card is the gate for the week: its result is the pair of bounds.

- **Upper bound:** the ceiling arm.
- **Trivial baseline:** the floor arm.

Result of the gate, before the main run: the gate is the run; see
section 7.

## 6. Success criteria and prediction

Score: balanced accuracy (the mean of the accuracy on each of the two
outcomes; 50% is chance). Proposed, to be fixed with the user.

1. **The test needs memory:** the floor is at most 60% on both scored
   events.
2. **The test can be passed:** the ceiling is at least 95% on both.
3. **Long-term memory works from few episodes:** with the ceiling
   key and a store of 10 episodes, the score is at least 85% on both
   (stores of 10, 30, 100 and 400 are reported).

Reported without a bar: the last-seen arm, split by the three shares
of section 4. It decides what cards 133–136 must beat.

**Prediction.** 1 and 2 pass. Last seen will be near the ceiling on
pulls and on unchanged cases, and fail where the value changed out of
view and where nothing was seen yet. If it is within 5 points of the
ceiling everywhere, stored play is too easy and the next card makes
harder episodes before any mechanism is tried. 3 is uncertain for
reopenings: 10 episodes hold about one reopening with the cube.

**Budget.** No training, no rendering. Each arm under 2 minutes,
capped.

## 7. Result

Run 2026-11-12 (`tools/card132/bed.py`, under a minute, capped;
numbers in [results.json](results.json)). Balanced accuracy, test
episodes, store of 400 episodes unless said.

| Key | Reopened drawer: cube inside? (198 events) | Pull: does it move? (2,226 events) |
|---|---|---|
| Floor, camera view | 100.0 | 81.3 |
| Floor, narrow view (the touch alone) | 51.6 | 64.4 |
| Last seen (narrow view) | 79.7 | 70.2 |
| Ceiling (true hidden values) | 100.0 | 94.8 |
| Ceiling, store of 10 / 30 / 100 episodes | 58.9 / 73.2 / 98.4 | 65.0 / 81.8 / 93.8 |

| Criterion | Bar | Result | Verdict |
|---|---|---|---|
| 1. The test needs memory | floor at most 60% on both | camera view 100.0 on reopenings; narrow view 51.6 and 64.4 | fail |
| 2. The test can be passed | ceiling at least 95% on both | 100.0 and 94.8 | fail, by 0.2 |
| 3. Works from 10 episodes | at least 85% on both | 58.9 and 65.0 | fail |

- **With perfect perception and the camera's view, the hidden cube
  needs no memory.** The cube missing from view says where it is:
  there is one cube and one place to hide. Tasks 4 and 5 fail on
  reading, and on absence not being a value (card 131's point), not
  on memory. Memory is needed only when the view is partial.
- **In the narrow view, memory is needed** (reopenings: 51.6 without,
  100.0 with the truth).
- **Last seen holds the right content and recall reads it badly.**
  On pulls the remembered lock was right in 1,955 of 2,226 events,
  yet the score is 70.2 against 94.8. The key carries every remembered
  variable at a fixed weight, and the irrelevant ones (the cube's
  place, the other slide) outvote the lock. A rule that reads only the
  true lock scores 96.1. What to attend to in the key has to be
  learned from whether recall paid off (graph,
  `principle/recall-key-learned-from-recall-outcome`).
- **Reopenings, last seen:** 89.5 when the cube was seen earlier in
  the episode (137 events), 50.0 when it was not (61 events): those
  61 cannot be answered from this episode by any memory and belong
  to the prior, not to this score.
- **Ten episodes are too few even with a perfect key**; 100 are
  enough. Ten episodes hold about one reopening with the cube, and a
  majority of five neighbours cannot return it.

Changes made during the run, after the first numbers:
- Pulls in which a lock also flipped are left out of store and score
  (315 of 2,541 test pulls): two touches recorded as one.
- Every key also carries which object is touched. In 14% of pulls
  the recorded contact point is not at the handle, and the ceiling
  stood at 88.3 before this. A touch that changed nothing gets its
  object from the contact point, so a failed pull far from the handle
  is not counted as a pull.
- The "points left behind" arm was not run; its result on pixels
  stands (0.67).

## 8. Decision

**Revise.** The test bed is kept with three changes, and the week's
order changes. (1) Score in the narrow view only; the camera view
makes the hidden cube a question of reading. (2) Score separately
the events where the answer was seen earlier in the episode (memory)
and where it was not (prior). (3) The pull ceiling is bounded by the
data at 96.1; the bar becomes 93. Next card: the key's weighting
learned from recall's outcome, over the last-seen contents, since
that is where 25 points are lost; what to write (the planned card
133) comes after, with episodes of longer delay from the agent's own
practice.
