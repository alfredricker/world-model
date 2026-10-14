---
id: "007"
title: condition discovery
rung: 0
serves: [P12, P16, P4, P9]
status: done   # draft | approved | gated | running | done | abandoned
verdict: pass
arch_version: 0
date: 2026-09-26
---

# 007: condition discovery

## 1. Question

Given only pixels and one supplied goal ("be on the goal square"), can the
agent discover the chain of conditions below it (door open, then holding
the matching key) as detectors it learned itself, with no list of
candidate conditions from anyone? Cards 003–004 found the chain with the
simulator's variables; card 005 learned from pixels when an action achieves
a goal; card 006 showed the conditions cannot be read off the network's
internal state afterwards. This card makes discovery the learner's own
procedure. It serves P12 (conditions and subgoals), P16 (conditions
inferred from successes), P4 (only conditions that matter are formed) and
P9 (walking to X is the skill each condition leans on).

## 2. The mechanism

It restates cards 003–004 so that every step uses something learned.

For a goal g the agent can recognise (supplied, or discovered one level up):

1. **Achieving action.** Learn A(frame, action, g), the chance an action
   achieves g now (card 005). The achieving action is the one with the
   most frames where A > 0.5.
2. **Ready states.** The frames where that action would achieve g
   (A > 0.5): for the goal square, facing it.
3. **Walking part.** Learn a walk value to the ready states (card 005's
   Q-learning on movement steps, target: the next frame is ready). Getting
   to a ready state by walking is the positional part of the condition
   (proximity is a condition, as the user argued; card 004).
4. **The discovered condition** is the detector "a ready state is within
   walking reach" (walk value > 0.1). It is off exactly when something
   walking cannot provide is missing: for the goal square, the door being
   open. No vocabulary is involved: it is defined by the agent's own
   values.
5. **Recursion.** The detector turning on becomes the next goal, with the
   agent's own detector as its success signal. Steps 1–4 repeat on frames
   where it is off: the door detector turns on at a toggle; ready states
   are facing the locked door with the key; the next detector is "that is
   within walking reach", i.e. holding the matching key; it turns on at a
   pickup; its detector is "facing the matching key with empty hands is
   within reach". That last one has no achieving action in this world (no
   drop action, keys only get picked up), so the chain stops there: a
   condition the agent must keep, not make (P12: keep the conditions built
   up; do not pick up the other key).

Each discovered condition is a separate learned detector by construction,
so separability of the internal state is not needed (card 006, revised).
Limit: one detector per level holds everything walking cannot provide; two
such conditions needed at once (a key and a switch) would merge into one
detector. This world cannot test that; chained rooms can.

## 3. What changes

Card 005's final network is frozen and supplies the frames' internal state
z; its goal and walk heads for anything but the goal square are not used.
New: small heads on z, trained per level (an achievement head for a
discovered goal, a walk head to learned ready states), and the procedure
above. Thresholds (A > 0.5, walk value > 0.1) are card 005's. At most 4
levels. Only one success signal is supplied: on the goal square (C1).
Acting (criterion 3) walks greedily on the learned walk values, with a
random movement 5% of the time to break loops, for at most 200 steps.
Stopping rule: a level whose best action achieves its goal fewer than 10
times in the training data has no achieving action. Approved by the user
2026-09-26. Code: `src/worldmodel/discover.py`.

## 4. Dependencies

Card 005 (achievement and walk values learned from pixels, passed); card
004 (walking to X as an action whose conditions say when X can be reached,
exact). Q-learning to a learned target set is card 005's method with a
different target.

## 5. Data check and feasibility gate

Card 005's data (10k training episodes, 500 new test layouts). Each level
needs enough successes of its own goal in random play: goal-square
arrivals ~1000, unlocks ~3200, matching-key pickups ~5100 in training.

- **Upper bound:** the same procedure with exact values (simulator
  achievement and exact walking reachability) recovers door open, then
  holding the matching key, then stops. Run first; if it does not, the
  mechanism is wrong and the card stops.
- **Trivial baselines:** acting on the top goal's walk value alone (fails
  whenever the door is shut); random play.

## 6. Success criteria and prediction

On the 500 new layouts:

1. **The chain.** Starting from the goal square, the first two discovered
   detectors match "door open" and "holding the matching key" with area
   under the curve ≥ 0.95 against the simulator's variables (on test frames
   where the level above is unmet), and the procedure stops within 4
   levels with a condition that has no achieving action.
2. **They behave as conditions** (checked with the simulator, C4): from
   test frames where a detector is on, walking (exact walker) to the
   nearest ready state and taking the achieving action achieves that
   level's goal ≥ 90% of the time; where it is off, ≤ 10%.
3. **Acting.** Pursuing the lowest unmet discovered condition, walking by
   the learned walk values and taking the learned achieving actions,
   reaches the goal square in ≥ 90% of new layouts, against acting on the
   top goal's walk value alone and random play in the same steps.

Prediction: the gate passes; criteria 1 and 2 pass for the door; the key
level is noisier because its success signal is itself a learned detector
(errors compound down the chain, as the P19 runs suggest); criterion 3 is
the most at risk because greedy walking on learned values can loop.
Runtime: heads on stored z, minutes per level; under 30 minutes in all.

## 7. Result

`runs/007_main` (smoke test on 400 episodes first; full run about 15
minutes). Numbers: [results.json](results.json).

**Gate (exact values):** level 1 achieved by forward, detector = door open
(0.998); level 2 by toggle, detector = holding the matching key (1.0);
level 3 by pickup, detector = empty hands (1.0); level 4: no action
achieves it (0 successes). Passed.

**Learned (only the goal-square signal supplied):**

| Level | Goal | Achieving action (training successes) | Discovered detector: best matches (area) | On in test frames where unmet | Behaviour: achieved from detector on / off |
|---|---|---|---|---|---|
| 1 | on the goal square (supplied) | forward (1014) | facing goal 0.998, **door open 0.996**, right room 0.996 | 9.3% | 100% / 0% |
| 2 | level-1 detector on | toggle (5176) | **holding the matching key 1.0**, door closed 0.88 | 33.4% | 100% / 0% |
| 3 | level-2 detector on | pickup (4907) | **empty hands 1.0** | 24.3% | 100% / 0% |
| 4 | level-3 detector on | none (0 successes) | stops: a condition to keep | | |

Level 1's best single match is "facing the goal" because the detector's
score is the walk value, which also rises near the goal; as an on/off
detector it is "the goal square is within walking reach", on 9.3% of test
frames, as in the exact gate. Behaviour used 300 frames per cell.

**Acting** on 500 new layouts, 200 steps: pursuing the lowest unmet
discovered condition reached the goal square in 99.2% (mean 17.6 steps;
the exact shortest solution averaged 15.4 in card 004); acting on the goal
square's walk value alone 0%; random play 1.4% (exact, 100 layouts).

| Criterion | Verdict |
|---|---|
| 1. The chain (door open, holding the key, ≥ 0.95; stops within 4 levels) | pass: 0.996, 1.0; stops at level 4 |
| 2. Behaves as conditions (≥ 90% on, ≤ 10% off) | pass: 100% / 0% at every level |
| 3. Acting ≥ 90% | pass: 99.2% against 0% and 1.4% |

**Caveat that limits the claim.** z comes from card 005's network, whose
encoder was trained with supplied success signals for door unlocked, door
open, holding the matching key and four walk targets. Discovery here used
only the goal-square signal, but the internal state it read was shaped by
the others, so it may have made these conditions easy to find. One seed.

## 8. Decision

**Keep.** The mechanism discovers the whole chain from one supplied goal,
the discovered conditions behave as conditions, and acting on them solves
new layouts. Before building on it, the next card removes the caveat: the
same procedure on an encoder trained with only the goal-square signal (or
with no goal signal), so that nothing but the one supplied goal has shaped
what the agent can see.
