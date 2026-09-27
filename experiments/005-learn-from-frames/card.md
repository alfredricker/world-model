---
id: "005"
title: learn from frames
rung: 0
serves: [P12, P6, P9, P19]
status: running   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-26
---

# 005: learn from frames

## 1. Question

Cards 003–004 showed, with the simulator's variables and exact answers, that
goals decompose into "an action plus the conditions under which it works",
with walking to X as one such action. Can a network learn the two quantities
that theory rests on from pixels alone, and do the same conditions come out
of what it learned? The quantities:

- **Achievement probability** A(frame, action, goal): will this action make
  the goal true now? (card 003)
- **Walk value** W(frame, target): can the target be reached by turns and
  forward steps alone, and how soon, as 0.95^steps, 0 if unreachable?
  (card 004)

This is step 1 of card 003's appendix. It serves P12 and P6 (predicting what
an action achieves), P9 (walking to X as a skill with its conditions) and
P19 (how many successes are needed).

## 2. What changes

No model yet (arch_version 0); this card builds the first one.

- **Observation (declared prior, C3).** The whole 8×8 room seen from above,
  8-pixel MiniGrid tiles, plus a ninth row showing what the agent carries
  (as Crafter shows its inventory): a 72×64×3 frame. The world is fully
  observed, so every answer is decidable from one frame (LESSONS: check the
  answer can be known from what the model sees); memory comes later.
- **Network.** One shared encoder (3 strided convolutions, 72×64 → 9×8 ×
  64, then a 256-unit layer), two heads: A gives a logit per goal and
  action; W gives a value per target and movement action (left, right,
  forward), W(frame, target) = the best of the three.
- **Goals and targets are supplied tasks (C1, until P20).** Goals: door
  unlocked, door open, holding the key that opens the door, on the goal
  square. Walk targets: on the goal square, facing the door, facing the
  matching key, facing the other key. Each comes with a success signal (does
  it hold in this frame), as a supplied task would; what a goal needs is
  never given.
- **Training.** A: logistic loss on every step of random play where the goal
  did not already hold, target 1 if it holds after the step. Steps where
  something was achieved are over-sampled (a quarter of each batch) with
  importance weights, so the probabilities stay calibrated. W: Q-learning on
  the movement steps only (the data is random play; Q-learning does not need
  the data to come from a good walker): target 1 if the next frame is at the
  target, else 0.95 × the best next value from a slowly updated copy.
- **Data.** Card 003's world and random play: 3000 episodes (seed 3) for
  training, 500 episodes on new layouts (seed 99) for testing.

## 3. Dependencies

Cards 003–004 (the world, the exact answers, the condition finder, the
planner). A frame renderer written for speed, tested pixel-for-pixel
against MiniGrid's own rendering. Q-learning from off-policy data on a small
deterministic world is established (LITERATURE: DQN-style fitted values).

## 4. Data check

From card 003: in 3000 training episodes, 985 unlocks, 1522 matching-key
pickups, 309 goal arrivals; failed toggles at the locked door in 43% of
episodes. The test set is checked for the same contrast cells (section 6).

## 5. Feasibility gate

- **Upper bound:** exact values from the simulator; a test checks that
  different states of a layout give different frames, so a perfect frame
  model can reach them.
- **Trivial baseline:** the base rate of each goal per action (the network
  without the frame), and for W the fraction of reachable states.

## 6. Success criteria and prediction

On the 500 new layouts:

1. **Achievement contrasts.** In each cell below, at least 95% of test
   states get the right side of 0.5, and the mean predicted probability is
   ≥ 0.9 where the truth is 1 and ≤ 0.1 where it is 0:
   toggle facing the locked door holding the matching key (1), the other key
   (0), nothing (0); pickup facing the matching key with empty hands (1),
   facing the other key with empty hands (0, for "holding the key that opens
   the door"); forward facing the goal square (1, for "on the goal square").
2. **Walk value.** Per target, area under the ROC curve ≥ 0.95 for
   reachable versus not; on reachable states, rank correlation ≥ 0.9 between
   W and the exact walking distance (negated); walking to the goal square
   from the left room correct (W > 0.1 exactly when the door is open) in
   ≥ 95% of states.
3. **Same conditions.** Card 003's and 004's condition finder, run on the
   test states with the learned predictions in place of the true outcomes
   (A > 0.5; W > 0.1), gives the same main rule for unlocking, holding the
   key and walking to the goal square as from the true outcomes.

Also reported: criterion 1's unlock cells when training keeps only 10, 30,
100 or 300 unlock episodes (all episodes without an unlock kept), P19.

Prediction: 1 and 3 pass; 2 passes on reachability but the distance
correlation may fall short near 0.9. From exact counts 10 unlocks sufficed;
from pixels I expect about 100. Budget: 5 training runs, each under 10
minutes (measured before the main run).

## 7. Result

**Main run** (`runs/005_main`, 30k updates, 408 s, 85 updates/s; numbers in
[results.json](results.json)). Test: 500 new layouts, 320k states.

| Achievement cell (truth) | n | Mean predicted | Right side of 0.5 |
|---|---|---|---|
| toggle, facing locked door, matching key (1) | 151 | 0.995 | 99.3% |
| toggle, facing locked door, other key (0) | 643 | 0.000 | 100% |
| toggle, facing locked door, nothing (0) | 242 | 0.000 | 100% |
| pickup, facing matching key, empty hands (1) | 247 | 0.963 | 97.6% |
| pickup, facing other key, empty hands (0) | 253 | 0.067 | **94.1%** |
| forward, facing goal square (1) | 50 | 0.999 | 100% |

Area under the curve per goal at the taken action: 1.0 for all four
(base rates 0.0006–0.004).

| Walk to | Reachable share | Area under curve | Rank corr. with steps | Mean error of W |
|---|---|---|---|---|
| Goal square | 0.093 | 0.999 | **0.72** | 0.11 |
| Door | 0.9998 | **0.76** (about 60 unreachable) | 0.92 | 0.04 |
| Matching key | 1.0 | undefined | **0.83** | 0.03 |
| Other key | 0.991 | **0.93** | **0.81** | 0.04 |

Walking to the goal square from the left room: W > 0.1 matches the truth in
95.3% of states (mean W 0.50 with the door open, 0.06 closed).

Conditions from learned predictions: unlocking gives the same rule as the
truth (facing the door + matching key, 0.993). Holding the key gives
"facing a key + empty hands" (0.51): the 6% of other-key pickups predicted
above 0.5 are counted as successes, so "matching" no longer keeps 95% of
them and the finder drops it. Walking to the goal square gives "holding
the matching key" (0.35) instead of "door open": the 5% of wrong W > 0.1
predictions swamp the 9% truly reachable.

| Criterion | Verdict |
|---|---|
| 1. Achievement contrasts | fail, narrowly: one cell at 94.1% (other key) |
| 2. Walk value | fail: goal reachability passes (0.999, 95.3%), distance ranking and rare-negative targets do not |
| 3. Same conditions | fail for the key and the walk; pass for unlocking |

Diagnosis: training loss for A reached about 0 while test errors remain on
the key colour, so the achievement head overfits training layouts; the walk
loss was still falling at 30k updates (values propagate ~25 steps through a
slowly updated copy). Separately, the condition finder's rule that a
condition must keep 95% of successes is brittle to a few percent of false
positives. P19 runs not done: not informative until the main run passes.

**Second round** (approved by the user 2026-09-26, runs up to about 25
minutes; criteria unchanged). Three changes, measured one at a time:
(1) the condition finder adds the atom with the largest FOIL gain (kept
successes × gain in log achievement probability) instead of requiring 95%
of successes kept; on cards 003–004's exact data it gives identical rules
for all ten goals and walk targets; re-scored on the main run's saved
network first, so its effect is seen alone. (2) 10,000 training episodes
instead of 3,000 (same test set), 30k updates. (3) the same with 100k
updates.

One scoring bug found and fixed during the round: the walk evaluation
included the last frame of episodes that ended on the goal square, where
the agent never walks again. These 50 frames were the only states in the
test set from which the door "could not be reached", which made the door's
area under the curve 0.52–0.76. With them excluded the door, like the
matching key, is always reachable in this world, so neither area is
defined. All three networks were re-scored; training is unaffected.

| Run | Wrong-key pickup right side | Goal-square walk: rank corr. / from left room | Key / other key rank corr. | Other key area | Conditions read back (unlock, key, walk to goal) |
|---|---|---|---|---|---|
| Main, old finder | 94.1% | 0.72 / 95.3% | 0.83 / 0.81 | 0.93 | right, wrong, wrong |
| Main, gain finder | same | same | same | same | right, right, right |
| 10k episodes, 30k updates | 95.3% | 0.77 / 91.3% | 0.85 / 0.81 | 0.96 | right, right, right |
| **10k episodes, 100k updates** | **99.6%** | **0.95 / 99.4%** | **0.95 / 0.94** | **0.9996** | **right, right, right** |

The final network (`runs/005_data10k_long`, 100k updates, 20 minutes):
every achievement cell ≥ 99.6% on the right side of 0.5 with mean
predictions 1.0 / 0.0 / 0.0 / 0.998 / 0.003 / 1.0; walk to goal area 1.0;
rank correlations 0.95 (goal square), 0.98 (door), 0.95 (key), 0.94 (other
key); from the left room W is 0.60 with the door open and 0.01 closed.
Conditions read back from its predictions: toggle needs facing the door +
matching key (1.0 each); pickup needs facing the matching key + empty hands
(0.996); walking to the goal square needs the door open (0.988).

| Criterion (final network) | Verdict |
|---|---|
| 1. Achievement contrasts | pass |
| 2. Walk value | pass where measurable (door and matching key are never unreachable here) |
| 3. Same conditions | pass |

Which change did what: the finder alone fixed criterion 3; more rooms
fixed the key colour (criterion 1); longer training fixed walk distances
(criterion 2). P19 runs (10, 30, 100, 300 unlock episodes, same settings)
are running: `runs/005_p19.sh`.

## 8. Decision
