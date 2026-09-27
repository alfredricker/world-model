---
id: "008"
title: discovery from one signal
rung: 0
serves: [P12, P16, P4, P20]
status: done   # draft | approved | gated | running | done | abandoned
verdict: fail
arch_version: 0
date: 2026-09-27
---

# 008: discovery from one signal

## 1. Question

Card 007 discovered the chain door open → holding the matching key → empty
hands from one supplied goal, but it read frames through card 005's
network, whose training also used supplied success signals for door
unlocked, door open, holding the matching key and four walk targets. Those
signals may have made the conditions easy to find. Does discovery still
work when the only success signal the network has ever seen is "on the
goal square", and if not, does it work when the network keeps learning
from the goals it discovers itself? Serves P12 (conditions and subgoals),
P16 (conditions inferred from successes), P4 (the agent's state keeps what
its goals need) and, in a first form, P20 (practising self-found goals).

## 2. What changes

One change from card 007: the network that turns frames into internal
state.

- **Encoder G (goal square only).** Card 005's architecture and settings
  (10k episodes, 100k updates), but trained with one achievement head
  (on the goal square) and one walk head (walk to the goal square). No
  other success signal.
- **Arm F, frozen.** Card 007's procedure unchanged, on encoder G's
  internal state.
- **Arm C, continual.** Before discovering the conditions of level k's
  goal, encoder G is fine-tuned for 20k updates to predict whether each
  action achieves that goal (an achievement head, trained as in card 005),
  while keeping its earlier heads trained (same data, replayed), so the
  internal state can learn to notice what that goal depends on. The
  goal's success signal is the agent's own discovered detector from the
  level above (C1: nothing supplied beyond the goal square). Discovery at
  level k then runs on the fine-tuned state, and its detector is stored as
  the next level's fixed goal labels. After the last level, every level's
  small heads are refitted on the final encoder so detectors, walking and
  acting all use one network.
- Everything else as in card 007: thresholds (A > 0.5, walk value > 0.1),
  stopping rule (fewer than 10 successes), at most 4 levels, acting with 5%
  random moves for at most 200 steps, the same test layouts.

- **Speed (declared 2026-09-27):** training computes in bf16 with TF32
  matrix products, fused Adam, a single-call weight-average update and a
  compiled network; about 3.5× faster than card 005 (measured 64 → 242
  updates/s). Numbers are therefore not bit-identical to card 005's.
  Approved by the user 2026-09-27, with the whole card run here.

Why arm C is expected to matter: walking to the goal square needs the door
open, so encoder G must notice the door; but from the locked side the key
makes no difference to that walk, so nothing in its training asks it to
notice the key.

## 3. Dependencies

Card 005 (the architecture and training that learn achievement and walk
values from frames); card 007 (the discovery procedure, passed).
Fine-tuning a network on new targets while replaying the old ones is
standard practice.

## 4. Data check

Card 005's data: 10k training episodes with about 1000 arrivals on the goal
square; 500 new test layouts. Card 007 found about 5000 successes each for
the door and key levels.

## 5. Feasibility gate

- **Encoder G learns its own task:** on new layouts, the forward-facing-
  goal cell ≥ 95% right (card 005 criterion 1), and walking to the goal
  square with area under the curve ≥ 0.95 and rank correlation ≥ 0.9
  (card 005 criterion 2). If not, the test is uninformative and the card
  stops.
- **Upper bound:** card 007's exact gate (passed).
- **What the state contains** (reported, not a criterion): card 006's
  linear read-out for door open, holding the matching key and empty hands,
  for encoder G and after each arm-C fine-tune.

## 6. Success criteria and prediction

Card 007's three criteria, for each arm on the 500 new layouts:

1. **The chain:** the first two detectors match door open and holding the
   matching key with area ≥ 0.95; the procedure stops within 4 levels.
2. **They behave as conditions:** from test frames with the detector on,
   walking and taking the achieving action achieves the level's goal
   ≥ 90%; with it off, ≤ 10%.
3. **Acting:** ≥ 90% of new layouts reach the goal square.

How to read the outcome:
- F passes: discovery needs no signal beyond the one goal; card 007's
  caveat is removed.
- F fails, C passes: the agent must learn from the goals it discovers;
  that becomes part of the mechanism (listed in ARCHITECTURE.md later).
- Both fail: one goal-square signal is not enough to start discovery;
  revise before building on it.

Prediction: the gate passes; F finds door open, then stalls or is noisy at
the key (read-out of "holding the matching key" well below card 005's
1.0); C passes all three. Runtime: encoder G about 20 minutes; arm F about
15; arm C about 40 (three fine-tunes plus discovery and refits). Arm C is
over 30 minutes, so it is handed to the user as a command unless the user
approves running it here.

## 7. Result

Script `runs/008.sh`; numbers in [results.json](results.json). Encoder G:
100k updates in 8 minutes (208 updates/s).

**Gate passed:** encoder G steps onto the goal square correctly 100% of the
time; walking to it: area 1.0, rank correlation 0.99.

**What its state contains** (linear read-out, area on test frames; card
005's network: 1.0 for all three):

| | Door open | Holding the matching key | Empty hands |
|---|---|---|---|
| Encoder G | 0.9999 | **0.73** | 0.95 |
| Arm C after training on level 1's goal | 1.0 | 0.71 | 0.88 |
| after level 2's goal (door open, self-found) | 1.0 | 0.88 | 1.0 |
| after level 3's goal (key, self-found) | 1.0 | **0.98** | 1.0 |

**Discovery** (detector's best matches; behaviour = achieved from test
frames with the detector on / off, 300 each):

| Level | Arm F (frozen) | Arm C (continual), final network |
|---|---|---|
| 1 | forward; door open 0.996; 100% / 0% | forward; door open 0.996; 100% / 0% |
| 2 | toggle; door closed 0.93, key 0.71; 36% / 30% | toggle; **holding the matching key 1.0**; 100% / 0% |
| 3 | "right" (spurious; detector on everywhere) | pickup; **empty hands 1.0**; 100% / 0.3% |
| 4 | none | "left" (spurious); red key held 0.80; 0% / 0% |

| Acting, 500 new layouts | Arm F | Arm C |
|---|---|---|
| Discovered conditions | 0% | **75.8%** (mean 37.7 steps) |
| Goal square only | 0% | 0% |
| Random play (exact, 100 layouts) | 1.4% | 1.4% |

| Criterion | Arm F | Arm C |
|---|---|---|
| 1. Chain; stops within 4 levels | fail (level 2 is not the key) | chain right (0.996, 1.0), but level 4 is spurious: the stopping rule did not fire |
| 2. Behaves as conditions | fail at level 2 | pass for levels 1–3; level 4 0% / 0% |
| 3. Acting ≥ 90% | fail (0%) | fail (75.8%) |

**Diagnosis of arm C's spurious level 4.** Level 3's detector (empty hands
with the key on the floor, within walking reach) "turned on" 1837 times on
ordinary moves in training. Being within walking reach does not change
when walking, so these are threshold crossings: the walk value is 0.95^steps,
and the far end of the room sits near the 0.1 threshold (0.95^30 = 0.21,
0.95^45 = 0.10), so learned errors there flip the detector as the agent
moves. Those flips became level 4's "successes". Card 007 had none (0
move-successes at level 3); the fine-tuned encoder's walk values are less
precise at long range. The same flicker likely costs acting (more steps,
37.7 vs 17.6 in card 007, and failures). The final network and level heads
were not saved, so acting cannot be re-run without the spurious level;
the next run saves them.

## 8. Decision

**Revise.** The main question is answered: discovery does not work from a
state trained on the goal square alone (arm F: the key is barely in it,
read-out 0.73), and it does when the agent keeps learning from the goals it
discovers (arm C: the key rises to 0.98 and the chain door open → matching
key → empty hands is found and behaves as conditions). Learning from its
own discovered goals becomes part of the mechanism. What failed is a
detail of the detector: "within walking reach" read as walk value > 0.1
confuses far with unreachable. Next: a reachability detector that does not
depend on distance (for example a separate "reachable at all" output
trained without discounting, or a threshold far below the smallest value
at the room's longest walk), then rerun arm C with saved networks.
