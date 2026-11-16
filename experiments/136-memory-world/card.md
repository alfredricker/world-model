---
id: "136"
title: a memory world — one factor per kind of memory, with the best possible answer known
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 136: a memory world

Roadmap step 3, rung 3. Seventh card of the memory work. The user
(2026-11-13): "is there a better test of our memory? Maybe we set up
a custom environment that the agent requires both strong short term
memory within an episode and strong long term memory across episodes
with many changing or static factors, some probabilistic, some
deterministic", and to the proposal below: "i agree".

Why: the lock-and-drawer scene has two conditions, nothing in it
changes by itself, and its upper bound in
[card 135](../135-state-that-changes-out-of-sight/card.md) could not
be reached by any memory. This card builds the test and reads the
current memory on it. The model is not changed. It is a test bed for
the memory component only; cards on the whole agent are still judged
on OGBench.

Biological counterpart: the factors follow the division of memory in
animals: working memory within a task, episodic memory of one past
occasion, semantic memory of what is always so, and the sense of how
long a fact stays true (Tulving 1972; Squire 2004).

## 1. Question

Does a small symbolic world, with one factor for each kind of
memory and an exact best possible answer, separate what the current
memory can do from what it cannot? P4, P12.

## 2. What changes

Nothing in the model. New: `tools/card136/world.py`.

```
world     17 two-state objects. A touch tries to flip the touched object; it works if the
          object's parents hold the values its rule requires. The agent sees only the object
          it touches, after the touch. 8 switches (no parents), 9 children, one per factor.
stream    500 episodes of 60 touches; the agent touches, with probability 0.6, one of the
          last 5 distinct objects it touched, else any object
```

| Child | Its parent | Memory it asks for |
|---|---|---|
| random | starts at random each episode, stays until touched | short term (the lock and drawer) |
| carry | keeps its state from one episode to the next | long term, one past episode |
| default | starts every episode at the same value | long term, a fixed fact |
| drift slow | flips unseen, 2–4% per step | trust that falls slowly |
| drift fast | flips unseen, 8–12% per step | trust that falls fast |
| timer | falls back T steps after being set | a learned timer |
| chance | allows; the touch then works 70% of the time | a probability |
| two | two parents, both must hold | a condition of two objects |
| rewired | swapped for another switch every 150 episodes | long term, recent over old |

Object numbers, required values, defaults, rates and T are drawn
afresh for each world; three worlds are scored.

**The best possible answer** ("observer"): exact Bayesian belief
over all 2^17 joint states, given everything the agent has seen and
the world's rules as they are at that moment. For states it is the
ceiling of any memory. For rules it is a reference only: no learner
knows of a rewiring at once.

**Scored:** every touch of a child in episodes 300 to 499: the
probability that it changes, by log loss (the penalty for the
probability given to what happened; lower is better). The store is
every earlier episode, refreshed every 20 episodes. For each factor,
"gap closed" is how far the memory moves from the base rate (the
share of that object's stored touches that changed it) to the
observer: 0% is no memory, 100% is the best possible.

**First reading:** card 134.1's recall as it is: one record per
object within the episode (seen, last value, whether the last touch
changed it, time since seen), other records chosen by contrast,
probability from the 25 nearest stored touches. Nothing is kept from
one episode to the next, as in the model now.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Objects are given, with two states | no | near | needs objects | no | declared: perception assumed perfect |
| An effect depends on a few other objects | mostly | mostly | varies | mostly | built in, 0 to 2 parents |
| Some facts last across episodes | yes (the map) | per seed | per game | yes (the room) | built in |
| Some things change unseen, at their own rates | yes | yes | varies | yes | built in |
| Episodes follow one another in one world | yes | no (new map) | levels of a game | yes | built in |

## 3. Dependencies

Cards 132 to 135. No simulator, no images.

## 4. Data check

Each child is touched 620 to 840 times in the 200 scored episodes of
a world, and about three times that in the store.

## 5. Feasibility gate

- **Upper bound:** the observer. **Trivial baseline:** the base rate.
- The gate passes if, for every factor, the observer's log loss is at
  least 0.1 under the base rate's (the factor can be tested), and the
  observer's stated probability is within 1 point of the observed
  share (it is right).

Result (2026-11-13, `tools/card136/reading.py gate`, 90 seconds,
`runs/136/gate.json`): passed. Mean of three worlds, log loss, base
rate against observer: random 0.694 / 0.097; carry 0.700 / 0.020;
default 0.564 / 0.020; drift slow 0.695 / 0.331; drift fast 0.694 /
0.522; timer 0.344 / 0.020; chance 0.658 / 0.379; two 0.552 / 0.076;
rewired 0.694 / 0.084. The observer's probability is within 0.35
points of the observed share in every world. (0.020 is the floor:
probabilities are clipped at 2%.)

## 6. Success criteria and prediction

The card tests the instrument, not the memory.

1. **It agrees with what is known:** on the "random" child, which is
   the lock and drawer of cards 134 and 134.1, the current recall
   closes at least 60% of the gap.
2. **It separates:** the best and the worst factor differ by at
   least 40 points of gap closed.
3. **It is steady:** for at least 7 of the 9 factors, gap closed
   differs by at most 20 points between the three worlds.

Prediction: all pass. The current memory should do well on random,
chance and two, partly on drift and timer, and badly on carry and
rewired, since it keeps nothing between episodes and weighs old and
new stored episodes alike.

## 7. Result

Run 2026-11-13, `tools/card136/reading.py` (90 seconds),
`runs/136/reading.json`, numbers in [results.json](results.json).
Mean of three worlds, log loss on every scored touch of each child.

| Child | Base rate | Current recall | Observer | Gap closed (three worlds) |
|---|---|---|---|---|
| random | 0.694 | 0.126 | 0.097 | 99, 94, 92% |
| carry | 0.700 | 0.128 | 0.020 | 80, 87, 86% |
| default | 0.564 | 0.020 | 0.020 | 100, 100, 100% |
| drift slow | 0.695 | 0.374 | 0.331 | 91, 86, 87% |
| drift fast | 0.694 | 0.559 | 0.522 | 79, 73, 85% |
| timer | 0.344 | 0.077 | 0.020 | 80, 82, 85% |
| chance | 0.658 | 0.475 | 0.379 | 72, 61, 63% |
| two | 0.552 | 0.174 | 0.076 | 78, 80, 80% |
| rewired | 0.694 | 0.304 | 0.084 | 59, 76, 57% |

1. **It agrees with what is known: pass.** 92 to 99% on the random
   child (bar 60).
2. **It separates: fail.** Best 100% (default), worst 64%
   (rewired): 36 points against a bar of 40.
3. **It is steady: pass.** All 9 factors within 20 points between
   worlds (the widest is rewired, 19).

Two readings beside the bars.

- **The recall of card 134.1 carries to a world it was not built
  on.** In every world and for every child the choice by contrast
  added the true parent (both parents for "two"), among 16
  candidates, with nothing changed. The rewired child keeps both its
  old and new parent.
- **The score over all touches hides the long-term factors.** A
  child is touched about 12 times in an episode, and after its first
  touch the outcome itself tells the parent's state. Scored only on
  cold touches (neither the child nor its parents seen yet in the
  episode; about 90 per child and world; run afterwards, not a
  declared measure), the factors separate fully:

| Child, cold touches | Base rate | Current recall | Observer |
|---|---|---|---|
| carry | 0.700 | 0.718 | 0.020 |
| default | 0.315 | 0.020 | 0.020 |
| timer | 0.124 | 0.020 | 0.020 |
| rewired | 0.693 | 0.708 | 0.510 |
| chance | 0.617 | 0.675 | 0.532 |
| two | 0.515 | 0.558 | 0.391 |
| random | 0.696 | 0.684 | 0.615 |

  The fixed facts (default, timer) are had in full from the store.
  What one past episode left behind (carry) is not had at all: the
  observer is certain and the recall is at the base rate. On
  chance, two and rewired the recall is worse than the base rate
  when nothing has been seen; the observer gains there by reasoning
  from one child's outcome to the parent it shares with another,
  which the recall does not do.

## 8. Decision

**Revise.** The world and the observer are kept. The scoring is
revised: cold touches (long-term memory) and warm touches (short-
term memory) are scored apart, and the behaviour should return to an
object less often so that cold touches are not a tenth of the data.
The first target the instrument names is memory that lasts from one
episode to the next.
