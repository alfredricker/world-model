---
id: "136.1"
title: the memory world scored cold and warm — long-term and short-term memory read apart
rung: 3
serves: [P4, P12]
status: done
verdict: pass
arch_version: og4
date: 2026-11-13
---

# 136.1: the memory world, scored cold and warm

Roadmap step 3, rung 3. [Card 136](../136-memory-world/card.md)
built the world and its best possible answer; its score over all
touches did not separate the factors (36 points against a bar of
40), because most touches of a child come after the child's own
outcome has shown its parent's state. The user (2026-11-13) approved
this revision. The model is not changed.

Biological counterpart: as card 136; the split is the one between
working memory and memory from before the task.

## 1. Question

Scored apart, do touches that only memory from before the episode
can help (cold) and touches that memory within the episode can help
(warm) separate what the current memory can do from what it cannot?

## 2. What changes

One component, the instrument (`tools/card136.1/reading.py`).

```
136:    500 episodes of 60 touches; returns to a recent object with probability 0.6;
        every touch of a child scored together
136.1:  1,000 episodes of 30 touches; returns with probability 0.4; rewiring every 300
        episodes; episodes 600-999 scored, store refreshed every 40
        cold touch = neither the child nor any of its parents seen yet in this episode
        warm touch = every other touch of a child
        a factor is scored in a part only if the observer has room there: its log loss
        at least 0.1 under the base rate's
```

Gap closed is as in card 136: from the base rate (0%) to the
observer, the exact best possible answer (100%). The memory read is
card 134.1's recall, records emptied at the start of each episode.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

As card 136; the split adds one assumption.

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Experience comes in episodes with a known start | days, deaths | yes | levels | tasks | built in; used only to score, and by the memory read here to empty its records |

## 3. Dependencies

Card 136 (world, observer), card 134.1 (recall).

## 4. Data check

Per child and world: 130 to 185 cold touches and 500 to 600 warm
ones in the scored episodes (card 136 had about 90 cold).

## 5. Feasibility gate

As card 136: observer against base rate, per factor and part.

Result (2026-11-13, `reading.py gate`, 70 seconds,
`runs/136.1/gate.json`): passed. The observer's probability is
within 0.4 points of the observed share. Cold, the observer has room
(base rate / observer, mean of three worlds) for carry 0.694 /
0.020, default 0.276 / 0.020, timer 0.141 / 0.020, rewired 0.695 /
0.568, two 0.591 / 0.481, chance 0.656 / 0.552; not for random
(0.099, just under) or the two drift children (0.002: nothing can be
known of them before the episode, which is right). Warm, it has room
for all nine (0.17 for fast drift up to 0.68).

## 6. Success criteria and prediction

The card tests the instrument.

1. **It agrees with what is known:** warm touches of the random
   child (the lock and drawer): the current recall closes at least
   60% of the gap.
2. **It separates:** among cold factors with room, the best and the
   worst differ by at least 40 points of gap closed.
3. **It is steady:** for at least four fifths of the scored entries
   (a factor in a part, with room), gap closed differs by at most 20
   points between the three worlds.

Prediction: 1 and 2 pass (default and timer near 100% cold, carry
near 0%); 3 is uncertain for the cold entries with little room
(chance, two, rewired).

## 7. Result

Run 2026-11-13, `tools/card136.1/reading.py` (2 minutes),
`runs/136.1/reading.json`, numbers in [results.json](results.json).
Gap closed by the current recall, mean of three worlds (the three
worlds in brackets); a dash where the observer has no room.

| Child | Cold | Warm |
|---|---|---|
| random | – | 92% (90, 92, 95) |
| carry | −4% (−5, −1, −5) | 99% |
| default | 100% | 100% |
| drift slow | – | 82% (83, 81, 81) |
| drift fast | – | 73% (62, 78, 80) |
| timer | 100% | 85% (85, 84, 87) |
| chance | – (room in two worlds: 4, 19) | 72% (75, 76, 64) |
| two | – (room in two worlds: −9, 4) | 76% (75, 75, 77) |
| rewired | −20% (−10, −45, −5) | 64% (61, 64, 66) |

1. **It agrees with what is known: pass.** 90 to 95% on warm touches
   of the random child (bar 60).
2. **It separates: pass.** Cold, default and timer are at 100%,
   carry at −4% and rewired at −20%: over 100 points apart (bar 40).
3. **It is steady: pass.** 12 of the 13 entries with room in all
   three worlds are within 20 points (bar four fifths); the one
   outside is rewired, cold (35 points, on about 130 touches a
   world).

What it says of the current memory: within an episode it is good to
very good everywhere (64 to 100%); the fixed facts of a world are
had from the store; what an earlier episode left behind is not had
at all, and where nothing has been seen its probability is no better
than the base rate, and for the rewired child worse.

## 8. Decision

**Keep.** The memory world scored cold and warm is the test bed for
the memory component. Next: memory that lasts from one episode to
the next (cold, carry: −4% against 100%).
