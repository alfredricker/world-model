---
id: "137"
title: records that outlast the episode — an object's record is never emptied; the store says how long each one stays worth reading
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 137: records that outlast the episode

Roadmap step 3, rung 3. First card on the memory world of
[card 136.1](../136.1-memory-world-cold-and-warm/card.md), on the
first gap it named: what an earlier episode left behind (cold
touches of the "carry" child: −4% of the gap closed, where 100% is
the best possible). The user (2026-11-13) approved memory across
episodes as the first target.

Biological counterpart: episodic memory does not empty when a task
ends. A hippocampal trace of where the keys were left last night is
read this morning; whether it is still to be trusted depends on the
kind of object, which is learned (keys stay, a parked car may be
moved).

## 1. Question

If an object's record is simply never emptied, does recall read what
the last episode left behind where it still holds, and ignore it
where it does not, with no rule saying which objects carry over?
P4 (long-term and short-term memory as one store).

## 2. What changes

One component: the life of a record.

```
136.1:  record per object = seen flag, last seen value, whether the last touch changed it,
        time since seen; emptied at the start of every episode
137:    never emptied; time since seen runs on across episodes; one more item says
        whether the object was last seen in this episode
```

Recall, the choice of records by contrast, and the store are card
134.1's. Whether an old record is worth reading is not declared: for
the carry child's parent, stored touches with a like record had the
outcome the old value predicts; for a parent that starts at random,
they did not, and the neighbours' share falls to the base rate.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The same objects are met again in later episodes | yes (one map) | no (new map each time) | within a game | yes | built in; where it fails, old records find no like stored case and are not chosen |
| An object met again is known to be the same one | needs tracking and place | – | needs objects | needs tracking | given, declared |
| The agent knows when an episode starts | deaths, days | yes | levels | tasks | used for one item of the record only |

## 3. Dependencies

Cards 136 and 136.1 (world, observer, scoring), 134.1 (recall).

## 4. Data check

Card 136.1's: per child and world 130 to 185 cold and 500 to 600
warm touches in the scored episodes.

## 5. Feasibility gate

Upper bound: the observer. Trivial baseline: card 136.1's reading
(records emptied each episode). Both are measured: on cold touches
of the carry child, log loss 0.718 against 0.020, base rate 0.694.
The gate passes (room 0.67).

## 6. Success criteria and prediction

Three worlds, episodes 600 to 999, nothing trained.

1. **What the last episode left is read:** cold touches of the carry
   child, gap closed at least 70% in the mean of three worlds (now
   −4%).
2. **No harm to what worked:** warm, every factor at most 5 points
   under card 136.1's reading; cold, default and timer at least 95%.
3. **Old records do not mislead:** on cold touches of the three
   children of which nothing can be known before the episode
   (random, drift slow, drift fast), log loss at most 0.02 above
   card 136.1's reading (0.683, 0.727, 0.711).

Prediction: 1 passes; 2 is the risk (five items per record in place
of four, and old values in every key); 3 passes.

## 7. Result

Run 2026-11-13, `tools/card136.1/reading.py persist` (2 minutes),
`runs/136.1/persist.json`, numbers in [results.json](results.json).
Gap closed, mean of three worlds, records emptied each episode (card
136.1) → never emptied (this card).

| Child | Cold | Warm |
|---|---|---|
| carry | **−4 → 99%** | 99 → 98% |
| default | 100 → 100% | 100 → 97% |
| timer | 100 → 98% | 85 → 78% |
| random | – | 92 → 89% |
| drift slow | – | 82 → 83% |
| drift fast | – | 73 → 62% |
| chance | – | 72 → 72% |
| two | – | 76 → 72% |
| rewired | −20 → −40% | 64 → 59% |

1. **What the last episode left is read: pass.** 99% against a bar
   of 70 (98 to 99 in each world); log loss 0.030 against 0.020 for
   the observer and 0.718 before.
2. **No harm to what worked: fail.** Warm, fast drift loses 11
   points and the timer 7 (allowed 5); the others lose 0 to 5. Cold,
   default and timer hold (100 and 98%).
3. **Old records do not mislead: fail.** Cold touches of the random
   child: log loss 0.722 against 0.683 before (allowed 0.02 more;
   base rate 0.694). The two drift children are unchanged (0.722 and
   0.708 against 0.727 and 0.711).

So the record that outlasts the episode gives the whole of the
long-term factor it was meant for, with no rule about which objects
carry over, and costs a few points elsewhere. The likely cause (not tested
apart): the old value of a parent that does not carry over now sits
in the key at the same weight as a fresh one. Nearest stored cases
are then matched on a value that means nothing, which thins the
neighbours that matter (timer, fast drift) and gives false
confidence where nothing can be known (random, cold; rewired, cold).
One sign of it: for the fast-drift child the choice by contrast now
adds an unrelated switch beside the true parent in two of three
worlds.

## 8. Decision

**Revise.** Kept: records are never emptied. To revise: how much an
old item counts must depend on the object, learned from the store.
That is the key's learned correction of
[card 133.1](../133.1-key-prior-corrected-by-evidence/card.md),
which had no room on the lock and drawer and has a defined job here:
the equal-weight key as the prior, corrected per object and per age
of the record.
