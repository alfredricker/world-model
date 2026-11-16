---
id: "138"
title: the key's learned correction on the memory world — how much an old item counts, learned per object from the store
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 138: the key's learned correction on the memory world

Roadmap step 3, rung 3. [Card 137](../137-records-that-outlast-the-episode/card.md)
kept every object's record across episodes: what the last episode
left is read (−4 → 99% of the gap), but warm touches lost 7 to 11
points on two factors and cold touches of the random child became
falsely confident. The user (2026-11-13) approved the learned key
correction of [card 133.1](../133.1-key-prior-corrected-by-evidence/card.md)
as the next step; on the lock and drawer it had no room.

Biological counterpart: what is retrieved depends on learned
attention to the features of a cue; features that have not
predicted outcomes come to be ignored (learned attention in
category learning, Kruschke 2001; selective attention in
reinforcement learning, Niv and others 2015).

## 1. Question

Can a correction to the key, trained only on how well stored
touches recall one another, learn for each object whether an old
value still counts, and so keep card 137's gain without its cost?
P4.

## 2. What changes

One component: the key, from fixed to prior plus learned correction.

```
137:      key(touched A) = A's record + the records chosen by contrast; items scaled to
          unit spread, equal weight
138:      the same key is the prior
weights   prior x exp(delta), one delta per item, starting at 0
network   those weights, plus 8 more dimensions from a small network over the key's items,
          starting near 0 (the declared arm: "an old value counts only if..." is a
          combination of items, which one weight per item cannot say)
cost      beta / (number of stored touches) x size of the correction, added to the recall
          error: with few touches the key stays at the prior
training  store only, per touched object: each stored touch's outcome is recalled from
          the touches of other episodes; error = log loss relative to the prior's
beta      one number per world and arm, chosen among 0.1, 1, 10, 100 on a fifth of the
          stored episodes held out, at the first block
```

Records are never emptied (card 137). Scoring is card 136.1's. One
training seed per world; the three worlds are the repeats.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A record is a short list of named items | items would be learned vectors; a weight per dimension and a network over them are still defined | the same | the same | the same | declared |
| There are enough stored touches of an object to train on | holds for common objects | holds | few per game: the cost keeps the key at the prior | holds | 1,000 to 1,700 per object |
| The same correction holds for a while | mostly | mostly | per game | mostly | refitted every 80 episodes |

## 3. Dependencies

Cards 136.1 (test bed), 137 (records), 134.1 (recall), 133.1 (the
correction and its cost).

## 4. Data check

As card 136.1. The store holds 600 to 1,000 episodes; each object
has 1,000 to 1,700 stored touches.

## 5. Feasibility gate

- **Upper bound:** the observer. **Trivial baseline:** card 137
  (the prior).
- **Reference, a key that knows which old values count:** every
  record but the carry switch's is emptied at the start of each
  episode. It uses the world's truth, so it is a reference and not
  an arm. The gate passes if it is at least 5 points above the prior
  on warm touches of the timer and fast-drift children with cold
  carry kept: then the cost of card 137 has the cause it was given,
  and a better key exists.

Result (2026-11-13, `tools/card138/correct.py gate`, 100 seconds,
`runs/138/gate.json`): passed. Gap closed, prior → reference: fast
drift warm 62 → 76%, timer warm 78 → 85%, two warm 72 → 78%, rewired
warm 59 → 65%; cold carry 99 → 100%; cold random log loss 0.722 →
0.669 (base rate 0.694).

## 6. Success criteria and prediction

Three worlds, the network arm.

1. **The cost is won back:** warm, fast drift at least 68% and timer
   at least 80% of the gap closed (prior 62 and 78; reference 76 and
   85).
2. **The gain is kept and old records do not mislead:** cold carry
   at least 95%; cold touches of the random child, log loss at most
   0.70 (prior 0.722, reference 0.669, base rate 0.694).
3. **Never under the prior:** every warm factor at most 2 points
   under the prior.

Prediction: the weights arm changes little. The network arm passes
3; 1 and 2 are uncertain, since card 134's check found that a mean
error gives rare deciding cases little weight.

## 7. Result

Run 2026-11-13, `tools/card138/correct.py world=0|1|2` (75 seconds
each), `runs/138/correct.json`, numbers in
[results.json](results.json). Warm touches, gap closed, mean of
three worlds.

| Child | Prior (card 137) | Weights | Network (declared) | Reference key |
|---|---|---|---|---|
| random | 89% | **100%** | 100% | 92% |
| carry | 98% | 100% | 99% | 99% |
| default | 97% | 100% | 99% | 100% |
| drift slow | 83% | **91%** | 88% | 81% |
| drift fast | 62% | **82%** | 56% | 76% |
| timer | 78% | **93%** | 94% | 85% |
| chance | 73% | 75% | 70% | 71% |
| two | 73% | 80% | **85%** | 78% |
| rewired | 59% | 62% | 63% | 65% |

Cold: carry 99% → 100% in both arms. Random child, log loss: prior
0.722, weights 0.712, network 0.716 (base rate 0.694).

The declared arm, the network:

1. **The cost is won back: fail.** Timer 94% (bar 80), but fast
   drift 56% (bar 68; the prior has 62). In the three worlds: 46,
   36 and 88.
2. **Gain kept, old records do not mislead: fail.** Cold carry 100%
   (bar 95); cold random 0.716 (bar 0.70).
3. **Never under the prior: fail.** Fast drift 5 points under,
   chance 3 (allowed 2).

The arm that was not declared, one weight per item, does what the
card asked on warm touches: every factor at or above the prior, fast
drift 62 → 82% and timer 78 → 93%, above the reference key that
knows the truth on six of nine factors. On the declared bars it
would pass 1 and 3 and fail 2 (cold random 0.712). My prediction was
the reverse: "an old value counts less" turned out to need no
combination of items. A likely reason, not checked (the learned
weights were not inspected): a larger weight on the item "seen in
this episode" already pulls fresh and stale records apart.

Why the network is worse where it is worse: on stored episodes held
out, its error was above the weights' at every beta and fell all
the way to the largest one tried (100), so the grid stopped before
the cost was high enough; it fits the store and is unsteady between
worlds.

What neither arm touches: cold touches where nothing can be known
or little (random 0.712, chance 0.663, two 0.651, rewired 0.729
with weights) stay at or above the base rate (0.694, 0.656, 0.591,
0.695). The key decides which stored cases are near, not how sure
the answer should be when no stored case is like this one.

## 8. Decision

**Revise.** Kept: the weights correction with its cost, on records
that are never emptied (the best memory so far on this test bed:
62 to 100% warm, 100% on what the last episode left). Dropped for
now: the network arm. To revise: cold touches where nothing is
known, which should fall back to the base rate, and the rewired
child (recent stored episodes over old ones).
