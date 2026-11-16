---
id: "139"
title: an answer that knows when not to trust itself — recall's confidence learned from its own past errors
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 139: an answer that knows when not to trust itself

Roadmap step 3, rung 3. [Card 138](../138-key-correction-on-the-memory-world/card.md)
left one failure across four factors: on cold touches (before the
child or its parents are seen in the episode) where little or
nothing can be known, recall's probability is worse than the base
rate. The user (2026-11-13): "yes our agent should understand
uncertainty".

Biological counterpart: the feeling of knowing. People judge, before
and apart from the answer itself, whether their memory is likely to
be right, and that judgement is learned from how retrieval has gone
before (metacognition; Fleming and Dolan 2012; confidence signals in
orbitofrontal cortex, Kepecs and others 2008).

## 1. Question

Can recall learn, from its own errors on stored episodes, in which
states of its records its answer is not to be trusted, and fall back
there to what happened in such cases? P4, and the fourth ingredient
(the most plausible state where it was not seen).

## 2. What changes

One component: the answer read from the neighbours. Records (never
emptied), key (prior x learned weights) and store are card 138's.

```
138:    p = share of the 25 nearest stored touches that changed the object
139:    on the store, each fifth of the episodes is answered from the others (as above).
        A read-out is fitted per touched object from that answer and the state of the
        records in the key (for each: seen at all, seen in this episode, time since seen)
        to what happened:
            logit p = a L + b + c.s + L d.s      L = logit of the share, s = the state items
        start: a = 1, the rest 0 (trust recall as it is); cost: 10 / (stored touches) x
        squared distance from the start
```

The read-out has 2 + 2 x (3 items per record in the key) numbers
per object, and sees no values, only how fresh the records are.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Recall's past answers can be scored against what happened | yes | yes | yes | yes | general |
| How fresh a record is can be read | needs tracking | near | yes | needs tracking | given with the record |
| Errors of the past resemble errors to come | mostly | mostly | per game | mostly | refitted every 80 episodes |

## 3. Dependencies

Cards 136.1 (test bed), 137 (records), 138 (key weights).

## 4. Data check

As card 136.1; per child and world 130 to 185 cold scored touches,
and about 500 cold stored touches to learn from.

## 5. Feasibility gate

- **Upper bound:** the observer. **Trivial baseline:** card 138's
  answer (k = 25).
- **Reachable reference where nothing can be known:** the base rate
  itself (random 0.694, drift slow 0.695, drift fast 0.695).
- A check of the first explanation: if the loss were the noise of
  25 samples, the share of the 200 nearest would remove it.

Result (2026-11-13, `tools/card139/sure.py gate`,
`runs/139/gate.json`): the room is real but small, and the first
explanation is wrong. Cold log loss, k = 25 against the base rate:
random 0.712 / 0.694, drift slow 0.721 / 0.695, drift fast 0.716 /
0.695, two 0.651 / 0.591, chance 0.663 / 0.656, rewired 0.729 /
0.695. With 200 neighbours cold touches do not improve (random
0.733) and warm ones collapse (random 100% → 67% of the gap), so the
error is a bias: stale items draw in stored touches from other
situations. The gate passes: a baseline measurably above a
reachable reference. Stated plainly, the prize is 0.02 to 0.06 of
log loss on about a quarter of the touches.

## 6. Success criteria and prediction

Three worlds, mean.

1. **Not fooled where nothing can be known:** cold log loss at most
   0.005 above the base rate for random, drift slow and drift fast,
   and at most 0.01 above it for two and chance.
2. **No harm:** every warm factor at most 1 point of gap closed
   under card 138's answer; cold carry at least 99%.
3. **Honest overall:** over all scored touches the stated
   probability within 1 point of the observed share (card 138's
   answer: 0.7).

Prediction: 1 passes for the three "nothing" children, uncertain
for two; 2 and 3 pass.

## 7. Result

Run 2026-11-13, `tools/card139/sure.py world=0|1|2` (75 seconds
each), `runs/139/sure.json`, numbers in [results.json](results.json).
Cold touches, log loss, mean of three worlds.

| Child | Base rate | Card 138's answer | With the read-out | Over the base rate |
|---|---|---|---|---|
| random | 0.694 | 0.712 | 0.704 | +0.010 |
| drift slow | 0.695 | 0.721 | 0.709 | +0.014 |
| drift fast | 0.695 | 0.716 | 0.698 | +0.003 |
| chance | 0.656 | 0.663 | 0.659 | +0.003 |
| two | 0.591 | 0.651 | 0.620 | +0.029 |
| rewired | 0.695 | 0.729 | 0.720 | +0.025 (no bar) |

1. **Not fooled where nothing can be known: fail.** Drift fast and
   chance are within their bars (0.005 and 0.01); random, drift slow
   and two are not. The read-out removes about half of the excess
   on every child.
2. **No harm: fail, narrowly.** Warm, fast drift is 1.4 points under
   card 138's answer (81.0 against 82.4%; allowed 1); the timer
   gains 3 (93 → 96%), the rest move by under 1. Cold carry 100%.
3. **Honest overall: pass.** Stated probability within 0.6 points of
   the observed share (0.7 before).

The read-out moves every cold number the right way and none far
enough. What is left is 0.003 to 0.03 of log loss on a quarter of
the touches.

## 8. Decision

**Stop.** The read-out is not adopted: it fails its bars, and the
prize left on this line is too small to chase on this test bed.
Recall's probability is already within 0.7 points of the observed
share overall. The uncertainty with a large measured cost here is
of another kind: knowledge that has gone out of date. On the
rewired child the memory closes 62% of the gap warm and is worse
than the base rate cold, the largest gap left on the memory world.
