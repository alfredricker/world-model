---
id: "134"
title: unseen state from effects — what the agent's own touches did is remembered beside what it saw, and recall answers with a probability
rung: 3
serves: [P4, P12, P21]
status: done
verdict: fail
arch_version: og4
date: 2026-11-12
---

# 134: unseen state from effects

Roadmap step 3, rung 3. Fourth card of the memory week (true visible
state, narrow view), and the first on the fourth ingredient: the
most plausible state where nothing was seen.
[Card 133.1](../133.1-key-prior-corrected-by-evidence/card.md) left
one tenth of the pulls unanswered: those made before the lock was
seen (prior 63.8%, true lock 98.2%).

Two checks before drafting (2026-11-12, under a minute each) changed
the shape of this card.

1. **Where nothing at all is known, there is nothing to find.**
   Locks start unlocked in half of the episodes, at random. On the
   223 test pulls made with the lock unseen (52 do not move), a
   classifier given everything observable scores 60.1% balanced
   accuracy (the mean of the accuracy on each outcome, 50% by
   chance), and its log loss (the penalty for the probability it
   gave to what happened; lower is better) is 0.521 against 0.544
   for always saying the base rate, "77% move". The equal-weight key
   of card 133.1 is already at 63.8. The right answer here is a
   probability, not a better guess.
2. **Where an effect was seen, the state can be inferred.** Card
   133's view showed a slide's lock whenever the slide was touched,
   which is generous. If a lock is seen only when the lock itself is
   touched, the 2,226 test pulls split three ways: lock touched
   earlier, 1,534 (143 do not move); lock never touched but the
   slide pulled earlier, 464 (27 do not move); nothing known, 228
   (52 do not move). In the middle group "the same as the last pull
   of this slide" is right 90.3% of the time (balanced).

Biological counterpart: a door that did not open is taken to be
locked without the bolt being seen. Inferring a hidden cause from
an outcome is latent cause inference; the orbitofrontal cortex is
thought to hold such unobservable task states (Wilson and others
2014), and confidence in them is reported separately from the
choice.

## 1. Question

If the memory of an episode holds, beside the last seen value of
each object, the last effect of touching it, does recall over
stored episodes answer pulls whose lock was never seen, without any
rule that names the lock? And does recall's answer carry a usable
probability, close to the base rate where nothing is known and
confident where something is? P4 (memory), P12 (conditions held in
short-term memory).

## 2. What changes

One component: what the memory of the current episode holds. The
key (equal weight per item, scaled to unit spread over the store),
the recall and the scored questions are card 133.1's; no training.

```
view      a lock is seen only when the lock itself is touched (card 133: also when its
          slide is touched)
133.1:    contents = contact point, touched object, last seen value of each object
134:      contents = the same + for each object, the effect of the last touch on it
          (how far it moved or whether it flipped), with a "touched before" flag
answer    pull moves or not: the median effect of the 5 nearest stored events (as before)
          probability that it moves: the share of the 25 nearest stored events in which
          the slide moved
```

Nothing says "a failed pull means locked". A stored pull that
followed a failed pull of the same slide lies near the present one
in the key, and it mostly failed too.

Arms: contents of 133.1 (baseline); contents of 134; "same as the
last pull" (a reference for the middle group only); the true lock
in the key (upper bound).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near | yes | no | declared for the memory week |
| A touch has an effect that can be measured | holds (block breaks or not) | holds | holds (frame changes or not) | holds | general |
| The touched object is identified | needs tracking | near | needs objects | needs tracking | given by perception, declared |
| A hidden cause stays the same until acted on | mostly; night and mobs change things unprompted | the same | varies by game | mostly | not assumed: recall over stored episodes says how long an effect stays predictive |
| Memory contents are a list of named values | no | no | no | no | declared, as in 133.1 |

## 3. Dependencies

- Cards 132 to 133.1: test bed, effect recall, the equal-weight key.
- [Episode history in the recall key](../../graph/principle/episode-history-in-the-recall-key.md),
  untested in the graph until now.
- A probability from the share of neighbours is the k-nearest-
  neighbour estimate; no new part.

## 4. Data check

Done above: three groups of test pulls (1,534, 464 and 228). The
middle group has 27 pulls that do not move, so one of them is worth
1.9 points of balanced accuracy there; scores in that group are
read with that in mind. Store of 400 episodes; 100 reported without
a bar.

## 5. Feasibility gate

- **Upper bound:** the true lock in the key, per group; and "same
  as the last pull" in the middle group (90.3).
- **Trivial baseline:** contents of 133.1 under the stricter view,
  per group; the base rate's log loss in the "nothing known" group
  (0.544).
- The gate passes if, in the middle group, the baseline is at least
  10 points under the upper bound. If it is not, the stricter view
  has not made the problem and the card is redrawn.

Result of the gate (2026-11-12): passed. In the middle group the
contents of card 133.1 score 50.2 (chance) and the true lock 91.9;
"same as the last pull" scores 90.3. The base rate's log loss in
the "nothing known" group is 0.538 (79% of such stored pulls move).

## 6. Success criteria and prediction

Test episodes, store of 400 episodes, no training and so no seeds.
Proposed, to be fixed with the user.

1. **Inferred from an effect:** in the 464 pulls with the lock
   never touched and the slide pulled earlier, balanced accuracy at
   least 88 (reference rule 90.3; the baseline is measured at the
   gate).
2. **No harm where the lock was seen:** in the 1,534 pulls with the
   lock touched earlier, at most 1 point under the baseline; and on
   reopenings at most 2 points under it.
3. **Honest where nothing is known:** in the 228 pulls with nothing
   known, the log loss of the probability at most 0.56 (base rate
   0.544; a confident wrong guesser scored 0.774 in the check), and
   over all pulls the stated probability within 5 points of the
   observed share, averaged over ten bins of probability.

Prediction: 1 passes narrowly, 2 passes, 3 is the uncertain one,
since nearest neighbours in a 33-item key may be too few and too
far to give a steady share.

## 7. Result

Run 2026-11-12, `tools/card134/effects.py`, `runs/134/effects.json`,
numbers in [results.json](results.json). Store of 400 episodes,
balanced accuracy on the test pulls by what the agent knows of the
lock.

| Key | Lock touched earlier (1,534) | Slide pulled earlier only (464) | Nothing (228) | Reopenings (198) |
|---|---|---|---|---|
| Last seen values (card 133.1) | 88.5 | 50.2 | 68.4 | 93.4 |
| + last effect per object (this card) | 86.3 | 57.9 | 61.3 | 93.4 |
| "Same as the last pull" | | 90.3 | | |
| True lock | 95.9 | 91.9 | 98.2 | 88.1 |

Probability from the 25 nearest events: in the "nothing known"
group the log loss is 0.580 with the new contents and 0.492 with
the old, against 0.538 for the base rate; over all pulls the stated
probability is within 2.1 points of the observed share (old
contents 1.7).

1. **Inferred from an effect: fail.** 57.9 against a bar of 88.
2. **No harm where the lock was seen: fail.** 2.2 points under the
   baseline (86.3 against 88.5; allowed 1). Reopenings unchanged.
3. **Honest where nothing is known: fail on the first half.** Log
   loss 0.580 against a bar of 0.56; the calibration half passes
   (2.1 against 5).

With 100 episodes (3 draws) the new contents are worse than the old
in every group (middle group 48.4 against 49.6, reopenings 83.5
against 93.7).

One part of criterion 3 holds without this card's change: with the
old contents, the share of 25 neighbours is an honest probability
(log loss 0.492, under the base rate's 0.538; gap 1.7 points).

**Why it failed, from three checks run afterwards** (not declared
arms; `tools/card134/own.py`, `learned.py`).

- The content is in the memory. The new item for the slide's last
  effect agrees with the last pull's outcome in every case, and a
  rule reading that one item scores 90.3.
- Recall does not read it. The item is one of 33 at equal weight,
  and "did not move" lies close to "moved a little" in distance.
  Card 133.1's learned correction does not find it either (55.0 in
  the middle group, mean of 3 seeds, 47.5 to 61.6): the pulls that
  decide are about 35 of 7,400 stored events and carry almost no
  weight in a mean error.
- A key made only of the touched object's own record (its identity,
  its last seen value, its last effect, and the two locks as last
  seen) lifts pulls with the lock touched earlier from 88.5 to 95.0,
  beside 95.9 for the true lock, with nothing trained. It reaches
  only 61.2 in the middle group, and it breaks reopenings (59.4),
  which depend on another object's record, the cube's.

Read together with cards 132 to 133.1: each time the memory held
the right content and one long key over everything read it badly;
adding content made recall worse. What helped most was structure:
compare the touched object's own record first.

## 8. Decision

**Revise.** The contents (last effect per object) and the
probability from neighbours are kept as parts; the flat key is what
fails. Next card, for the user to approve: recall keyed on the
touched object's own record, plus the records of the other objects
that stored outcomes show to matter for it (the lock for a slide,
the cube for a reopening), which is the condition found by contrast
in card 127 used as the key.
