---
id: "133.1"
title: the recall key as a prior corrected by evidence — every remembered item counts equally until stored outcomes say otherwise
rung: 3
serves: [P4, P12, P21]
status: done
verdict: fail
arch_version: og4
date: 2026-11-12
---

# 133.1: the key as a prior corrected by evidence

Roadmap step 3, rung 3. Third card of the memory week (true visible
state, narrow view). [Card 133](../133-key-learned-from-recall-outcome/card.md) showed
that a key trained on the error of the recalled effect finds what
matters by itself (pulls 76.7% to 87.5%, truth 96.0%), and failed
twice: with 100 episodes the learned network was worse than the
fixed key (reopenings 82.8 against 93.1), and the simpler form's
weights grew without bound. Both are the same fault: nothing tied
the learned key to a starting point, so with little evidence it
wandered.

Biological counterpart: attention starts broad and narrows with
experience. Learners weight all features of a remembered case alike
until outcomes single some out (Kruschke 1992); a strong prior that
few observations cannot overturn is the Bayesian reading.

## 1. Question

Does a learned key that starts as "every remembered item counts
equally" and may leave that prior only in proportion to the stored
evidence keep the gains of card 133 at 400 episodes and stop losing
to the fixed key at 100 and 30? P4 (memory).

## 2. What changes

One component: how the learned key is tied to a prior.

```
133:    key = learned map of the contents, free; trained on the error of the recalled effect
133.1:  key = prior map + learned correction
        prior       each item of the contents scaled to unit spread over the store, equal
                    weight; no hand-set number (card 132's key had three)
        correction  arm 1: a change to each item's weight; arm 2: extra dimensions from a
                    small network, starting at zero
        cost of the correction = beta / (number of stored events) x its size, added to the
                    error of the recalled effect: with few events the key stays at the
                    prior, with many it is free. This also bounds the scale.
        beta        one number for both arms, chosen among 0.1, 1 and 10 on stored episodes
                    held out from training (never on the test episodes)
```

This is version 27's rule (own evidence against a prior, weighted
by counts) and the user's "one prior, corrected by experience"
(2026-10-11), applied to the key. The contents, the recall (5
nearest, median effect), the narrow view and the scored questions
are card 133's.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near | yes | no | declared for the memory week |
| "Equal weight per item" is a usable prior | items would be learned vectors; equal weight per dimension still defined | the same | the same | the same | general; measured at the gate against card 132's hand-set key |
| Memory contents are a list of named values | no | no | no | no | declared; arm 2 takes any vector |
| Evidence is counted in stored events | holds | holds | few events per game | holds | general |

## 3. Dependencies

- Cards 132 and 133: test bed, contents, effect recall, training.
- A penalty that shrinks with the amount of data is the standard
  form of a prior on parameters (maximum a posteriori estimation).
- Version 27's blend of own counts with a prior (ARCHITECTURE,
  MiniGrid line); card 130 applied it to tries.

## 4. Data check

As card 133: 2,226 test pulls (222 do not move), 198 reopenings (40
with the cube inside, all with the cube seen earlier under the
view's rule). Stores of 30, 100 and 400 episodes hold about 560,
1,900 and 7,400 events.

## 5. Feasibility gate

- **Upper bound:** from card 133: 96.0 on pulls (true values) and
  96.5 on reopenings (hand-set key), at 400 episodes.
- **Trivial baseline:** the prior alone (equal weights), at 30, 100
  and 400 episodes, beside card 132's hand-set key (76.7 and 96.5 at
  400; 71.5 and 93.1 at 100). If the prior alone is more than 10
  points below the hand-set key on either event, that is reported to
  the user before the main run, since the card would then be
  measuring the prior and not the correction.

Result of the gate (2026-11-12, mean of 3 draws of episodes):
passed. The equal-weight prior is above the hand-set key on pulls
(84.9 against 76.7 at 400 episodes, 76.5 against 67.9 at 100, 64.7
against 55.4 at 30) and within 4 points of it on reopenings (93.1
against 96.5 at 400).

## 6. Success criteria and prediction

Balanced accuracy on the test episodes, the mean of 3 seeds (card
133 used one, and its margins were under a point). The better arm
is judged, the same arm for all three criteria. Proposed, to be
fixed with the user.

1. **Keeps the gain at 400 episodes:** pulls at least 88 and
   reopenings at least 95 (card 133's bar, unchanged; it scored 87.5
   and 93.2).
2. **Holds at 100 episodes:** pulls at least 85 and reopenings at
   least 93 (card 133: 80.8 and 93.1 for the weights, 73.3 and 82.8
   for the network).
3. **Never worse than its prior:** at 30, 100 and 400 episodes, at
   most 2 points below the prior alone on either event.

Reported without a bar: the right-reason test of card 133 (lock and
cube shuffled), the chosen beta, each arm at each store size.

**Prediction.** 3 passes: it is what the penalty is for. 2 is the
real test; the weights arm reached 80.8 unbounded, and a bound
should help more than it costs. 1 is uncertain on reopenings, where
the learned key lost 3 points to the hand-set one.

**Budget.** 3 store sizes, 2 arms, 3 seeds, 3 values of beta: 54
short trainings of about a minute, in runs of under 10 minutes each,
capped. No rendering.

## 7. Result

Run 2026-11-12 (`tools/card133.1/prior.py`, about half a minute per
store size, capped; numbers in [results.json](results.json)).
Balanced accuracy on the test episodes, mean of 3 seeds. The network
is the better arm and is the one judged.

| Stored episodes | Prior alone: pulls, reopenings | Weights: pulls, reopenings | Network: pulls, reopenings |
|---|---|---|---|
| 30 | 64.7, 94.6 | 64.9, 79.8 | 64.6, 94.6 |
| 100 | 76.5, 89.9 | 73.4, 82.1 | 76.0, 90.0 |
| 400 | 84.9, 93.5 | 87.3, 93.2 | 87.9, 93.1 |

| Criterion | Bar | Result (network) | Verdict |
|---|---|---|---|
| 1. Keeps the gain at 400 | pulls at least 88, reopenings at least 95 | 87.9 and 93.1 | fail (by 0.1 and 1.9) |
| 2. Holds at 100 | pulls at least 85, reopenings at least 93 | 76.0 and 90.0 | fail |
| 3. Never worse than its prior | at most 2 points below, at 30, 100, 400 | worst case 0.5 below | pass |

- **Tied to the prior, the network no longer loses to it** at any
  store size (card 133's free network was 10 points below the fixed
  key on reopenings at 100 episodes). The per-item weights still do:
  14.8 below on reopenings at 30 episodes.
- **The correction only pays at 400 episodes** (pulls 84.9 to 87.9).
  At 30 and 100 the chosen penalty holds the key at the prior, so
  the 100-episode bar is missed by the prior's own margin.
- **The plain prior is the largest gain of the card:** equal weight
  on items scaled to unit spread beats card 132's hand-set weights
  by 8 to 9 points on pulls at every store size.
- **The bar on pulls was set wrong (found after the run).** In 223
  of the 2,226 pulls the lock had not been seen in the episode, and
  52 of the 222 failed pulls are among them. No memory of this
  episode can answer those; all keys score about 63 there, the truth
  98.2. On the 2,003 pulls where the lock had been seen:

| Key, 400 episodes | Pulls, lock seen earlier | Pulls, lock not seen |
|---|---|---|
| Prior alone | 89.4 | 63.8 |
| Network (3 seeds) | 93.2 (91.8 to 94.1) | 61.3 |
| True values | 95.3 | 98.2 |

  Card 132's decision already said to score these apart; card 133
  and this card did so for reopenings and not for pulls.
- Right reason, network at 400: lock shuffled 52.2, cube shuffled
  86.4 against 87.9.

Changes made during the run:
- beta was chosen among 0.1, 1, 10, 100 and 1,000 (the card named
  the first three; the unit of "size" was not fixed there). Chosen:
  network 1,000, 10 and 0.1 at 30, 100 and 400 episodes.
- The network's new dimensions start near zero, not at zero: at
  exactly zero they receive no gradient, and the first run returned
  the prior unchanged. Rerun with that fixed; the weights arm's
  numbers did not change.
- The split by "lock seen" was computed after the run.

## 8. Decision

**Revise.** Two of three bars fail as declared. What stands: the key
is the equal-weight prior plus a network correction priced by the
evidence; it is never worse than the prior, and where memory can
answer at all it reaches 93.2 on pulls against 95.3 with the true
values. What remains open is not the key. (1) With 100 episodes the
correction adds nothing. (2) A tenth of pulls and a quarter of the
failed ones come before the lock was seen: that is the prior over
likely states (the user's fourth ingredient), not recall. The score
of the test bed is to be split by "seen earlier or not" for every
event from the next card on.
