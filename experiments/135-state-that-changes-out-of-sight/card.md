---
id: "135"
title: state that changes out of sight — a remembered value grows less trustworthy with time, and memory should know it
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 135: state that changes out of sight

Roadmap step 3, rung 3. Sixth card of the memory week, the "harder
memory test" the user asked for after
[card 134.1](../134.1-recall-keyed-per-object/card.md).

A check before drafting (2026-11-13) showed that longer delays alone
test nothing here. Accuracy does fall with the number of touches
since the lock was seen (98.4 at 1 touch, 89.3 at 8 or more), but
the remembered lock equals the true lock in all 1,408 pulls checked;
the errors are pulls that fail with the lock open (the slide at its
limit, or a miss), and the true-lock key falls the same way (98.2 to
88.4). Nothing in this scene changes unless the agent touches it, so
a delay of any length costs nothing.

Biological counterpart: a remembered location is trusted less the
longer ago it was seen when others share the space; animals and
people look again before acting on an old memory (information
seeking under uncertainty; Gottlieb and Oudeyer 2018).

## 1. Question

When objects can change out of the agent's sight, does memory know
how far to trust a remembered value, from stored episodes alone?
P4 (memory, short and long term), P12.

## 2. What changes

The test bed, not the model: a share P of the touches of every
recorded episode, drawn at random, are made by "another hand". They
happen (the recorded physics is untouched), but the agent does not
see them, does not store them, and its records are not updated. Only
the agent's own touches are stored and scored. Recall is card
134.1's.

```
134.1:  record per object = seen flag, last seen value, touched flag, last effect,
        "that touch changed it"
135:    the same + time since the object was last seen (log of 1 + touches elapsed,
        the other hand's included)
```

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near | yes | no | declared for the memory week |
| Things change out of sight | yes (mobs, night, growth) | yes | varies by game | yes (people, other robots) | made by hiding recorded touches |
| The agent knows how much time has passed | yes | yes | yes (steps) | yes | given |
| How fast things change is the same in stored and new episodes | roughly | roughly | per game | per place | the same P in store and test |

## 3. Dependencies

Cards 132 to 134.1. `tools/card135/drift.py`, `ceiling.py`.

## 4. Data check

With P = 0.25 the test has 1,032 pulls with the lock touched earlier
(104 do not move), 362 with the slide pulled earlier only, 240 with
nothing known and 138 reopenings; the store has 5,592 events. With
P = 0.5: 546, 273, 254 and 94; 3,741 stored events.

## 5. Feasibility gate

Run 2026-11-13 (`runs/135/gate.json`, `ceiling.json`; 5 seconds and
1 minute). Pulls with the lock touched earlier.

| | P = 0 | P = 0.25 | P = 0.5 |
|---|---|---|---|
| Remembered lock is wrong | 0% | 13.9% | 21.1% |
| ... at 1–2 touches since seen / at 6–10 | | 8% / 28% | 11% / 43% |
| Recall, card 134.1's record (baseline), balanced accuracy | 95.3 | 79.5 | 64.6 |
| True lock (upper bound) | 96.1 | 96.2 | 94.5 |
| Log loss: baseline / true lock / base rate | 0.078 / 0.054 / 0.312 | 0.212 / 0.055 / 0.328 | 0.230 / 0.066 / 0.296 |

So the problem exists. But the upper bound is not reachable from
what the agent knows. On the store alone, with a fifth of its
episodes held out in turn:

| P = 0.25 | Balanced accuracy | Log loss |
|---|---|---|
| Recall, plain record | 81.0 | 0.214 |
| Recall, + time since seen | 87.1 | 0.192 |
| Small trained network, plain record | 84.8 | 0.180 |
| Small trained network, + time since seen | 86.0 | 0.169 |

With P = 0.5 time since seen gives nothing (recall 63.5 to 67.3,
log loss 0.282 to 0.271; the network is worse with it). Knowing how
old a memory is closes about a seventh of the gap to the true lock
(0.214 to 0.192, against 0.055). The rest cannot be remembered; it
has to be looked at again.

**The gate as first drawn does not pass:** a bar near the true lock
could not be met by any memory. The card is to be redrawn with the
user before any criterion is set (section 6).

### The active form (the user, 2026-11-13: "let's try active")

```
look      before its own touch of object A the agent may look at the objects whose records
          recall uses for "A moves" (chosen by contrast, card 134.1: the drawer's lock for
          the drawer, the window's lock for the window). A look shows the true value and
          costs one look.
when      each time the agent sees an object again it stores (object, age of the memory,
          was the memory wrong). Before a touch, the estimate for a memory of this age is
          the share wrong among the 25 stored re-sightings of that object nearest in age;
          an object never seen counts as the oldest age. Look if the estimate is over 0.2.
store     the agent's own past under the same looking rule
```

Arms: never look (baseline); always look (upper bound, 1 look per
pull); looks at random, as many as the rule makes; the rule; an
oracle that looks exactly when the memory is wrong or missing
(reference, not available to the agent).

Gate for the active form (2026-11-13, `tools/card135/look.py gate`,
`runs/135/look_gate.json`, 40 seconds; store only, a fifth of its
episodes held out in turn, P = 0.25, balanced accuracy on all pulls):
never 80.8; always 95.7; oracle 95.2 with 0.18 looks per pull; the
rule 93.3 with 0.38 looks per pull; random looks at the same rate
88.3. **Passed:** 15 points between baseline and upper bound, and
the rule sits between them. The threshold 0.2 was chosen here, on
the store.

## 6. Success criteria and prediction

Test episodes, P = 0.25, store of 400 episodes; no training. Random
looks are the mean of 3 draws.

1. **Most of the loss recovered for under half the looks:** balanced
   accuracy on all pulls at least 91, with at most 0.45 looks per
   pull (never and always look are measured in the same run; 80.8
   and 95.7 at the gate).
2. **Memory chooses better than chance:** at least 3 points above
   random looks at the same rate, on all pulls (5.0 at the gate).
3. **Where a lock was seen, age decides:** in pulls with the lock
   touched earlier, at least 91 with at most 0.35 looks per pull
   (never 87.7, always 96.3, the rule 93.0 at 0.27 at the gate).

Prediction: 1 and 2 pass; 3 is close. P = 0.5 is reported without a
bar; there the choice of records is unsteady (card 134.1's limit),
so the agent does not always know which object to look at.

## 7. Result

Run 2026-11-13, `tools/card135/look.py` (20 seconds),
`runs/135/look.json`, numbers in [results.json](results.json). Test
episodes, P = 0.25, balanced accuracy.

| Looking rule | Looks per pull | All pulls | Lock touched earlier (looks) |
|---|---|---|---|
| Never | 0 | 80.7 | 85.2 |
| Random, as many as the rule | 0.37 | 83.8 | 86.3 (0.38) |
| **By the age of the memory** | **0.38** | **91.7** | **89.9 (0.25)** |
| Always | 1 | 95.5 | 95.9 |
| Oracle (looks only when wrong or missing) | 0.19 | 95.0 | 95.3 (0.07) |

1. **Most of the loss recovered for under half the looks: pass.**
   91.7 against a bar of 91, with 0.38 looks per pull (allowed
   0.45): 74% of the way from never to always.
2. **Memory chooses better than chance: pass.** 7.9 points above
   random looks at the same rate (bar 3). The log loss of the stated
   probability falls from 0.223 to 0.108 (random 0.194, always
   0.065).
3. **Where a lock was seen, age decides: fail.** 89.9 against a bar
   of 91, at 0.25 looks per pull (never 85.2, always 95.9).

Where the gain comes from: every pull whose lock was never seen gets
a look (96.6 against 58.3), which no random rule does; that is most
of criterion 2. Among locks already seen, age is a weak sign of a
wrong memory: the oracle reaches 95.3 with 0.07 looks per pull, the
age rule 89.9 with 0.25.

A check afterwards (not a declared arm, `posthoc.py`): the estimate
from 100 nearest re-sightings is steadier (7% wrong at age 1 rising
to 30% at age 12; with 25 it was uneven, since ages are whole
numbers) but traces the same curve (90.2 at 0.27 looks). The limit
is what age can tell, not how it is estimated.

P = 0.5, no bar: never 72.6, the rule 83.7 with 0.29 looks per pull,
always 84.5 with 0.51. With half of the touches hidden the choice by
contrast found the drawer's lock but no lock for the window, so the
agent never looked before a window pull; this is card 134.1's limit
(conditions need enough clean stored cases) and it caps everything
here. Random looks in that row were drawn at half the rule's rate,
so they are not compared.

Also kept from the gates: a delay alone costs this memory nothing,
and passive memory (time since seen as an item of the record)
recovers about a seventh of the loss.

## 8. Decision

**Revise.** Kept as parts: the test bed with hidden touches; a look
directed at the condition objects of the intended touch; a store
made of the agent's own past under the same rule; "never seen means
look". To revise: what the decision to look reads. Age alone leaves
a wide gap to the oracle (0.25 against 0.07 looks). For the user:
candidates are the price of being wrong against the price of a look
(a failed pull is itself a look at the lock), and how active the
other hand has been.
