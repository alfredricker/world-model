---
id: "158"
title: events as changes of an object's latent state, and their conditions
rung: 2
serves: [P7, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-21
---

# 158: events as changes of an object's latent state, and their conditions

Roadmap step 2, rung 2, on the direction under trial. The energy modules
have objects and what one action does to them (cards 147 to
[157](../157-transitions-read-from-the-binding-layers-own-states/card.md)),
but no events: nothing in an object module lasts from one step to the
next, so the model cannot say "it is held now", when that began, or what
made it so. Card 157 dropped "held" because no frame shows it. The user,
2026-11-21: "do we have event segmentation? Could our agent understand
goals by pixel transitions?", and "yes" to this card. Runs take under
ten minutes each. Biological counterpart: event segmentation theory
(Zacks): a model of the current event is kept until prediction fails,
and is replaced at the boundary, with hippocampal and prefrontal
responses there; the inference of a hidden cause that persists (Gershman
and Niv; Franklin and others 2020).

**Terms**, beyond cards 153.1 to 157. The **latent state** $h_t$ of an
object module at step $t$: a discrete variable that is not observed and
that persists from step to step. An **event** is a step at which the
most probable sequence of latent states changes value; that step is its
**boundary**. The **conditions** of an event are the tokens of that step
on which the change depends (GOAL P12: a statement is an effect and the
conditions it needs). Own tokens, dynamics class $z_i$: as card 157.

## 1. Question

Does a latent state kept in each object module, inferred from the
module's own tokens, place the events (taken, released) at the steps
where they happen, including when nothing moves at that step; does it
answer the question card 157 had to drop (the action that leaves an
object held without moving it); and can the conditions of each event be
read off its learned transition? P7 (events as learned units), P12
(conditions of an effect).

## 2. What changes

One component, the latent state of an object module
(`tools/card158/events.py`). Binding, appearance couplings, own tokens
and dynamics classes are card 157's, unchanged.

```
before (157)   a transition is one step: (action, contact) -> displacement; nothing lasts from step to step
after          each object module keeps h_t in {0, 1}; which value means what is learned
  emission     p(came along_t | h_t, the hand moved_t)
  transition   p(h_t | h_{t-1}, contact_t, grip_t, the hand moved_t; class z), pairwise couplings between h_t and each
               of the four tokens, kept sparse; a shared set plus each dynamics class's difference
  learning     expectation-maximisation over each object's sequence of own tokens in an episode (forward-backward)
  an event     a step at which the most probable sequence of states (Viterbi) changes
  conditions   of an event: start from the likeliest situation for it; a token is a condition if changing it alone
               makes the event improbable (under 0.5)
  an action    for a wanted state: the grip and move that make it most probable under the belief over classes
```

Declared priors: two values of $h$; the conditions of a change are
tokens of the same step. The state that makes "came along" probable when
the hand moves is the one the take test asks for.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| An object's latent state has two values | no | no | no | no | declared: the number must be learned (as the classes are, card 153.1) |
| A state shows in the object's motion relative to the hand | partly | no | no | partly | declared: states that show otherwise (a door open, an inventory) need other emissions |
| The conditions of a change are tokens of the same step | partly | yes | yes | partly | declared |
| Conditions are facts about the object and the hand only | no | no | no | no | declared: relations between objects are a later card |
| Own tokens as card 157 (contact as boxes, whole-pixel shifts) | no | partly | partly | no | declared, as card 157 |

## 3. Dependencies

Card 157 (own tokens, the dynamics classes fitted on them), 156 (the
agent, appearance couplings), 154, 153.1, 152 (the take test).

## 4. Data check

The events to find must include ones that no frame shows at once. Card
152's script grips without moving in half of its tries and lets go
without moving; card 156's agent always moves as it grips (0 of 103
takes without a move). The criteria are therefore judged on the
script's experience (100 episodes; binding, appearance couplings and own
tokens learned from it as in cards 156 and 157). The agent's experience
(saved by card 157) is fitted too and reported.

## 5. Feasibility gate

On the development worlds (seed 0, by shape and by colour), the same
latent-state model given the simulator's contact and displacement and
each object's true behaviour:

- boundaries: F1 at least 85%, and the baseline (below) under 60%;
- take test: at least 90%; one class for all, under 60%.

**Passed, with two departures from the letter** (by shape, by colour).
Boundaries: F1 99.0 and 99.3 with a sparsity penalty of 5 nats per unit
of coupling, 80.9 and 85.3 with the 1 nat used for the method; the
baseline 58.0 and 63.7 (one world over 60). Take test: 100.0 and 100.0;
one class for all 40.7 and 37.8.

**Development** (seed 0, the script's experience). Three things were
changed after the first version, and are part of the method as tested:

- The sparsity penalty is counted in nats over the whole experience, not
  per step: per step it removed every coupling of the rare events (the
  model answered "cannot" for all movable objects). Its size, 1 nat, was
  chosen among 1, 5 and 20 by the sum of F1 and take over the two
  development worlds. With the simulator's tokens 5 and 20 are better:
  at 1 nat the model uses "the hand moved" to place releases.
- The conditions of an event are read starting from the situation in
  which the event most often happened, and only among situations the
  experience produced (LESSONS: check a condition only in a situation
  the learned effects produce). Starting from the likeliest situation
  named nothing for "released", because two reasons held at once.
- "Cannot" is answered when that is more probable than the best
  action's success, not when the success is under 0.5.

Development figures with own tokens (by shape, by colour): F1 67.5 and
78.2 (baseline 52.3 and 53.8); take 55.9 and 99.7 (100.0 and 100.0 with
the true behaviours given); conditions right 2 of 4 and 4 of 4. In the
world by shape the never-met sticky objects are believed fixed from
their appearance (0.60 against 0.24): card 157's weakness, not this
component.

## 6. Success criteria and prediction

Written before any number of this card existed. Method: the script's
experience, own tokens, card 157's classes. Test worlds: seeds 1 to 5,
by shape and by colour; one run per world.

1. **Events.** Boundaries of the latent state against the simulator's
   taken and released steps (same object, same step, same direction; for
   objects the model holds a pattern for, from the step after the
   pattern's first): F1 at least 70.0% on the mean of the 10 worlds, and
   at least 15.0 points above the baseline that reads the state from
   what is seen (the last value of "came along" at a step in which the
   hand moved).
2. **The action for a state.** Card 152's take test (the action that
   leaves a never-met object held, scored in the simulator; "cannot" for
   a fixed object): right for at least 80.0% on the mean.
3. **Conditions.** For takeable and for sticky objects, the conditions
   the model names for "taken" and for "released" (40 statements in 10
   worlds) are exactly the world's (taken: contact and the right grip;
   released: the other grip) in at least 32 of 40.

Prediction: 1 passes (about 75 against 45); 2 is near its bar; 3 fails
(the own contact token misses 3 contacts in 10, most while an object is
carried, so "contact" will be a weak condition). On the agent's
experience the take test fails: no take without a move was ever seen.
Budget: about 90 minutes of runs, none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). Ten worlds (seeds 1 to 5, by
shape and by colour), one run each. "F1": the model's events against the
simulator's taken and released steps, same object, same step, same
direction. "Take": card 152's test, the action that leaves a never-met
object held, scored in the simulator.

| The script's experience (the criteria), mean of 10 worlds | Events, F1 % | Take right, % |
|---|---|---|
| Simulator's tokens, true behaviours (bound; penalty 5 nats) | 99.7 | 100.0 |
| Own tokens, true behaviours given | 70.0 | 100.0 |
| **Own tokens, latent state, card 157's classes** | **71.3** (lowest world 65.8) | **78.8** (by colour 99.9, by shape 57.6) |
| Own tokens, one class for all | 56.4 | 39.1 |
| Own tokens, the state read from what is seen (baseline) | 53.5 | |

Conditions named for "taken" and "released", takeable and sticky
objects: 30 of 40 statements exactly the world's. Of the 10 wrong, 9 add
"the hand did not move" to "taken" or "the hand moved" to "released";
one adds "no contact".

| Criterion | Verdict |
|---|---|
| 1. Events: F1 at least 70.0% and at least 15.0 points above the baseline | **pass**, 71.3 and +17.8 (above the baseline in 10 worlds of 10) |
| 2. Take right for at least 80.0% of never-met objects | **fail**, 78.8 |
| 3. Conditions exactly right in at least 32 of 40 statements | **fail**, 30 |

| The agent's own experience (reported, not a criterion) | Events, F1 % | Take right, % | Conditions |
|---|---|---|---|
| Simulator's tokens, true behaviours (penalty 5 nats) | 99.6 | 100.0 | |
| Own tokens, true behaviours given | 71.3 | 74.0 | |
| Own tokens, latent state | 70.1 (baseline 64.9) | 61.3 | 27 of 40 |

What the card shows.

- **"Held" can be a learned latent state.** With the true behaviours
  given, a state inferred from own tokens answers card 152's take test
  for 100.0% of never-met objects in every one of the 10 worlds: the
  question card 157 had to drop. Events that no frame shows at once are
  found: in the development world 85 of 102 takes without a move.
- **Events are placed better than by what is seen**, in every world
  (71.3 against 53.5), but far from the bound (99.7). The gap is the
  tokens, not the classes: giving the true behaviours changes nothing
  (70.0).
- **Conditions come out of the learned transition** in the form GOAL
  P12 asks for (an effect and the facts it needs), right in 30 of 40
  statements, with no list of conditions supplied.

Where it is weak.

- **Criterion 2 fails on the class read from appearance**, card 157's
  weakness: where behaviour goes with shape, never-met movable objects
  are believed fixed (three worlds under 50%). Not this component.
- **Criterion 3 fails on the contact token** (checked in one
  development world). Own contacts that are not contacts come mostly
  while the hand moves (31 of 215, against 4 of 87 with the hand
  still), and in 16 more another object was held; "taken" with a move
  becomes less certain and "the hand did not move" is named a
  condition. The raw rates (79 and 89%) differ less than the model's
  (about 0.5 and 0.93), so this is part of the cause, not all of it.
- **Releases without a move are mostly missed** (22 of 151 and 96 of
  140 in the two development worlds). Where an event shows only later,
  its place is decided by the sparsity prior, not by the likelihood,
  and the best penalty differs between own tokens (1 nat) and the
  simulator's (5 nats; at 1 nat the bound drops to 81.3).
- **The agent's experience** holds no take without a move (0 in 10
  worlds), its events are mostly visible (baseline 64.9), and takes are
  right for 61.3%. With the simulator's tokens the same experience
  gives 100.0%, so sparse couplings can carry "taken" over to a
  situation never met; the own tokens do not.
- **Three changes in development** (section 5), one of them a setting
  chosen on two worlds. Criterion 1 passes by 1.3 points.
- **Declared priors and givens:** two values of the state; conditions
  among four tokens of the same step that I chose; own tokens as card
  157; the hand's pose and action.

## 8. Decision

**Keep.** The latent state of an object module is kept as the model's
event mechanism: it restores "held" as something inferred, places events
17.8 points better than what is seen, and yields conditions in P12's
form. The two failed criteria trace to what feeds it (the contact token,
the class read from appearance). Next, with the user's go-ahead: contact
as a learned relation between the hand's and the object's places, in
place of the rule on boxes.

**Correction (card [159](../159-contact-as-a-learned-relation-of-places/card.md)).** The contact rule is not the cause of the failed criteria; a learned relation in its place does no better. The cause is where the model takes an object to be: objects bound by appearance are partial.
