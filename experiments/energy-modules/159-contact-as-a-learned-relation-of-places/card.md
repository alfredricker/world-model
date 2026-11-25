---
id: "159"
title: contact as a learned relation between the hand's place and the object's
rung: 2
serves: [P3, P7, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-21
---

# 159: contact as a learned relation between the hand's place and the object's

Roadmap step 2, rung 2, on the direction under trial. Cards
[157](../157-transitions-read-from-the-binding-layers-own-states/card.md)
and [158](../158-events-as-changes-of-an-objects-latent-state/card.md)
read "contact" by a rule I wrote: the box around an object's points lies
side by side with the hand's box. The rule catches 69% of the
simulator's contacts, and card 158's failed criteria trace to it. It is
also a hand-built mechanism (GOAL C1). The user, 2026-11-21: "yes" to
this card. Runs take under ten minutes each. Biological counterpart:
neurons of premotor and parietal cortex whose visual receptive fields
are anchored to the hand (peripersonal space), and which change with
use (Iriki and others 1996).

**Terms**, beyond cards 157 and 158. The **relative place** of an object
and the hand at a step is a discrete token: the gap between their boxes
along each axis (negative where they overlap), each cut off at -2 and 3,
the two taken together (36 values). It says where the two are with
respect to each other and nothing about what that means. A relative
place is **in reach** if the learned transition of the object's latent
state couples it positively to "taken". Latent state $h_t$, event,
conditions: as card 158.

## 1. Question

Can "contact" be learned as the set of relative places from which the
object's latent state can change, with no rule, and does that repair
what card 158 lost to the rule (events, and the conditions named)? P3
(a relation between two things, learned), P7, P12.

## 2. What changes

One component: how the hand-object relation enters an object module's
transition (`tools/card159/`). The latent state, its learning, the
dynamics classes' rule and the tests are cards 158's and 157's.

```
before (157, 158)  contact_t = 1 if the box around the object's points lies side by side with the hand's box
                   p(h_t = 1 | h_{t-1}, contact_t, grip_t, moved_t; z)
after              r_t = the relative place of the object's box and the hand's box, one of 36 tokens
                   p(h_t = 1 | h_{t-1}, r_t, grip_t, moved_t; z) = sigmoid(b_z + couplings of z + u[r_t]),  u shared, sparse
                   in reach: the relative places with u > 0
  where an object is   the box around those of its points that do not lie on the hand's own box (the agent knows
                       where its hand is; cards 157 and 158 let points of the hand widen an object's box)
  the classes          card 157's rule needs a two-valued contact token: a first fit of the latent state with one
                       class for all objects gives "in reach", which is that token; then the classes; then the
                       latent state with the classes.  No rule on boxes is left.
```

Declared: that the relation is a function of the two boxes' gaps; the
hand's place and size are given.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| An object's place is a box in the image | no | yes | yes | no | declared, as card 152 |
| The relation that matters is a function of the gaps between two boxes | no | yes | partly | no | declared: in 3D the token must hold depth |
| One of the two things is the agent's own body | yes | yes | partly | yes | declared: relations between two objects are a later card |
| Latent state, tokens and classes as cards 157 and 158 | | | | | declared there |

## 3. Dependencies

Cards 158 (latent state, events, conditions), 157 (own tokens, classes),
156, 154.

## 4. Data check

The script's experience, as card 158 (100 episodes a world), gathered
again because an object's points must be read apart from the hand's.
The agent's experience is not run in this card.

## 5. Feasibility gate

On the development worlds (seed 0, by shape and by colour):

- bound: the same learning given the simulator's true boxes and true
  behaviours: in reach catches at least 95% of the simulator's contacts
  with at most 5% false, and events F1 at least 95%;
- baseline: the rule of cards 157 and 158 on the same boxes.

**Passed** (by shape, by colour). True boxes: in reach catches 100.0
and 98.7% with 0.0% false, and is exactly the simulator's rule; events
F1 99.1 and 99.3. The rule on own boxes: 68.6 and 74.8% caught, 14.2
and 11.9% false; events 66.5 and 80.5.

**Development** (seed 0). Changed after the first version: "in reach"
was every relative place with a positive coupling, which gave 16 to 37%
false contacts even on true boxes; it is now the places from which
"taken" is at least half as probable as from the best place. The place
to aim for in the take test is where objects were most often taken.
Tried and dropped: the object's place as its frame carried by its own
displacement (exact in 50.5% of steps against 49.7% for the box of its
points). Development figures on own boxes: in reach catches 80.8 and
88.8% with 44.1 and 59.3% false; events 65.4 and 75.5, under the
rule's. The test was run knowing this.

## 6. Success criteria and prediction

Written before any number of this card existed. The script's
experience; test worlds seeds 1 to 5, by shape and by colour; one run
per world; means of the 10 worlds.

1. **Contact.** "In reach" catches at least 85.0% of the simulator's
   contacts, with at most 10.0% of its own contacts not the simulator's
   (the rule in card 157: 69.4% and 7.5%).
2. **Events.** F1 at least 76.3%, 5.0 points above card 158's 71.3 on
   the same worlds.
3. **Conditions.** At least 32 of 40 statements exactly the world's
   (taken: the relative place and the right grip; released: the other
   grip), where card 158 had 30.

Reported: the take test (the class read from appearance is unchanged,
so it should stay near card 158's 78.8); the rule on the new boxes, to
tell what leaving out the hand's points does alone.

Prediction: 1 passes on what is caught (about 88%) and is near its bar
on false contacts; 2 fails narrowly (about 75: releases without a move
are missed for another reason); 3 passes. Budget: about 45 minutes of
runs, none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). Ten worlds (seeds 1 to 5, by
shape and by colour), the script's experience, one run per world.

| Mean of 10 worlds | Contacts caught, % | Own contacts that are false, % | Events, F1 % | Conditions right, of 40 |
|---|---|---|---|---|
| True boxes, true behaviours (bound) | 98.4 | 0.0 | 99.7 | 40 |
| **Learned relation, own boxes** | **83.4** | **46.6** | **66.8** | **8** |
| The rule of cards 157 and 158 on the same boxes | 74.7 | 13.3 | 68.2 | 16 |
| Card 158 (its own records) | 69.4 (card 157) | 7.5 | 71.3 | 30 |

Take test: 79.1% with the learned relation, 79.2 with the rule, 100.0
with the true behaviours given.

| Criterion | Verdict |
|---|---|
| 1. In reach catches at least 85.0% of contacts with at most 10.0% false | **fail**, 83.4 and 46.6 |
| 2. Events F1 at least 76.3% | **fail**, 66.8 (above the rule's in 2 worlds of 10) |
| 3. Conditions right in at least 32 of 40 statements | **fail**, 8 |

What the card shows.

- **The relation is learnable with no rule, when the places are
  right.** Given true boxes, "in reach" is exactly the simulator's rule
  (boxes overlapping along one axis and touching along the other) in 10
  worlds of 10, and all 40 statements of conditions are right.
- **On the model's own boxes it is worse than the rule it replaces.**
  It catches more contacts by calling near places contacts too (in
  reach equals the simulator's rule in 0 worlds of 10; in the first fit
  with one class, in 5).
- **The cause is where the model takes an object to be** (checked, on
  the development worlds), not the rule and not points of the hand, as
  cards 157 and 158 said. The box around an object's points is the
  object's true box in half of the steps. For an object stored after it
  moved and recognised later it is exact in 85 and 95% of first steps;
  for an object bound by the appearance couplings and not yet moved, in
  43 to 64%, the group covering about 70% of the object's pixels. Card
  156 counted an object bound at an overlap of 0.5, too loose to show
  it. And such a pattern is never corrected: when the object later
  moves, the moved group is taken as known and the partial pattern
  stays.

Where it is weak.

- **This card's records are worse than card 158's**, and I have not
  separated why. Card 158's own model on these records gives 5 of 8
  statements right in two worlds where it had 8 of 8, and F1 68.4
  against 81.2 in one of them. Either leaving the hand's points out of
  an object's box and displacement hurts, or the experience differs
  from run to run (the pattern counts do).
- **A coupling shared by all classes for the relative place** names
  "the hand did not move" as a condition more often than card 158's
  class-specific contact coupling (rule arm here 16 of 40).
- **Run-to-run variation is unmeasured**; one run per world.
- **Declared:** the relative place as gaps between two boxes; the
  hand's place and size.

## 8. Decision

**Stop.** The learned relation is not adopted: on the model's own boxes
it fails all three criteria and does not beat the rule. Cards 157 and
158 stay as they are. What is kept is the finding that the relation
needs no rule once places are exact, and the cause found: objects bound
by appearance are partial, and stay partial. Next, with the user's
go-ahead: a pattern bound by appearance is rewritten when the object
moves, and "bound" is scored by exact extent.

**Checks after the decision (2026-11-21), before any next card.** The
next step named above would not help, and a second idea fails too
(development worlds; numbers in results.json).

- Rewriting a still pattern when its object moves would touch 3 of 335
  and 2 of 286 object sequences. Most partial objects are fixed objects,
  which never move.
- The partial patterns were stored while the appearance couplings were
  young. Patterns stored in episodes 10 to 19 overlap their object by
  0.41 to 0.49 (stored at the end of training: 0.65 to 0.85) and are
  used again and again (123 and 72 sequences from patterns stored in
  episodes 10 to 39). The network as trained at the end binds 0.90 to
  0.97 of an object in fresh scenes where behaviour goes with shape.
- When the appearance couplings switch on they also spoil groups bound
  by motion: the overlap of patterns stored after an object moved falls
  from 0.95 (episodes 0 to 9) to 0.51 (10 to 19). The predicted
  coupling is added at the size of a whole episode of motion evidence
  and stays for the episode.
- Weighting the appearance couplings by how well the network predicts
  motion evidence it has not been fitted to would not help: that skill
  is 0.94 right after the first fit, when the stored patterns overlap
  their objects by 0.44.

