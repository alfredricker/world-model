---
id: "155"
title: object modules fed by binding instead of masks
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-20
---

# 155: object modules fed by binding instead of masks

Roadmap step 2, rung 2, on the direction under trial. The user,
2026-11-20: "strip away the information that we feed to the model", and,
between feeding the object modules from card 154's binding and learning
couplings from appearance, "let's do the former". This is a measurement:
no new component, the simulator's masks removed from learning. Runs take
two to four minutes per world. Biological counterpart: an infant's
knowledge of objects begins with the things it has itself moved
(Needham's "sticky mittens").

**Terms**, as cards 153.1 and 154: object $i$; its appearance features
$a_i$ (attractor indices of a shape memory and a colour memory); its
observed transitions $D_i$; its dynamics class $z_i$ (a latent variable;
each class is an energy over a transition's tokens); the appearance
prior $p(z \mid a)$; a stored pattern (an object once bound by motion:
its points in a 16 by 16 frame, their appearance, their binding
couplings).

## 1. Question

With the masks removed from learning, do the object modules built on
card 154's binding still give an object its dynamics class, and what is
lost against card 153.1 with masks? P7.

## 2. What changes

One given is removed (`tools/card155/pipeline.py`); every mechanism is a
kept card's.

```
before (153.1)  object = the simulator's mask of it, in every episode and at test
                a_i from the mask's silhouette and mean colour;  D_i = every step of every episode it is in
after           object = a stored pattern of card 154's memory (bound by motion, or recognised at first sight)
                a_i from the pattern's silhouette and mean colour
                D_i = the steps from the one in which the group first moved (every step of an episode in which the
                      pattern was recognised at first sight)
                what no pattern explains is not an object: predicted to stay put, "take it" answered "cannot"
```

Still given, and declared: the transition tokens (in contact, held,
displacement, held after) come from the simulator, and are matched to a
stored pattern by the true object the pattern lies on; which module is
the hand; the shape and colour memories' two thresholds (card 152).

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A thing that never moves is scenery (predicting "stays put" is right) | partly | partly | no | partly | declared: true of this world's fixed objects only |
| An object is met again, looking the same pixel for pixel | no | no | partly | no | declared, as card 154 |
| Transition tokens are given per true object | no | no | no | no | declared; the next given to remove |
| The dynamics class depends on one appearance feature | partly | partly | partly | partly | declared, as card 152 |

## 3. Dependencies

Card 154 (binding with feedback, the memory of stored patterns), card
152 (shape and colour memories, the tests), card 153.1 (dynamics classes
and the appearance prior).

## 4. Data check

Four development worlds, 100 training episodes: 67 to 82 objects get
transitions, all of them takeable or sticky (43 to 59 and 20 to 26),
none fixed, of 166 to 204 true objects; 1,961 to 2,359 of the 2,740 to
2,874 transitions in contact or held are used. At test 77 to 100% of the
known takeable and sticky objects are bound at first sight, 0 to 1% of
the fixed ones, and no never-met object.

## 5. Feasibility gate

- **Upper bound:** card 153.1 with masks, same worlds, episodes and
  tests (seeds 1 to 5, 100 episodes): the take action right for 98.8% of
  known objects and 99.8% of never-met ones.
- **Trivial baseline:** one class for all: 39% on development worlds.

Passed (both were measured in card 153.1).

## 6. Success criteria and prediction

Test worlds: seeds 1 to 5, by shape and by colour, 100 training
episodes, card 153.1's episodes and tests. Written after the development
runs and before any test result was read.

1. **What the binding finds behaves as with masks.** No mask anywhere:
   for known objects bound at first sight, the take action right for at
   least 95.0%, and effects within 3.0 points of the bound (true
   behaviour given) on the same transitions, in each of 10 worlds.
2. **The classes are learned without masks.** With the true mask given
   at test only, never-met takeable objects and never-met sticky objects
   each get the right take action in at least 90.0% of cases, in each of
   10 worlds.
3. **End to end on known objects.** No mask anywhere, all known objects:
   the take action right within 10.0 points of card 153.1 with masks
   (98.8) on the mean of the 10 worlds.

Reported and not criteria, because they cannot pass by construction:
never-met objects with no mask (none is bound at first sight), and fixed
objects when a mask is given (no fixed object ever moves, so no class
for them is learned).

Development (seed 0 by colour and by shape, 11 by shape, 12 by colour):
two classes of purity 100% in each world (one world adds a class of one
object); bound known objects, take 99.4 to 100.0% and effects within 0.3
of the bound; never-met with the mask at test, takeable 100, 81.7, 100,
100 and sticky 100 in all; all known objects, take 90.0, 92.2, 96.1,
100.0.

Prediction: 1 and 3 pass; 2 fails in a world or two by shape, as one
development world did. Budget: about 15 minutes of runs, none over 10.

## 7. Result

Numbers in [results.json](results.json). Ten worlds (seeds 1 to 5, by
shape and by colour), 100 training episodes; "with masks" is card 153.1
on the same worlds, episodes and tests. "Take" is the share of objects
for which the proposed action is right in the simulator, "effects" the
share of transitions in contact or held that are predicted exactly.

| Mean of 10 worlds, % | No masks (this card) | With masks (153.1) | One class for all |
|---|---|---|---|
| Known objects bound at first sight: take | **100.0** (every world) | | |
| Known objects bound at first sight: effects, against the bound on the same transitions | at most 0.4 under | | |
| All known objects: take | **94.5** (lowest world 90.4) | 98.8 | 40.0 |
| All known objects: effects | 89.0 | 92.9 | |
| Never-met objects, no mask at test: take | 39.1 (none is bound) | 99.8 | |
| Never-met, mask at test only: takeable / sticky / fixed, take | 100.0 / 92.0 / 0.0 | 100 / 100 / 99.5 | |

Learning without masks: 61 to 95 objects with transitions per world, one
of them fixed in all ten worlds together; 73.2% of the transitions in
contact or held are used; two classes in every world, purity 98.4% in
one world and 100% in nine. At test 91.7% of known takeable and 90.2% of
known sticky objects are bound at first sight (lowest worlds 80.7 and
64.7), and no fixed one.

| Criterion | Verdict |
|---|---|
| 1. Bound known objects: take at least 95.0%, effects within 3.0 of the bound, in each of 10 worlds | **pass**, 10 of 10 |
| 2. Mask at test only: never-met takeable and sticky objects each at least 90.0%, in each of 10 worlds | **fail**, 9 of 10 (world 1 by shape: sticky 20.0) |
| 3. All known objects: take within 10.0 points of card 153.1 on the mean | **pass** (94.5 against 98.8) |

**The failure of criterion 2** (checked by rerunning world 1 by shape).
Only 13 sticky objects got transitions there, and the masks stored from
binding are less clean than true ones (mean overlap 0.81 for the
patterns on the squares' shape attractor, the sticky shape). That
attractor holds 7 sticky and 3 takeable objects, and 6 sticky objects
fall on other attractors, so the appearance prior gives a square only
0.66 for the sticky class. Appearance read from an imperfect mask blurs
the feature the class depends on; with few objects of a class the prior
does not recover.

What is lost without masks, in order of size.

- **Never-met objects are not objects until they move** (take 39.1
  against 99.8; the 39.1 is the fixed objects, for which "cannot" is
  right). Nothing binds an object the agent has not yet moved, so its
  appearance cannot be read before acting on it.
- **Known objects never moved in training are invisible too**: 8 to 10%
  of the known takeable and sticky objects on the mean, 35% of the
  sticky ones in the worst world. Every miss on known objects is one of
  these (94.5 against 98.8 with masks).
- **No class is learned for fixed objects.** Unbound, they are treated as
  scenery, which is right in this world. Given a mask they are called
  takeable or sticky (take 0.0, effects 46 to 53).
- **A quarter of the informative transitions are not used** (those before
  an object's first move in the episode where it is first stored).

What is not lost: an object the binding has found is handled exactly as
with masks, exceptions included (take 100.0 for bound objects in all ten
worlds), and both dynamics classes of things that move are found in
every world.

Limits: the transition tokens are still the simulator's and are matched
to patterns by the true object; the hand is given; 100 episodes; in this
world a thing that never moves really is scenery.

## 8. Decision

**Keep**, as the measured cost of removing the masks. The verdict is a
fail because criterion 2 failed in one world of ten. The loss is almost
all one thing: an object exists for the model only after it has moved.
The user, 2026-11-20: couplings as a function of appearance need "a
training incentive", and "an interactive environment where the agent
learns by what objects do may also help". Next, with the user's
go-ahead: an agent that acts on what it has not yet bound, with
appearance couplings trained to predict the co-motion its own action
then shows; after that, the transition tokens read from the modules'
states instead of the simulator.
