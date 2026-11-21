---
id: "157"
title: an object's transitions read from the binding layer's own states
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-21
---

# 157: an object's transitions read from the binding layer's own states

Roadmap step 2, rung 2, on the direction under trial. In cards
[155](../155-object-modules-fed-by-binding/card.md) and
[156](../156-appearance-couplings-taught-by-the-agents-own-action/card.md)
the binding layer and the appearance couplings read pixels only, but the
dynamics classes did not: an object's transitions were the simulator's
tokens, and the simulator's masks said which true object a stored
pattern lay on. The user, 2026-11-20: "strip away the information that
we feed to the model"; 2026-11-21, on this step: "yes draft step 1".
Runs take under ten minutes each. Biological counterpart: the outcome of
one's own action is judged by comparing what is seen to move with the
copy of the motor command (corollary discharge); neurons in premotor and
parietal cortex respond to objects seen next to the hand (peripersonal
space).

**Terms**, beyond cards 153.1 to 156. A **transition** of object $i$ at
step $t$ is a set of discrete tokens: the condition tokens (the action,
whether the hand is in contact with the object) and the effect tokens
(the object's displacement). **Simulator tokens**: these read from the
simulator's state, as until now. **Own tokens**: the same read from the
binding layer. A **point** is a place of the frame followed from frame
to frame by the binding layer; a stored pattern's points in an episode
are the group it was stored from or recognised on.

## 1. Question

Do the dynamics classes, and the action proposed for a never-met object,
survive when each object's transitions are read from the binding layer's
own states instead of the simulator's, and nothing says which true
object a pattern is? P7.

## 2. What changes

One component: where an object module's transitions come from
(`tools/card157/`). The agent, the binding layer, the appearance
couplings, the memory of patterns and the rule for the classes are card
156's and 153.1's, unchanged.

```
before (155, 156)   object    a stored pattern, given the true object under it by the simulator's mask
                    tokens    contact, held before | displacement, held after: from the simulator's state
                    left out  patterns on the ground and on the hand (the mask said they are no object)
after               object    a stored pattern and its points in this episode; no mask
                    contact   the box around the pattern's points lies side by side with the hand's box
                    effect    the displacement most of its points were seen to make (the motion modules' reading),
                              and "came along": that displacement is the hand's own and is not zero
                    held      dropped: whether the hand holds an object cannot be seen until the hand moves
                    left out  a pattern lying on the hand's own box (the agent knows where its hand is)
```

Patterns on the ground now have transitions too, like any object. The
hand's place and size, its displacement and the action stay given: they
are the agent's own pose and motor command.

**The test changes with it, and this is declared.** Card 152's test
asked for the action that leaves an object held without moving it.
"Held" is a state of the simulator that no frame shows. The test here
asks for the action that brings the object along: the hand is put at
the object's side (by the simulator, as before), the wanted effect is
the object displaced by three pixels toward the middle of the frame, the
action settles, and it is right if the object has moved with the hand in
the simulator; for a fixed object the right answer is "cannot". The
simulator-token arm is scored on the same test. Card 156's failure (the
agent never takes without a move) is therefore not tested here; this
card neither cures it nor measures it.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The agent's own pose, extent and action are given | yes | yes | partly | yes | declared (CHARTER C1) |
| Contact is two boxes side by side in the image | no | partly | partly | no | declared: in 3D it needs depth or touch |
| An object's displacement is one of a few whole-pixel shifts | no | yes | yes | no | declared, as card 152 |
| An object responds within the step in which it is acted on | partly | yes | yes | partly | declared: delayed effects need a state the module keeps |
| Transitions are a few discrete tokens | partly | yes | yes | no | declared, as card 153 |

## 3. Dependencies

Cards 147 and 150 (points, motion, binding couplings), 154 (stored
patterns and feedback), 156 (appearance couplings and the agent), 152
and 153.1 (the transition energy and the dynamics classes).

## 4. Data check

Card 152's sprite worlds. One run of the agent per world (100 episodes)
is saved, and both token sources are fitted on that same experience, so
the two arms differ in the tokens only.

## 5. Feasibility gate

On the development worlds (seed 0, by shape and by colour), the new
test with the simulator's tokens:

- upper bound: the true behaviour of each object given, at least 90%;
- baseline: one class for all objects, under 60%;
- the simulator-token arm (card 156 as it is) at least 80%, so that
  criterion 1's bar is one this experience can reach.

**Passed** (by shape, by colour; never-met objects): true behaviour
given 100.0 and 100.0; one class for all 40.7 and 43.1; the
simulator-token arm 88.3 and 100.0.

Development, one version, nothing tuned: own tokens 84.1 and 100.0;
purity 86.2 and 94.5. The own contact token catches 77.8 and 66.9% of
the simulator's contacts; widening the box rule by one or two pixels
catches 2 to 6 points more and triples the false contacts, so it was
left exact. Most missed contacts are cases where the box around the
pattern's points reaches outside the object (340 of 509, and 627 of
809).

## 6. Success criteria and prediction

Written before any number with own tokens existed. Method: the agent's
experience, own tokens. Test worlds: seeds 1 to 5, by shape and by
colour; 100 training episodes; objects never met; one run per world.

1. The action that brings the object along is right for at least 80.0%
   of never-met objects on the mean of the 10 worlds.
2. That figure is within 5.0 points of the simulator-token arm's on the
   same experience.
3. The dynamics classes agree with the true behaviours: purity at least
   90.0% on the mean of the 10 worlds (a pattern on the ground counted
   as fixed).

Reported, not criteria: how often the own tokens equal the simulator's
(contact, displacement); the number of classes.

Prediction: the simulator-token arm about 88%; own tokens about 80%, so
1 is near its bar, 2 fails (contact read from boxes will miss cases),
3 passes. Budget: about 60 minutes of runs, none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). Ten worlds (seeds 1 to 5, by
shape and by colour), 100 episodes of the agent's own experience per
world, both token sources fitted on that same experience. "Right": the
proposed action brings a never-met object along in the simulator
("cannot" for a fixed object).

| Never-met objects, action right, % | Mean of 10 | By shape | By colour | Takeable | Fixed | Sticky |
|---|---|---|---|---|---|---|
| True behaviour given, own tokens (bound) | 100.0 | | | | | |
| Simulator tokens (cards 155, 156) | 85.0 | 76.8 | 93.2 | 98.8 | 72.1 | 81.1 |
| **Own tokens** | **86.4** | 72.8 | 99.9 | 80.2 | 98.6 | 75.1 |
| One class for all, own tokens | 36.7 | | | | | |

| | Simulator tokens | Own tokens |
|---|---|---|
| Purity of the dynamics classes, mean (lowest world), % | 97.8 (95.4) | 88.9 (81.2) |
| Classes per world | 3 or 4 | 4 to 6 |
| Known objects, action right, % | 87.2 | 91.4 |

The own tokens against the simulator's, where a pattern lies on a true
object: 69.4% of the simulator's contacts are caught, 7.5% of the own
contacts are not the simulator's; the displacement is the same in 98.2%
of the steps in which the object moved and in all in which it did not.

| Criterion | Verdict |
|---|---|
| 1. Own tokens: action right for at least 80.0% of never-met objects on the mean | **pass**, 86.4 (lowest world 61.1) |
| 2. Within 5.0 points of the simulator-token arm on the same experience | **pass**, 1.4 points above |
| 3. Purity of the classes at least 90.0% on the mean | **fail**, 88.9 |

What the card shows.

- **The simulator's tokens and the masks are not needed.** With each
  object's transitions read from the binding layer, and nothing saying
  which true object a pattern is, the mean is no lower (86.4 against
  85.0). With the true classes given, couplings fitted to the own tokens
  answer right for 100.0%: the loss is in inferring the class, not in
  the tokens' content.
- **Criterion 2 passes on the mean only.** Per world the difference runs
  from -29.8 to +19.7 points (three worlds lower, four higher, three
  equal). Own tokens are better on fixed objects (98.6 against 72.1) and
  worse on takeable ones (80.2 against 98.8). One run of the agent per
  world; its run-to-run variation is not measured.
- **The ground and the hand become objects with dynamics.** Patterns on
  the ground join the class of fixed objects. Up to 6 patterns per world
  that lie on the hand escaped the rule that leaves the hand out and
  formed a class of their own: what always comes along, in contact or
  not.

Where it is weak.

- **Class inference where behaviour goes with shape** (72.8%; three
  worlds at 61 to 76%). Movable objects fall into the fixed class (5 to
  12 of about 60 takeable objects per world by weight), and the classes
  split (4 to 6 for 3 behaviours).
- **The contact token.** It misses 3 in 10 of the simulator's contacts,
  most often because the box around a pattern's points reaches outside
  the object. A miss turns "taken" into "came along untouched".
- **Two explanations I checked and did not confirm.** (a) That the
  patterns on the ground cause the errors: refitted without them, one
  world rises from 75.1 to 86.0 and another only from 61.1 to 64.8.
  (b) That the simulator arm is weak on fixed objects because they are
  seldom in contact by the simulator's rule: they are in contact as
  often as by the own rule. The simulator arm's weakness on fixed
  objects is left unexplained.
- **The test is not card 156's**: it asks for an effect that can be
  seen. Card 156's failure is neither cured nor measured.
- **Still given:** the hand's place, size and displacement, the action,
  the hand put at the object's side at test, the form of the tokens
  (contact as boxes side by side, whole-pixel displacements), the two
  appearance features, the motion front end.

## 8. Decision

**Keep.** Own tokens replace the simulator's tokens and the masks in
learning: on the same experience they do no worse on the mean, and two
of three criteria pass. Carried forward as known limits: purity 88.9%,
the contact token, and class inference where behaviour goes with shape.
Next, with the user's go-ahead: the goal task against an ordinary
learned baseline.
