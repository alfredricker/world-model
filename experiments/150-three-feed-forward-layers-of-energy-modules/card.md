---
id: "150"
title: three feed-forward layers of energy modules on the sprite world's pixels
rung: 2
serves: [P7]
status: done
verdict: pass
arch_version: og4
date: 2026-11-20
---

# 150: three feed-forward layers of energy modules

Roadmap step 2, rung 2, on the direction under trial. The user, 2026-11-20:
"maybe we should try out experiments with modularizing energy into a
layered structure, each module performing parallel computation ... the
deeper layers update continuously in time based on the abstractions of
intermediate layers", then "let's start with feed forward and then we can
experiment with feedback. Starting with sprite world may be too complex,
but give it a shot, and go to something much simpler if the code and
formulation proves to be difficult." Runs take one to two minutes.
Biological counterpart: the fast feed-forward sweep through the visual
areas that comes before any recurrent processing (Lamme and Roelfsema 2000).

## 1. Question

Do three layers of energy modules, each layer reading only the layers
below, give the right state at the top when run on the pixel stream, and
does that survive few updates per frame and asynchronous updating? Which
message from a lower module serves the upper one: its settled state, its
state with its scalar energy, or its whole belief? P7.

## 2. What changes

Nothing in og4. Pieces of card 147 are arranged as a stack and a third
layer is added.

```
layer 0  motion modules, one per pixel (4,096).  State: a belief over 121 displacements.  Energy: the matching
         cost of each displacement.  Each settles alone, in one step.
layer 1  binding modules, one per followed point.  State: a belief over at most 8 objects.  Energy: pairwise
         couplings learned from layer 0's settled states (card 147's couplings).  They settle together,
         `ticks` updates per frame, carried over from the last frame; optionally only a random share per tick.
layer 2  object modules, one per object.  State: the object's displacement.  It reads which points are its own
         (layer 1) and those points' layer-0 output, in one of three ways:
           state    each point's settled displacement; the object takes the commonest
           gated    the same, only from points whose layer-0 energy is low (the scalar energy as confidence)
           belief   the sum of its points' energies over all displacements; the object takes the lowest
```

Nothing flows downward. **What was left out, and why:** the action layer of
card 148 was not put on top. It reads "touched" and "held", and no layer
here produces those; I would have had to write them by hand from the
groups' outlines, and an object that has not moved yet has no module to
carry a wanted effect. That is the part of the formulation that proved
difficult in this world.

### Portability check (CHARTER rule 9)

As card 147 (fixed camera, flat rigid motion, texture, only moved things
are bound). One addition.

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A layer's modules are of one kind and can be updated as one tensor | yes | yes | yes | yes | general |

## 3. Dependencies

Card 147: the front end, the point follower and the couplings (arm C),
unchanged except that the number of updates per frame and the share of
points updated per tick are exposed. Its test and small splits.

## 4. Data check

Card 147's 200 test episodes: 60 steps each; the hand or an object that has
moved is in view and moving in about 80 step-thing pairs per episode.

## 5. Feasibility gate

- **Upper bound:** layer 1 replaced by the true masks.
- **Trivial baseline:** one object module for the whole frame.

Result of the gate, before the main run: passed. With the true masks 99.8%;
with one module for the whole frame 0.0%.

## 6. Success criteria and prediction

The measure: over every step in which the hand or an already-moved object
is moving, the share for which its object module's displacement is exactly
right (belief reading). 200 test episodes.

1. **The stack is right:** at least 97.0% with 40 updates per frame.
2. **Few updates:** with 5 updates per frame, within 2.0 points of 40.
3. **Asynchronous:** a random half of the points updated per tick, 80
   ticks, within 2.0 points of all points for 40.

Reported without a bar: the three readings side by side; 1, 2 and 10
updates; the small split.

Nothing was tuned. On 20 validation episodes the stack scored 99.3% by all
three readings (true masks 99.4 to 99.5); the bars were written after
seeing that.

Prediction: all three pass, and the three readings will not differ by a
point, because this world's noise is too low for a single pixel's motion
to be in doubt. That would leave the user's question about what to pass
unanswered here.

## 7. Result

Percent of moving steps for which the object module's displacement is
right, 200 test episodes (17,124 object-steps and the hand's); numbers in
[results.json](results.json).

| Layer 1 | Updates per frame | Points updated per tick | State | Gated | Belief |
|---|---|---|---|---|---|
| true masks (gate) | - | - | 99.8 | 99.8 | 99.8 |
| one module (gate) | - | - | 0.0 | 0.0 | 0.0 |
| binding | 40 | all | 99.5 | 99.5 | **99.5** |
| binding | 10 | all | 99.5 | 99.5 | 99.5 |
| binding | 5 | all | 99.4 | 99.5 | **99.5** |
| binding | 2 | all | 99.3 | 99.3 | 99.3 |
| binding | 1 | all | 98.8 | 98.8 | 98.8 |
| binding | 80 | half, at random | 99.5 | 99.5 | **99.5** |
| binding | 10 | half, at random | 99.5 | 99.5 | 99.5 |
| binding, small split | 40 | all | 99.9 | 99.8 | 99.9 |

| Criterion | Verdict |
|---|---|
| 1. At least 97.0% with 40 updates | pass (99.5) |
| 2. 5 updates within 2.0 of 40 | pass (99.5) |
| 3. Asynchronous within 2.0 of synchronous | pass (99.5) |

The stack works as a feed-forward chain on pixels, and how it is clocked
hardly matters: one update per frame costs 0.7 points, and a random half
of the points per tick costs nothing, because the binding layer's state is
carried from frame to frame and changes rarely. Objects alone (not the
hand) are right in 99.0%, against 99.3% with true masks.

The three readings are the same to within 0.1 points everywhere, as
predicted: with this little sensor noise every pixel's motion is already
certain, so nothing is gained from the lower module's energy, as a scalar
or as a whole belief. This world does not answer what should be passed
between layers.

## 8. Decision

**Keep**, as a demonstration that the layered form runs on our pixel
world. Two things it could not show go to a simpler world, as the user
allowed: what a lower module should pass up when its input is in doubt,
and a stack that reaches an output token with attractors learned at every
level ([card 151](../151-layers-of-attractor-modules-in-a-world-of-parts/card.md)).
