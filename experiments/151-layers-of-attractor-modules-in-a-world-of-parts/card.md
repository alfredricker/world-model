---
id: "151"
title: three feed-forward layers of attractor modules in a world built from parts
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-20
---

# 151: layers of attractor modules in a world of parts

Roadmap step 2, rung 2, on the direction under trial. The simpler world
the user allowed for ("go to something much simpler if the code and
formulation proves to be difficult", 2026-11-20), after
[card 150](../150-three-feed-forward-layers-of-energy-modules/card.md)
ran the layered form on sprite pixels but could not say what should pass
between layers or reach an output token. It also tests the user's first
condition for the whole direction (2026-11-19): "meaningful attractor
states that represent features at low levels, concepts at intermediate
levels" and an output at the top. Runs take seconds. Biological
counterpart: the ventral stream's hierarchy read out by graded population
activity, and the longer intrinsic timescales of higher cortical areas
(Murray and colleagues 2014).

## 1. Question

In a stack of attractor modules where each layer reads only the layer
below, what must a module pass up for the top to be right: its settled
state, its state with its scalar energy, its energy for every stored
pattern, or the scalar alone? Can every layer's attractors be learned in
one shot, locally and without labels, and be the world's own parts? And
does the stack work when run continuously on a changing stream? P7.

## 2. What changes

Nothing in og4. A new world and a new stack: `tools/card151/layers.py`.

```
world    16 patches (16 numbers, +1 or -1).  24 objects of 4 patches; 12 are twins differing from another object
         in one patch.  32 scenes of 4 objects.  Each scene has one of 8 output tokens.
         An image: the scene's 256 numbers + noise; a patch may be covered (zero before the noise).
module   a memory of stored patterns; its energy for a pattern is how badly the pattern fits its input.
         All modules of a layer are one kind, share their patterns and update as one tensor.
layer 1  16 patch modules   input: 16 numbers                        stored: patches
layer 2  4 object modules   input: its 4 patch modules' messages     stored: objects as 4 patch codes
layer 3  1 scene module     input: the 4 object modules' messages    stored: scenes as 4 object codes + token
message  state   which stored pattern fits best
         gated   the same, or nothing when even the best fits worse than the threshold for "new"
         belief  the energy of every stored pattern (as a normalised log-belief); the layer above adds them
         energy  only the scalar: how badly the best pattern fits
learning a module stores its input as a new pattern when nothing it has fits (one threshold); a pattern met
         twice is kept; two patterns that come to fit each other are merged.  Layer by layer, no labels;
         the token is seen by the scene module only.
stream   fresh noise every tick, a new scene every 16 ticks; each layer reads the layer below as it was one tick
         ago; a layer may keep a memory of earlier ticks, v <- (1 - 0.3) v + 0.3 * drive.
```

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Parts sit on a fixed grid and it is given which module reads which | no | no | partly | no | declared: binding (cards 146–147) has to supply this |
| A part is an exact pattern plus independent noise | no | partly | yes | no | declared |
| The threshold for "new" suits the noise during learning | - | - | - | - | declared: set by hand for noise 0.5 |
| Nothing moves and nothing is done; the token is an association | no | no | no | no | declared: actions are cards 148–149 |

## 3. Dependencies

None from earlier cards. A memory that retrieves by the softmax of fit is a
modern Hopfield network (`hopfield-networks-is-all-you-need`).

## 4. Data check

Every world is drawn at random from a seed: world 0 for development,
worlds 1 to 5 for the test. Per condition 4,000 images; streams of 400
by 9 scored segments.

## 5. Feasibility gate

- **Upper bound:** one flat memory of the 32 clean scenes over all 256
  numbers, the best possible reader when the scenes are known.
- **Trivial baseline:** the scalar-only message; the commonest token.

Result of the gate on world 0, before the main run: passed. At noise 4.0
the flat memory names the token in 93.6%, the scalar-only message in 16.4%
(eight tokens).

## 6. Success criteria and prediction

Mean over worlds 1 to 5; the output token must be right.

1. **What to pass** (noise 4.0, the true patterns stored): the belief
   message at least 10.0 points above the state message, within 2.0 of the
   flat memory, and the scalar alone at most 25.0%.
2. **Learned attractors are the world's parts:** in every world, 2,000
   unlabelled scenes at noise 0.5 leave exactly 16, 24 and 32 patterns in
   the three layers, one for each true patch, object and scene; and the
   stack with learned patterns is within 2.0 of the stack with the true
   ones (belief, noise 4.0).
3. **Running on a stream** (noise 4.0): with memory in the scene layer,
   the token over ticks 6 to 15 after a change is right at least 5.0
   points more often than without memory.

Reported without a bar: covered patches; the gated message; a new scene
shown once; memory in the patch layer, in every layer; half the modules
updated per tick; learning from 96 scenes.

On world 0 (after merging duplicate patterns, which the first version
lacked: it left 17, 37 and 139): belief 93.6, state 37.7, scalar 16.4,
flat 93.6; learned 16/24/32 and 93.7; stream 100.0 against 93.0. The bars
were written after seeing these.

Prediction: all three pass. Asynchronous updating will cost accuracy on
the stream, unlike in card 150, because here the input changes.

## 7. Result

Percent of output tokens right, mean of worlds 1 to 5, 4,000 images each;
numbers in [results.json](results.json). The true patterns are stored
unless "learned" is said.

| Noise | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|
| Flat memory of whole scenes (best possible) | 100.0 | 100.0 | 99.0 | 92.0 | 79.7 |
| Message: belief (energy of every pattern) | 100.0 | 100.0 | 99.0 | **92.0** | 79.7 |
| Message: belief, patterns learned | 100.0 | 100.0 | 99.0 | **92.0** | 79.7 |
| Message: state (best pattern only) | 99.9 | 84.8 | 52.2 | **35.3** | 27.7 |
| Message: gated (state, or nothing over the threshold) | 58.3 | 13.7 | 13.9 | 13.6 | 14.0 |
| Message: the scalar energy alone | 15.9 | 16.3 | 16.8 | **16.4** | 16.3 |

| Noise 0.5, patches covered | a quarter | half |
|---|---|---|
| Belief / flat | 100.0 / 100.0 | 99.5 / 99.5 |
| State | 97.3 | 80.2 |
| Gated | 97.9 | 85.2 |

| Stream, noise 4.0: token right, ticks 6–15 after a change (ticks 0–5) | |
|---|---|
| No memory (a single look) | 92.1 (66.1) |
| Memory in the scene layer | **100.0** (50.2) |
| Memory in the patch layer | 100.0 (50.4) |
| Memory in every layer | 82.6 (14.4) |
| Memory in the scene layer, half the modules updated per tick | 78.4 (16.8) |

| Criterion | Verdict |
|---|---|
| 1. Belief ≥ state + 10.0, within 2.0 of flat; scalar alone ≤ 25.0% | pass (92.0 against 35.3; flat 92.0; scalar 16.4) |
| 2. Exactly 16, 24, 32 learned patterns, one per true part, in every world; learned within 2.0 of true | **fail as worded** (worlds 1–4: 16/24/32, one to one; world 5: 15/24/32; learned 92.0 against 92.0) |
| 3. Stream: memory in the scene layer ≥ 5.0 above no memory | pass (100.0 against 92.1) |

Criterion 2 fails on its own wording, not on the learner: in world 5 one
of the 16 patches appears in no scene (checked), so 15 exist to be learned
and 15 were stored. I wrote the bar without making the generator use every
patch. Everywhere a part appears, the stack stored exactly one pattern for
it, at all three layers, from unlabelled scenes. From 96 scenes it has
every patch that appears, 23 or 24 of the objects and 24 to 28 of the 32
scenes (a scene seen once is not kept).

What the card shows.

- **Pass the energy of every stored pattern, not the winner.** With
  beliefs passed up and added, three layers of small modules equal the
  flat memory at every noise level: nothing is lost by the layering. With
  only the winner passed, 92.0 falls to 35.3 at noise 4.0. The single
  scalar alone is at chance. The scalar used as a gate helps a little
  when patches are covered (85.2 against 80.2) and destroys the answer
  once noise exceeds what its threshold was set for (58.3 at noise 1.0).
- **A memory of settled codes is a clean memory.** A new scene of known
  objects, shown once at noise 1.0 with its token, is then recognised in
  100.0% by the stack, which stores four object codes, and in 48.4% by a
  flat memory, which stores the noisy image (0.2% if shown at noise 1.5).
- **Time.** One layer with memory lifts the stream from 92.1 to 100.0
  and takes about four ticks to follow a change; it does not matter which
  layer. Memory in every layer adds the lags (ten ticks). Updating half
  the modules per tick, which cost nothing in card 150, costs 22 points
  here, because the input changes every 16 ticks and stale messages are
  integrated.

Limits: the world is built so that the answer exists (exact repeats, a
fixed grid, a given wiring of modules, noise that is independent and
Gaussian); the equality with the flat memory follows from that
independence and is a check that layering loses nothing, not a discovery;
the threshold for "new" was set by hand for the learning noise; the
output token is an association, not an action with an effect; feedback is
not tested, so a twin object whose telling patch is covered is settled
only at the scene layer.

## 8. Decision

**Keep.** The layered form holds up where it could be tested cleanly:
attractors at three levels learned in one shot and locally, an output
token at the top, and a definite answer to what goes between layers. The
verdict is "fail" by criterion 2's wording; that bar was mine and wrong
for world 5, and I have not changed it. Next, with the user's go-ahead:
feedback (the scene layer telling the object layer which twin it is, and
telling lower layers when the scene has changed, so that memory can be
dropped at an event instead of waited out), then the same stack fed by
card 147's binding in place of a given grid.
