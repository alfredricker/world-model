---
id: "148"
title: the action as an output token, completed from records of what actions did
rung: 2
serves: [P7]
status: done
verdict: pass
arch_version: og4
date: 2026-11-19
---

# 148: the action as an output token

Roadmap step 2, rung 2, on the direction under trial (energy modules).
The user, 2026-11-19: "I want to know if we could implement (or if we
already have) some implementation of output tokens where a token can
represent an action, a word", then "let's do the action token and effect
records. I definitely lean towards the coupling version since it is more
true to the energy formulation and the CRAM paper." og4 has no such token:
it replays a recorded gripper path. Every run is under ten minutes.
Biological counterpart: the ideomotor principle, that an action is selected
by the effect it is expected to have, with action and effect bound in one
record (Hommel and colleagues' event coding, 2001).

## 1. Question

Can the action be one more token in the same kind of energy as the objects,
so that clamping the state now and the wanted effect and settling the rest
gives the action that produces that effect, and how many steps of the
agent's own experience does that take? P7.

## 2. What changes

Nothing in og4. A one-component test in the sprite world's simulator; the
component is the memory that ties action to effect, in two forms.

```
one step = tokens   hand: e_y, e_x (how far it moved)       action: a_y, a_x (9 values each), a_g (grip)
                    each object: c (in contact before), h (held before), o_y, o_x (how far it moved), h2 (held after)
query               clamp the hand's and every object's tokens (state now + wanted effect); the action is free
arm P  couplings    E = - sum over pairs of tokens K[value, value] - biases; one set of couplings for every object,
       (as CRAM)    one action shared by all objects of the scene.  Learned by pseudolikelihood (each token
                    predicted from the others), L1 penalty, no backpropagation through settling.
                    Read: settle the three action tokens (mean field, falling temperature).
arm R  records      every (step, object) stored whole, in one shot.  Read: retrieve by softmax(beta x number of
       (Hopfield)   clamped tokens that match); the retrieved records vote; votes multiply across objects.
```

The chosen action is run in the simulator from the same state. It succeeds
if the hand, every object and what is held end exactly as wanted.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Contact and "held" are given (touch, proprioception) | no | no | no | yes (OGBench's contact sensor, card 120) | declared: simulator state, as the one-ingredient rule allows; co-motion (card 147) could supply "held" |
| An effect is a few tokens with few values | partly | yes | yes | no (continuous) | declared: continuous effects need a learned codebook |
| Objects affect each other only through the hand | no | no | no | no (stacking, drawers) | declared: no couplings between objects |
| One step from action to effect | no | no | no | no | declared: sequences are the next card |
| The world is deterministic | no | no | yes | nearly | general: both forms hold distributions |
| Objects come from the simulator | - | - | - | - | declared: card 147's groups are not connected yet |

## 3. Dependencies

`tools/card148/env.py`: card 147's world with the action made explicit
(grip first, then the move), read as state, without the button and lock.
Pseudolikelihood for a pairwise energy: CRAM (LITERATURE, `2611_17309`).
Retrieval as one attention step: `hopfield-networks-is-all-you-need`.

## 4. Data check

Experience is a scripted behaviour (go to an object, take it, carry it, let
go, or wander) with 20% random actions, 60 steps an episode. In 6,000
held-out validation steps: move 3,336, carry 1,644, take 408, let go 368,
in contact but not taken 244. Records are the first n steps of a stream of
50 episodes; five streams.

## 5. Feasibility gate

- **Upper bound:** the recorded action run again, and a hand-written rule
  (move = the hand's wanted move; grip = some object wanted held).
- **Trivial baseline:** the commonest action in the records; a random one.

Result of the gate, before the main run: passed. Recorded action 100.0%,
the rule 100.0%, the commonest action 5.2%, a random action 1.1%.

## 6. Success criteria and prediction

Each arm is judged alone on 6,000 test steps (100 episodes not used for
anything else), mean of five streams of records.

1. **With 3,000 steps of experience:** the action achieves the wanted
   effect in at least 95.0% of steps, and in at least 90.0% of each kind
   (take, carry, let go, in contact but not taken).
2. **With 100 steps:** at least 80.0%. This is learning from little, the
   reason to want memory inside the model.
3. **Settling finds the minimum** (arm P): the settled action is the
   lowest-energy one of the 162 in at least 99.0% of queries. Trying every
   action will not be possible for sequences; settling has to do.

The card passes if one arm meets its criteria. The arms differ if criterion
2's measure differs by more than 5 points.

Settings, chosen on validation (one stream): the L1 penalty 0.001 of
{0, 0.001, 0.01}; beta 4 of {1, 2, 4, 8, 16}. On validation arm P scored
20.5, 55.9, 100.0, 98.1, 100.0, 100.0% with 10, 30, 100, 300, 1,000 and
3,000 steps, and arm R 13.1, 40.0, 68.8, 88.9, 99.2, 100.0%. The bars were
written after seeing these.

Prediction: arm P passes all three; arm R passes 1 and fails 2. Budget: one
run of under ten minutes.

## 7. Result

Percent of 6,000 test steps in which the completed action gives exactly
the wanted effect, mean of five streams; numbers in
[results.json](results.json). Baselines: the commonest action 5.2%, a
random one 1.1%.

| Steps of experience | 10 | 30 | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|---|---|
| Arm P, couplings | 18.8 | 63.1 | **95.6** | 98.9 | 100.0 | **100.0** |
| Arm R, records | 16.4 | 40.9 | **77.5** | 93.6 | 99.3 | **99.9** |
| Arm P: settled action is the lowest of the 162 | 96.8 | 99.7 | 100.0 | 100.0 | 100.0 | 100.0 |

| Criterion | Arm P, couplings | Arm R, records |
|---|---|---|
| 1. 3,000 steps: ≥ 95.0%, each kind ≥ 90.0% | pass (100.0; every kind 100.0) | pass (99.9; lowest kind 98.7, in contact but not taken) |
| 2. 100 steps: ≥ 80.0% | pass (95.6; streams 86.2 to 99.7) | fail (77.5; streams 73.7 to 79.6) |
| 3. Settling finds the minimum, ≥ 99.0% | pass (100.0 from 100 steps on; 99.4 over all six sizes) | not applicable |

The card passes on arm P, and the arms differ (18 points at 100 steps).
Criterion 3 did not say at how many steps: with 10 records, where the
energy is nearly flat, settling finds the minimum in 96.8% of queries, and
the success of the settled and the exhaustive action is the same (18.8,
18.9).

What the couplings became (one stream, 3,000 steps): between the hand's
effect and the action's move on the same axis, 15 of 81 entries are above
0.05, the diagonal at 5.5; between different axes, 0 of 81; the grip's
strongest coupling is to "held after" (2.7). So the energy holds "move by
what you want the hand to move by; close if something should be held" as
separate pairwise facts, which is why 100 steps are enough: it need not
have seen a combination to produce it. A stored record is one whole
combination, and arm R needs about ten times the experience; its weakest
kinds at 100 steps are the rare ones (let go 53.4%, in contact but not
taken 58.0%).

Limits. The tokens were designed by me and their values come from the
simulator (contact, held, displacements). The right action here is a sum of
pairwise facts, which a pairwise energy represents exactly; an effect that
depends on three tokens jointly would need a hidden token, which CRAM's
authors designed by hand. Nothing is a sequence: every query is one step,
and a wanted effect that needs an approach first was not asked.

## 8. Decision

**Keep.** An action can be an output token of the same kind of energy that
binds the objects: clamped on the state and the wanted effect, the energy
settles to the action that produces it, from 100 steps of the agent's own
experience, and settling finds the same action as trying all of them. The
couplings form (the user's preference, and CRAM's) is carried forward; the
records form is kept as the comparison that shows why. Nothing enters og4.
Next, with the user's go-ahead: sequences, where the wanted effect needs
several steps, with the same couplings between the tokens of consecutive
steps.
