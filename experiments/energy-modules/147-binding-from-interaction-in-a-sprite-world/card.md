---
id: "147"
title: binding from the agent's own interaction, as couplings and as motion prediction, in a sprite world
rung: 2
serves: [P7]
status: done
verdict: pass
arch_version: og4
date: 2026-11-19
---

# 147: binding from the agent's own interaction

Roadmap step 2 (objects), rung 2. The user, 2026-11-19, on cards 146 and
146.1: "I think the learning signal should be by agent interaction with
objects: their effect and event segmentations and the tracking of their
motion", and then "we should test out both of those directions on sprite,
coupling and motion prediction". Every run takes under five minutes and
trains nothing by gradient. Biological counterpart: grouping by common fate,
which infants use before static cues (Kellman and Spelke 1983); for arm M,
object files (Kahneman, Treisman and Gibbs 1992).

## 1. Question

Can the points of a scene be bound into objects using only how they move
while the agent acts, with no labels and no use of appearance, and does it
matter whether that evidence is kept between pairs of points (couplings) or
between each point and each object (motion prediction)? P7.

## 2. What changes

Nothing in og4. This is the second step of the test ladder agreed on
2026-11-19: a one-component test in our own world. Card 146.1's belief over
at most K objects is kept; what changes is where the grouping evidence comes
from.

```
shared   motion read from two frames by block matching -> tokens (points followed from frame to frame; a
         static point that gets covered is remembered and comes back) -> one observation per seen token
         per step: the displacement it made, possibly zero
arm C    J_ij += 1.2 if tokens i and j moved alike; J_ij -= 4.6 if one moved and the other did otherwise
         (log-odds: an object carried by the hand moves like it, 0.3; a misread motion, 0.01); nothing if
         both stayed.  Read: settle  E = -1/2 sum (J_ij + 0.05) <p_i, p_j> + T sum <p_i, log p_i>,
         40 steps from the last frame's state.  Memory: tokens x tokens.
arm M    at most K object files.  d_k = the displacement most of file k's tokens made.
         S_ik += 1 when token i did not move by d_k.  Token i belongs to argmin_k S_ik (a tie: where it
         is).  Three or more tokens that every file now predicts worse than their own did, and that
         moved alike, open a new file.  Memory: tokens x K.
baseline per step, no memory: tokens that moved alike in this step are one object, the rest another
all      K = 8, never the number of objects.  A token not seen before joins the largest group.
         Neither arm reads masks, the action, or appearance.
```

In both arms one step of moving apart outweighs several steps of moving
together, so a carried object stays apart from the hand that also moved
without it. Arm M without the accumulated count is what I first built: it
merged a held object with the hand in 37% of frames and found a quarter of
released objects.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The camera is fixed, so "stayed" means zero displacement | no | no | yes | fixed cameras only | declared: motion would be taken relative to the scene's own motion |
| An object moves as one displacement (rigid, flat, no turning) | no | yes | yes | no | declared: needs a richer motion per object (turning, depth) |
| Motion can be read at every pixel (texture, low noise) | partly | yes | yes | no (plain cubes) | declared: a pretrained point tracker would supply it, as CHARTER allows |
| Objects move only when the agent moves them | no | no | mostly | yes (OGBench) | general: a thing that moves by itself is found the same way |
| Parts of one object never move apart | no (doors) | yes | yes | no (arm links, drawers) | declared: jointed things split into rigid parts |
| An object that never moves need not be found | - | - | - | - | declared: it stays part of the background until moved; only moved things are scored |

## 3. Dependencies

Card 146.1 (the belief form and its settling; `tools/card146/energy.py`'s
update, rewritten for tokens). Block matching with shiftable windows
(Bobick and Intille 1999), checked in the gate. The world, the token
follower and both arms are new here: `tools/card147/`.

## 4. Data check

`tools/card147/world.py`: 64 by 64 pixels, 61 frames an episode. A hand
moves up to 4 pixels a step, takes an object by touching its side, carries
it and lets go; 2 to 4 objects of width 4 to 16 in five shapes, lying over
each other in a fixed order; in 30% of episodes two objects look identical;
in half, a button and a lock rule (not used by this card). Everything is
textured and each frame has fresh sensor noise. Validation 100 episodes
(settings), test 200, small 200 (every object 3 or 4 pixels wide). In test
an object is held in 45% of steps and 77% of objects get moved (small: 41%,
76%).

## 5. Feasibility gate

- **Front end:** on 50 validation episodes the motion read for moving
  pixels is right for 99.6% (read for 99.8% of them); 0.11 still pixels a
  frame are read as moving.
- **Upper bound:** each arm given the simulator's true motion.
- **Trivial baseline:** the per-step grouping.

Result of the gate, before the main run: passed. With true motion the arms
find 98.4 to 99.7%; the baseline finds 39.9% (test) and 41.8% (small). Section
7 has the table.

## 6. Success criteria and prediction

The measure: for the hand and every object, in every frame after it first
moved and while in view, it is *found* if the group covering most of it has
intersection over union of at least 0.5 with its true mask. Each arm is
judged alone, on motion read from pixels, K = 8.

1. **Found at least 93.0%** on the 200 test episodes, and within 3.0 of the
   same arm given true motion. This is binding from interaction alone.
2. **The two faults of cards 124 and 141–145:** of held objects, at most
   5.0% share their group with another moved thing (the hand); of released
   objects, at most 5.0% are lost into the background's group.
3. **Small objects:** on the small split, objects found at least 85.0%.

The card passes if one arm meets all three. The arms differ if criterion
1's measure differs by more than 1.5 points; otherwise the cheaper is
carried forward. Reported without a bar: found in the first frame after an
object first moves, look-alike pairs, a ceiling of 16, memory and time.

Settings: the front end's two thresholds and two rules (a pixel must confirm
its window's reading; no reading may land where a point was seen to stay)
were set on validation episodes 50–79. The arms' numbers were fixed
beforehand and not tuned. The bars were written after seeing validation
(arm C 97.7% found, arm M 97.4%, true motion 97.9 and 98.2).

Prediction: both arms pass 1 and 2. Small objects are open: nine pixels
leave little margin, and arm M can run out of files when misread motion
opens spurious ones. Budget: about ten runs of one to two minutes.

## 7. Result

Percent found, 200 episodes each; numbers in [results.json](results.json).

| | Baseline, no memory | Arm C, couplings | Arm M, motion prediction |
|---|---|---|---|
| Gate: test, true motion | - | 98.4 | 98.9 |
| Gate: small, true motion, objects | - | 99.2 | 99.5 |
| **Test, motion from pixels** | 39.9 | **97.9** | **98.4** |
| the hand / objects | 73.1 / 16.7 | 98.6 / 97.4 | 99.5 / 97.7 |
| held objects sharing a group with the hand | 98.7 | **4.0** | **2.3** |
| released objects lost into the background | 95.1 | **1.8** | **1.8** |
| **Small split, objects** | 0.0 | **98.3** | **98.9** |
| found in the first frame after first moving | 56.7 | 100.0 | 99.5 |
| look-alike pairs | 17.2 | 96.9 | 97.2 |
| ceiling of 16 instead of 8 | - | 97.9 | 98.5 |
| numbers kept per episode / seconds for 200 episodes | - | 37.7 million / 78 | 49 thousand / 13 |

| Criterion | Arm C | Arm M |
|---|---|---|
| 1. Found ≥ 93.0%, within 3.0 of true motion | pass (97.9; 0.5 below) | pass (98.4; 0.5 below) |
| 2. Held with the hand ≤ 5.0%; released into background ≤ 5.0% | pass (4.0; 1.8) | pass (2.3; 1.8) |
| 3. Small objects ≥ 85.0% | pass (98.3) | pass (98.9) |

Both arms pass, and by the declared margin they do not differ (0.5 points).

What the misses are (test, arm M: 400 of 17,124 object-frames, in 38 of the
200 episodes; arm C alike): an object partly covered by another object or
the hand is found in 87% of frames against 98.4% when fully in view, and
partly covered frames are 6.4% of all. Most misses are a released object
read as background (279), then an object sharing the hand's group (106).
I did not trace these further.

Two things the table does not show.

- **The arms differ in what a misread motion costs.** Before the front
  end's two rules, on validation episodes 50–79, arm C found 96.4% and arm
  M 90.1%: three or four background pixels misread as moving alike opened a
  file, and the eight files ran out (268 times in 30 episodes; 9 times in
  200 after the rules). In arm C such a pixel leaves the background's group
  and costs nothing else. Arm M has no way to close or merge a file.
- **Evidence over time is the part that matters.** The baseline sees the
  same motion and finds 17% of objects; my first arm M, which kept only the
  last frame's grouping, found a quarter. Both arms as run keep one
  accumulated number per pair, of tokens or of token and file.

Limits: the world was built so that motion says everything (rigid, flat,
textured, a fixed camera), so near-ceiling scores were likely once the
bookkeeping was right, and the test does not separate the arms. The lock
rule, the button's change of colour and the action were not used: of the
user's three signals this card uses tracked motion, and events only in that
a file opens in the step an object is first carried.

## 8. Decision

**Keep.** Binding from the agent's own interaction works in this world
without labels, appearance or the number of objects, including for objects
3 pixels wide, held objects and look-alikes, and an object is bound in the
first frame after it is first moved. By the card's rule arm M (object files
with an accumulated prediction error) is carried forward as the cheaper,
with arm C's tolerance of misread motion noted as what M lacks. Nothing
enters og4. The next card should break the assumptions this world grants,
in the order of the portability table: plain surfaces and motion in depth,
which is the ladder's third step (OGBench's renderer), with point motion
taken from the simulator first so that binding is tested apart from
tracking.

**After the card (the user, 2026-11-19):** "I definitely lean towards the
coupling version since it is more true to the energy formulation and the
CRAM paper." Arm C is the form carried forward; this card's rule had picked
arm M as the cheaper.
