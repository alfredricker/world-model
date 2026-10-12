---
id: "103.1"
title: the world layer and the camera layer — the view's motion with a robust likelihood, the camera's pixels learned
rung: 1
serves: [P7, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 103.1: the world layer and the camera layer

Drafted and run on 2026-10-12, in series
[103](../103-tracking-with-general-priors/card.md), under the user's
instruction to propose and run the series without waiting for approval.
The method took shape over smoke tests of up to 8,000 steps (section 5).
The criteria below were fixed before the full run.

## 1. Question

Suppose each pixel of a frame is explained either by the world (a canvas
seen through a shifting window) or by content the camera carries (the
same view position in the last frame), with a likelihood that caps any
one pixel's vote. Then:
- Is the view's motion read as well as when the agent's tile is declared
  (98%)?
- Does the model find, unaided, which pixels the camera carries (the
  agent's sprite)?

Rung 1. Serves P7 and P2. The view's motion is the frame every other
memory is placed in.

## 2. What changes

One component: the likelihood of card 102.8's pixel placing.

```
102.8:      score(s) = log p(s|a) - SSD(s)/(2 sigma^2) - 3/2 x unseen pixels         (every pixel the world's)
this card:  score(s) = log p(s|a) + sum_{q in S} log( pi_q P_cam(q) + (1 - pi_q) P_world(q|s) )
            P_world, P_cam: (1 - eps) N(residual; sigma_t^2) + eps, after removing a global brightness offset
            S: pixels known under every shift the prior allows (same evidence for every hypothesis)
            pi_q: share of view position q carried by the camera, by EM over steps where the view moved,
                  counted only on world content confirmed from two view positions; then fixed
```

- **The brightness offset** is the exposure term of direct visual
  odometry. Crafter's renderer truncates, so every pixel drops one level
  whenever daylight dips below 1.
- **Arm "none":** every pixel is the world's. This is the robust
  likelihood alone.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The world moves by one 2-D shift per step | yes | yes (none) | no | declared; card 103.3 keeps it |
| Some pixels move with the camera | yes (agent) | yes (HUD) | yes (gripper in a wrist camera) | learned, not declared |
| The agent's tile is the agent | yes | no | no | removed (098.5 declared it) |
| A global brightness change between frames | yes | yes | yes (exposure) | general |
| The adapter crops the world region (49 × 63) | yes | depends | no | declared; the full frame was not run |

## 3. Dependencies

- Card 102.8's placing.
- Card 102.5's learned prior p(s | a).
- Robust matching as a mixture (MLESAC, LITERATURE).
- The exposure term of direct visual odometry.
- Card 098.5's Crafter play: workers 0–1 for training, 8 for testing.

## 4. Data check

24,204 test steps, 22% of them moves; night (light < 0.3) is 7% of them.

## 5. Feasibility gate

- **Upper bound:** 098.5's placing with the agent's tile declared reads
  98.1–98.5%.
- **Trivial baseline:** "the view never moves" reads 78.0%.
- **Gate, run today:** 102.8's likelihood with the agent's tile masked out
  by the evaluator reads only 91.4% (moves 88.2%, still steps 92.3%).
  Without the sprite, still steps fail.

**What the smoke tests settled while building:**
- **Estimating pi from its own placements feeds on itself.** A pixel judged
  camera content was no longer written, so it went stale, the camera
  explained it better, and every pixel ended as the camera's. So pi is
  estimated once, from placements made with every pixel the world's, and
  only on world content confirmed from two view positions.
- **The canvas takes every world pixel's latest value.** Writing only
  pixels the world explained left stale pixels behind.

## 6. Success criteria and prediction

1. **View's motion.** At least 98% of test steps right, and at least 95% at
   night.
2. **The camera's pixels found.** At least 90% of the pixels with pi > 1/2
   lie in the agent's tile, and none in the rest of the world region.

**Prediction:** both hold, since the sprite's vote for "no move" was 102.8's
main miss.

**Decision rules:** keep if both hold; revise if criterion 2 holds and 1
fails.

## 7. Result

Run 2026-10-12 (`tools/card103.1/place.py crafter --pi learned|none`,
about 7 minutes each, capped).

| | Right | Moves / still steps | Day / night | Camera pixels |
|---|---|---|---|---|
| 102.8, pixels (reference) | 94.9% | 78.7% / 99.5% | 95.5% / 89.9% | none |
| Robust likelihood, every pixel the world's | 94.7% | 77.8% / **99.4%** | 94.7% / **94.0%** | none |
| **Robust likelihood, camera layer learned** | 90.6% | **89.8%** / 90.8% | 92.6% / 72.4% | 26: all in the agent's tile |
| Gate: 102.8 with the agent's tile masked by the evaluator | 91.4% | 88.2% / 92.3% | 92.9% / 77.6% | 49 given |

- **Criterion 2 holds.** The model learned that 26 pixels move with the
  camera, all of them in the agent's tile (its sprite), and none elsewhere.
  Nothing named the agent, the centre or a tile.
- **Criterion 1 fails.** With the sprite left out, moves are read far better
  (78% → 90%), but still steps fall (99% → 91%).
- **Why still steps fall:** the misread still steps are mostly not moves
  (noop, "do", place); the prior says "no move". Traced steps show a
  creature that stepped one tile on plain grass. Its 44 pixels are matched
  exactly by the view shifting the other way. Each pixel counts as
  independent evidence (about 19 nats each), so together they outvote the
  prior (3–5 nats).
  - From two frames the two explanations are equally good. What separates
    them is that a cow stepping is common and the view moving under "noop"
    is not.
  - The agent's own sprite had been hiding this: it always voted "no move".
- **The robust likelihood alone helps at night:** 94.0% against 89.9%
  (102.8) and 67.9% (102.5).

## 8. Decision

**Revise**, as card 103.3 (2026-10-12). The camera's pixels are learned
exactly, which removes 098.5's declared agent tile. The view's motion still
fails because evidence is counted per pixel, not per thing that moved. A
mover's pixels must be explained by the mover, not by the view. So the
view's motion is read after card 103.2's files claim their pixels, which is
the layered model the literature recommends (SfMLearner's mask, EISEN's
subtraction). The robust likelihood is kept: it lifts night by 4 points.
