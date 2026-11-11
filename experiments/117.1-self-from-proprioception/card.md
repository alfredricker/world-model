---
id: "117.1"
title: the self per patch from proprioception — gate only, not passed
rung: 2
serves: [P7, P4]
status: draft
verdict:
arch_version: 27
date: 2026-11-10
---

# 117.1: the self per patch from proprioception

Roadmap step 2 (event segmentation), rung 2. Card
[125](../125-conditions-by-contrast-over-touches/card.md) left two
problems that both come from the arm: it hides a button just before a
touch there, and it appears as a "condition" at the button places.
Card [117](../117-self-and-boundaries/card.md) finds the self on
slots, which cards 116–124 did not deliver. This card looks for the
self per patch. Only the gate was run, and it did not pass, so
sections 6–8 are open.

Biological counterpart: the body schema is built from the agreement
of sight with proprioception (Botvinick and Cohen 1998), and infants
tell their own limbs by that contingency (Watson 1966, LITERATURE).

## 1. Question

Can the patches that show the agent's own body be found from its pose
(12 numbers: joint angles, effector position and yaw, gripper opening)
and its remembered frames, with no labels? P7, P4.

## 2. What changes

One component: a self mask per patch, to be applied to card 125's
records. Two forms were tried in the gate.

```
pose model:  pose → predicted features of every patch (a small network, 5 minutes of training)
             self = patches whose prediction depends on the pose
recall:      pose → the 8 or 16 remembered frames with the nearest pose
             self = patches unlike their usual look and like those frames
```

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The pose is given | position and view angles; the body is mostly out of view | the player's tile | none | holds | proprioception (GOAL C1) |
| The same pose shows the same body in the frame | holds for the held item | holds | not applicable | fixed cameras only | fixed camera, declared since card 119 |
| The scene varies independently of the pose | weak | weak | not applicable | weak: the arm is near what it handles | **failed here** (below) |

## 3. Dependencies

Cards 122 and 124 (256-pixel features, stored for 450 episodes). Chen
et al. 2026 (LITERATURE) match joint signals to the motion of given
masks; without masks this is not an established method.

## 4. Data check

- The arm covers 20% of the patches, and 47% of that is its base,
  which never moves. No rule based on the pose can find the base.
- The premise holds on the true masks: the arm's mask in the frame
  with the nearest pose (among 100,000) overlaps the true one with IoU
  0.90 (a random frame: 0.51).

## 5. Feasibility gate

Run 2026-11-10, each step under 10 minutes and capped. 400 validation
frames; threshold by Otsu on episodes 900–999. The bar is card 117's:
IoU at least 0.7 with the arm, at most 10% of other objects included.
Baseline: k-means on block 12 gives the arm 0.60 (card 122).

| Form | IoU with the arm | With the best threshold | Other objects included |
|---|---|---|---|
| Pose model: prediction depends on the pose | 0.27 | 0.29 | cube 72%, drawer 24% |
| Recall: nearer to pose-neighbours than to random frames | 0.30–0.32 | 0.35–0.38 | cube 26–29%, drawer 9–22% |
| Recall: unlike the usual look, like the pose-neighbours (block 12) | 0.22–0.23 | 0.39 | cube 16–18%, drawer 3–4% |
| The same, closest neighbour per patch | 0.24 | 0.33 | cube 63%, drawer 28% |

Leaving the base out of the scoring does not change the picture (best
0.35–0.36).

**Gate result: not passed.** Every form is below the appearance
baseline. What is established:
- The first two forms mark every patch the arm often covers, covered
  now or not.
- The pose is correlated with the scene: the cube and the drawer are
  near the gripper when it handles them, so they are predicted from
  the pose too.
- Measured by region: on the moving arm the frame is 1,305 from the
  usual look and 646 from its pose-neighbours (block 12, squared
  distance); on the background 335 and 294. The signal exists but the
  neighbours are not close enough to be sharp, and block 2 is worse.

Not established: whether a longer-trained pose model would be sharp
enough (its loss was 2.06, against 1.4–1.5 for card 124's
reconstruction).

## 6. Success criteria and prediction

Not written: the gate did not pass.

## 7. Result

None.

## 8. Decision

Open, for the user. Options: train the pose model for longer; or drop
the self for card 125's two problems and use stability instead (read a
condition only where its patches are steady, which the arm near a
touch never is).
