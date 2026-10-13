---
id: "103"
title: tracking with general priors — the series' plan
rung: 1
serves: [P7, P2, P6]
status: draft
verdict:
arch_version: 27
date: 2026-10-12
---

# 103: tracking with general priors

Opened 2026-10-12 by the user, after series 102's overnight cards: "do the
review of object centric video models / tracking … then propose a series of
up to 5 cards … priors should be limited and general." This card is the
series' plan; each sub-card is its own experiment.

## 1. Question

Series 102 removed most of MiniGrid's assumptions. What is left concerns
**correspondence**: which part of this frame is which part of the last one.
- **The view's motion:** Crafter's is read right on 93–95% of steps; the bar
  is 98% (cards 102.5, 102.8).
- **Identity:** a creature keeps its token 61–94% of the time per step and
  about 10% at return without layers.py's written rules (card 102.4).
- **Middles of one-colour shapes:** 0% (the aperture problem).

Version 27 has no object tracker. Belief assumes a static world, and
"identity" means "the same code". Cards 098.5–098.6 built object files on
tile tokens and reached an identity score (IDF1) of 0.717, against 0.934 on
the true state. Identity there was carried by tile codes, and a creature's
tile includes the ground beneath it (LESSONS, "Tokens from frames").

The series asks whether one tracker, built on a handful of general priors,
gives the view's motion and object identity on Crafter. It must have no
tiles, no kinds and no ground. It must also keep working on the breaking
worlds and carry over to ARC-AGI-3, Minecraft and robots by changing only
the adapter. Serves P7 (units tracked across frames, from continuity), P2
(belief about what is out of view) and P6 (the units feed prediction).

## 2. What the literature says

Reviewed 2026-10-12: about 45 papers on object-centric video models and
tracking (LITERATURE, current focus).
- **Slot models need labelled cues or pretrained features.** Slot-attention
  video models (SAVi, STEVE, VideoSAUR) need first-frame cues or pretrained
  features (DINO), and GPU-days. Without cues SAVi falls from 78 to 48 FG-ARI
  on MOVi; with a moving camera, from 19 to 3.
- **Propagation plus discovery fits small budgets.** These models (SCALOR,
  SILOT, AXIOM) carry each object forward with an explicit presence and
  start new ones only from what the tracked ones leave unexplained. They
  work on dozens of small sprites within minutes. All assume a fixed camera.
- **Motion makes a better teacher than appearance.** Using motion as the
  training signal lifted Atari detection from 4 to 95 (MOC) and TRI-PD from
  10 to 51 (DOM). Subtracting the known mover finds the object it moves
  (EISEN).
- **Content needs a way to be hidden or revealed.** The forward–backward
  check, a "no match" bin, layers with an order, or presence variables all
  give one. Card 098.8's random walk had none, so "they swapped" beat "it
  moved".
- **A layered sprite model is the closest match to Crafter** (Jojic and Frey
  2001): an appearance, a mask and a transformation per layer.
- **Ambiguous steps need the action and time.** An action-conditioned prior
  on the view's motion helps (+8 points of navigation success), and so does
  smoothing over several frames. Pixels the agent controls are left out.

## 3. The priors, all general

| Prior | Statement | Crafter | ARC-AGI-3 | Minecraft, robots |
|---|---|---|---|---|
| Continuity | Content moves a bounded distance per step; the distribution is learned | yes | yes | yes (at the frame rate) |
| Cohesion | A file's pixels share one displacement per step | yes | yes | rigid parts only; declared |
| Persistence | Files and the world persist unseen; existence is earned and lost | yes | yes | yes |
| Contingency | Some content moves with the camera (the agent, a HUD); learned | yes | yes | yes (hand, gripper, HUD) |
| Exposure | A frame may differ from the last by a global brightness offset | yes | yes | yes |
| The world as one layer | The static world moves by one 2-D shift per step | yes | yes | no: rotation and parallax; declared |

None of them names a tile, a kind, a ground or an object size.

## 4. The cards

| Card | Component | Test | Status |
|---|---|---|---|
| [103.1](../103.1-world-and-camera-layers/card.md) | The world layer and the camera layer: robust likelihood, exposure offset, the camera's pixels learned | View's motion on Crafter ≥ 98%, night ≥ 95%; the camera's pixels found | **revise**: the camera's 26 pixels found exactly; 90.6% (robust alone 94.7%, night 94.0%) |
| [103.2](../103.2-object-files-as-sprites/card.md) | Object files as sprite layers: propagation plus discovery with existence, on pixels | IDF1 on Crafter's movers against 098.6's 0.717 | **revise**: 0.531 (learned origins 0.303) |
| [103.3](../103.3-view-motion-with-files/card.md) | The view's motion with files explained away | 103.1's criteria, with 103.2's files | **stop**: false files remove the view's evidence (79.4%) |
| [103.4](../103.4-evidence-per-observation/card.md) | Evidence counted per independent observation: the likelihood tempered by τ, chosen by self-consistency | 103.1's criteria, τ chosen without labels | **keep**: 99.3% on held-out play (night 96.9%), τ = 300 chosen by self-consistency; world 3 100% |
| [103.5](../103.5-files-level-on-sprites/card.md) | Files on the right frame: 103.2's revision (sprite templates under 098.6's files level), on 103.4's origins | IDF1 against 0.717 | **stop**: 0.623 (evaluator's origins 0.641); detections on no creature 4.3% |

**Order.** 103.1 gives the frame of the world; 103.2 needs it. 103.3 tried
to join the two by letting files explain their pixels away. On learned
origins most files were false, so it removed the view's evidence. 103.4
attacks the same failure at its root: a mover's pixels are not independent
observations. One-colour shapes (the aperture problem) and files in belief
(card 102.3's memory in place of layers.py) wait for files that hold.

**Series closed** (2026-10-13). The view's motion is solved by general
rules (103.4, kept). Object identity is not: the best hand-built tracker
reached IDF1 0.62–0.64, against 0.717 on tile tokens and 0.934 on the true
state. The user then amended GOAL C1 so that a declared, pretrained
perception front end may supply objects' outlines and correspondence.
Card [104](../104-sam-correspondence/card.md) gives that job to SAM 2.1.
What carries over: 103.4's view motion, 103.1's camera pixels, and 103.5's
earned existence and scoring.
