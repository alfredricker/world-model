# Overnight 2026-10-12

Objective (the user): the encoder and model as portable as possible between
MiniGrid, Crafter, ARC-AGI-3, Minecraft and robotics, without losing
significant performance; judged on Crafter; principled enough for a paper.
Every run capped (CHARTER rule 5).

## 102.4 revision 1: minimum description length

[Card 102.4](experiments/102.4-where-content-went/card.md), revised and
closed for tonight. The user's general rule: explain each step's change by
the shortest description (MDL). Copying costs at most log 63 nats, drawing
hundreds, and ways are no longer blended. Fates are read from observed
steps by the posterior.
- **World 2:** pushed shapes of several colours are read right 82.8% of the
  time (0% before), the agent 99.99%, all object tiles 94.4% (bar 95%).
  One-colour middles stay 0%, the aperture problem as predicted.
- **Moving views:** the network does not converge in world 3 or Crafter
  (Crafter: false groups 81%, identity 15% against the rules' 81%).
- **Decision: revise.** layers.py stays a declared exception. The next
  revision places the view first (card 102.5) and asks the pointer only for
  motion in the world frame.

## 102.5 the agent as a token

[Card 102.5](experiments/102.5-the-agent-as-a-token/card.md). Placing as
one Bayesian rule: shift = argmax log p(shift | action) (learned from own
placements) + the mean robust likelihood of the known places (still or
moved, by each code's learned stillness). The agent is the codes whose
world-frame motion the action predicts (contingency, U ≥ 0.5). Breaking
world 4b (no body) was built.
- **Worlds 2, 3 and 4b:** the view's motion is read 100% right; the
  controlled codes are found exactly (precision and recall 100%).
- **Crafter:** the player's 5 codes are found exactly, but own motion is
  read 93.0% right (bar 98%; 098.5: 98%), with misses at night and on
  uniform ground. A noise per step gave 94.5% but broke the U threshold.
- **Decision: revise** (night noise, a U test against shuffled actions,
  rotations for MiniGrid).

## 102.6 actions with a where

[Card 102.6](experiments/102.6-actions-with-a-where/card.md). Breaking
world 5 ("clicks") was built: shapes recolour as wholes, lamps toggle,
and column 6 is never clicked in training. An action is a kind plus an
optional where, which enters 102.1's pair function as each token's offset
from it.
- **Results:** lamps 99.8%, all steps 94.6% ("nothing changes": 82.8%),
  clicks in the never-clicked column 73.6% (elsewhere 67.8%); without the
  where, 0%.
- **Shapes:** 64.6%. Which tiles form a shape is not visible in one frame;
  102.2's groups are needed.
- **Decision: revise** (attach the groups; run Crafter's "do" and MiniGrid).

## 102.7 tokens off the grid (new)

[Card 102.7](experiments/102.7-tokens-off-the-grid/card.md), a
diagnostic of the tile-grid prior: Crafter cut into 7-pixel patches at the
tiles' phase (A), at a phase 3 px off (B), and into 8-pixel patches (C).
Each form used the same encoder recipe and card 102.5's placing.
- **Results:** the view's motion read right 95.5% (A), 94.4% (B) and 78.0%
  (C). C reads 0% of moves, the same as "the view never moves".
- **Finding:** phase costs about 1 point and size costs everything. The
  grid prior lives in "the view moves by whole tokens".
- **Decision: keep (as a finding).** Next: read the view's motion in pixels,
  then cut patches on a lattice fixed in the world.

## 102.8 the view's motion in pixels (new)

[Card 102.8](experiments/102.8-motion-in-pixels/card.md). The view's
motion is read in pixels (102.5's rule on a pixel canvas, every shift up to
9 px), and patches are then cut on a lattice fixed in the world.
- **Motion:** read right 94.9% (bar 98%; 102.7's whole 8-px patches 78.0%,
  tokens 93.0%); night 89.9% (tokens 67.9%). The prior learned exactly
  ±7 px per arrow, the tile size, with nothing declared.
- **Patches:** 8-px world patches keep their code 82.0% across a move and
  93.8% across a still step.
- **Misses:** real moves read as "no move". On uniform ground the agent's
  own sprite, fixed at the view's centre, argues that nothing moved.
- **Decision: revise.** Leave the controlled tokens (102.5's contingency)
  out of the match.

## 102.4 revision 2: the view placed first

[Card 102.4](experiments/102.4-where-content-went/card.md). The view's
shift is read by card 102.5's placing, the next frame is aligned to the
world, and the MDL pointer explains only world-frame motion. Copy is now
priced by its best source when reading an observed step.
- **Groups:** false groups 0.01% (world 3) and 0.02% (Crafter), against 81%
  before; criterion 2 holds. World 2's moved shapes of several colours are
  one token 71% of the time (18% before).
- **Tracking:** identity at return is 10–11% against the rules' 81–98%. With
  every tile a token, belief files the grass too; which tokens deserve a
  file (the user's criterion) is next. Grass and walls do count as static
  (move rates under 1%); the loss is in movers' identity from step to step
  (Crafter 61–94% by kind; world 3's ball 8%). A fifth way, "move and
  change", was never chosen and changed nothing.
- **Decision: revise, closed.**

## Where the night ends

- **Cards:** five run (102.5, 102.6, 102.7 and 102.8 new; 102.4 revised
  twice). Nothing adopted: architecture still version 27.
- **Portable now, measured:**
  - the agent found as what the actions move, in fixed views, following
    views, with no body, and in Crafter;
  - the view's motion read by one learned rule: 100% in breaking worlds,
    93–95% in Crafter;
  - motion read in pixels, so tokens need not fit the world's grid (the step
    length learned);
  - actions that carry a place, whose effects carry to cells never clicked;
  - objects from the shortest description, not written rules, in fixed views,
    and with no false groups once the view is placed.
- **Not yet:** Crafter's own motion at 98% (moves on uniform ground);
  identity of creatures from step to step under the pointer; one-colour
  middles; shapes under clicks (they need groups).

# Afternoon 2026-10-12: series 103, tracking with general priors

The user's instruction: review object-centric video models and tracking,
reread GOAL and CHARTER, then propose and run a series of up to five cards
on portability with limited, general priors, without waiting for approval.
Every run capped (CHARTER rule 5). Plan:
[card 103](experiments/103-tracking-with-general-priors/card.md).

**The review** (about 45 papers) is in LITERATURE's current focus. Its
recommendations:
- **Propagation plus discovery with explicit existence.** Carry each object
  forward and start new ones only from what is unexplained (SCALOR, SILOT,
  AXIOM). Slot models need cues or pretrained features and GPU-days.
- **Motion as the teacher; subtract known movers** (MOC, DOM, EISEN).
- **A way for content to be hidden or revealed.** Card 098.8's random walk
  lacked one.
- **For the view's motion:** the action's prior, smoothing over time, and
  leaving out what the agent controls.

## 103.1 the world layer and the camera layer

[Card 103.1](experiments/103.1-world-and-camera-layers/card.md). Each pixel
is explained by the world (a canvas) or by the camera (the same view
position). The likelihood is robust and removes a global brightness offset.
- **The camera's pixels are learned exactly:** 26, all in the agent's tile.
  This removes 098.5's declared agent tile.
- **The robust likelihood alone reads 94.7%,** with night at 94.0%
  (102.8: 89.9%).
- **With the sprite left out, 90.6%:**
  - moves rise from 78% to 90%;
  - still steps fall from 99% to 91%.

  A cow stepping on plain grass now outvotes the action's prior: 44 pixels
  at about 19 nats each, against 3–5.
- **Decision: revise.**

## 103.2 object files as sprite layers

[Card 103.2](experiments/103.2-object-files-as-sprites/card.md). Files are
pixel templates with masks, followed near where they were. New ones are
born only from what is unexplained, and the canvas learns the ground a
mover reveals.
- **IDF1 0.531 with the evaluator's origins**, against 0.717 for 098.6 on
  tile tokens. With learned origins, 0.303.
- **Where it loses:**
  - 13% of creature frames: the creature has not moved yet, so it has no
    file (098.6's kinds gave it one);
  - 22%: lost after moving (median file life 10 frames);
  - 25% of detections sit on scenery.
- **Decision: revise.** Sprite templates go under 098.6's files level.

## 103.3 the view's motion with files explained away

[Card 103.3](experiments/103.3-view-motion-with-files/card.md). Pixels near
a file are left out of the view's evidence.
- On learned origins most files are false (ID precision 0.11). They leave
  out two thirds of the view: 79.4%, and moves at 57%.
- Carrying the canvas through sleep likely cost 8 points: the world changes
  while the agent sleeps.
- **Decision: stop.**

## 103.4 evidence per independent observation

[Card 103.4](experiments/103.4-evidence-per-observation/card.md). The
placing likelihood is divided by τ, the number of pixels per independent
observation. τ is chosen without labels: the value whose placements leave
re-seen world content most often as written.
- **Self-consistency picked τ = 300,** which was also the evaluator's most
  accurate on training.
- **Held-out worker 8: 99.3%** (moves 99.6%, still steps 99.2%, night
  96.9%), against 98.1–98.5% with the agent's tile declared. Breaking world
  3: 100%.
- **Files on these origins:** IDF1 0.411 (0.303 on 103.1's, 0.522 on the
  evaluator's).
- **Decision: keep.** Not adopted as a version: that needs MiniGrid's tiers,
  and MiniGrid's view turns.

## Where the afternoon ends

- **Cards:** four run (103.1–103.4), plus the plan (103). One keep, two
  revise, one stop. Architecture still version 27.
- **Solved:** Crafter's view motion by general rules. The camera's own
  pixels are learned, the likelihood is robust with an exposure term, and
  evidence is counted per independent observation with the count chosen by
  self-consistency. No tile, agent or grid declared.
- **Not yet:** an object tracker better than 098.6's files on tile tokens.
  Pixel sprites separate a creature from its ground, but without kinds and
  earned existence they keep identity worse (0.53 against 0.72).
- **LESSONS is at 287 lines** (276 before), above what three pages allow; it
  needs a pruning pass.
