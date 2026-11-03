---
id: "102.5"
title: the agent as a token — found by what the actions move, its motion predicted like any token's
rung: 1
serves: [P7, P6, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 102.5: the agent as a token

Drafted 2026-10-11 as the fourth card of the 102 series, at the user's
request to remove MiniGrid's assumptions in a sensible order. Approved
and run overnight on 2026-10-12, under the user's overnight objective
(portability without significant loss). The design below was made
concrete before the run (section 2, "As run").

## 1. Question

The model treats the agent apart from every other token:
- **It is assumed to be at the centre of the view, which follows it.**
  MiniGrid's adapter turns the view so the agent faces up. Crafter's view
  is centred on the agent.
- **Its motion is a fixed transformation per action.** These are card
  061's fitted turns and shifts in MiniGrid and card 098.3's learned
  shifts in Crafter. The placing step only chooses between "the action's
  transformation happened" and "it did not".

ARC-AGI-3's games break both:
- a player sprite moves within a fixed view;
- in some games there is no body, and an action moves a group of tiles
  or nothing.

A robot's camera moves with its arm, but not as a fixed step.

Suppose the world frame is held by the tokens that do not move (102.3's
kinds of zero spread). Then every token's motion, the agent's included,
is read against that frame. The agent is the token or tokens whose
motion the actions predict best (contingency). Its motion is predicted
by the same rule as any token's, with the action as an input. Is the
agent's own motion then read as well as now, without a table per action?
And are the controlled tokens found in worlds where the view does not
follow them, or where there is no body? Rung 1. Serves P7 (the self as a
learned unit), P6 and P2.

## 2. What changes

One component: how the agent's own motion is found (the placing step,
level 1).

```
now:        the agent is the view's centre; its motion is one of {no change, the action's fixed transformation}
this card:  world frame = the tokens of kinds never seen to move (102.3)
            the view's placement in that frame = the transformation that best matches belief (as now),
              among those the motion rule predicts for the action, ranked by their probability
            controlled tokens = those whose motion in the world frame the action predicts best
              (the mutual information between action and motion, from stored steps)
            their motion: the motion rule of 102.3 with the action as an input, like any token's
```

- **One motion rule.** A cow's motion and the agent's are predicted the
  same way. For the agent, the action carries the information.
- **Blocked moves are predicted, not only detected.** The motion rule
  reads what stands in the way (through card 102.1's relations), so
  "forward into a tree does not move me" is learned. Today the placing
  step finds out afterwards.
- **The view's own motion is separate.** Whether the view follows the
  agent (Crafter), turns with it (MiniGrid) or stays fixed (ARC) is
  found from the frames, as card 061 found MiniGrid's turns. Nothing
  says which.
- **No body is allowed.** If no token's motion is predicted by the
  actions, there is no agent token, and nothing else assumes one.
- **Not in this card:** the agent's appearance as a condition (heading,
  asleep), which arm C already reads as a token; continuous motion.

**As run** (`tools/card102.5/self.py`). The placing step is one Bayesian
rule for every world. The view's shift s, among rigid shifts up to two
tiles each way, maximises log p(s | action) plus the mean over known
places of each place's log-likelihood:
- **p(s | action)** is learned from the agent's own placements (counts,
  Dirichlet 1). It replaces 098.3's fixed shift per action.
- **Each place** holds a still token, matched to belief at the matching
  noise b, or a token that moved, priced as new content by the data's
  spread. The mixture weight is its code's stillness, learned in a first
  pass. This is a standard outlier model (MLESAC, Torr and Zisserman 2000);
  it replaces 098.8's "leave out the centre tile".
- **Controlled codes:** a code's motion in the world frame is the shift
  that best overlays its tiles from one step on the next. U = I(action;
  motion) / H(motion) on training play, and U ≥ 0.5 is controlled
  (declared). This is Watson's contingency.
- **Not run:** blocked moves predicted before the frame, the controlled
  tokens' motion against the upper bound, and MiniGrid (its view turns,
  which needs rotations among the candidates).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The agent is at the view's centre | yes | no | no (hand-eye) | removed |
| The view follows the agent | yes | no | partly | removed: found from the frames |
| The agent's motion per action is a fixed transformation | almost (blocked) | no | no | removed: predicted by the motion rule |
| Some tokens never move, and they hold the world frame | yes | yes | yes (the room) | declared |
| Transformations are rigid | yes | yes | yes for the camera | declared (card 061) |

## 3. Dependencies

- Card 102.3's motion rule and its kinds of zero spread. Card 102.2's
  layers, where the background is the world frame.
- Card 061 (transformations fitted from the view) and 098.5's placing
  step (98% of Crafter's own moves; card 098.8's `place.py`).
- Literature, to be added: Watson 1966 and Gergely and Watson 1999, the
  self found by perfect contingency between action and motion.

## 4. Data check

From stored play (counted when the card is approved): Crafter's blocked
moves and their causes; MiniGrid's turns and blocked forwards.

**Breaking world 4** (`tools/card102/worlds.py`, Appendix A):
- **4a, "a body in a fixed view":** a player sprite moves in a fixed
  16 × 16 view, ARC-style.
- **4b, "no body":** the four arrow actions move every tile of one
  colour by one cell, and other tiles never move.

## 5. Feasibility gate

- **Upper bound:** the true agent position and the true view
  transformation (the evaluator's), for criterion 1's rate and
  criterion 2's motion.
- **Trivial baselines:**
  - the current placing step: the centre is the agent, with fixed
    transformations;
  - in 4a and 4b it has no way to place the view correctly, which is the
    assumption being broken.
- **Check:** in MiniGrid's stored play, the transformations the motion
  rule ranks first are card 061's, on every step where card 061's
  placing was right.

Result of the gate, before the main run:

## 6. Success criteria and prediction

1. **Own motion as well as now, without the table.** In Crafter's worlds
   8–9, the agent's own motion is read right on at least 98% of steps
   (098.5's placing step: 98.1–98.5%). Blocked moves are predicted before
   the frame in at least 80% of cases. MiniGrid's placements on stored
   tier play are identical to version 27's.
2. **The controlled tokens found.** In breaking worlds 4a and 4b, the
   tokens whose motion the actions predict are the evaluator's
   controlled tokens with precision and recall of at least 95%. Their
   motion is predicted right in at least 95% of the upper bound's rate.
3. **The view's motion found from the frames.** The view's own
   transformation is read right in at least 99% of steps in all three
   kinds of world:
   - Crafter's view follows the agent;
   - 4a's and 4b's view is fixed;
   - MiniGrid's view turns with the agent.

**Reported:** contingency per token kind; and in 4b, how long it takes
to find which tiles the actions move.

**Prediction.**
- Criterion 1 holds, since the placing step is unchanged in form and
  only its candidates come from the motion rule.
- Blocked moves are predicted well in front of trees and water, less
  well in front of creatures.
- In 4b the controlled colour is found within about 20 steps.

**Decision rules.**
- **Keep** if 1–3 hold.
- **Revise** if only the prediction of blocked moves fails.
- **Stop** if the world frame from never-moving kinds is wrong in any
  world, for example where everything moves.

**Budget.**
- Breaking world 4 takes about 2 minutes to build.
- Crafter's worlds 8–9 take about 10 minutes, and MiniGrid's stored play
  about 10 minutes, each timed first.

## 7. Result

Run 2026-10-12 (two minutes per world, capped). Breaking world 4b was
built (`worlds.py 4`; blue tiles moved in 134,058 of 200,000 training
steps), and world 2 stood for 4a.

| World | View | View's motion read right (bar 99%) | Controlled codes: precision, recall (bar 95%) | U: controlled, highest other |
|---|---|---|---|---|
| 2 (4a, a body in a fixed view) | fixed | 100% | 100%, 100% | 0.68, 0.22 |
| 4b (no body) | fixed | 100% | 100%, 100% | 0.65, 0 |
| 3 (out of sight) | follows | 100% | 100%, 100% | 0.82, 0.15 |
| Crafter, worlds 8–9 | follows | 93.0% (bar 98%: 098.5's 98.1–98.5%) | 100%, 100% (5 player codes) | 0.53–0.55, 0.41 |

- **The prior learns each world's view.** In worlds 2 and 4b every
  action's most probable shift is none; in world 3 and Crafter each arrow
  gets its shift and every other action none. Nothing said which.
- **Crafter's misses are real moves read as none** (76.8% of moves read
  right, 97.5% of non-moves), mostly at night (67.9% against 95.7% by
  day) and on uniform ground.
- **The likelihood mattered.** Summing mismatch only over known places
  favoured shifts into the unseen (89.6%). Pricing every unseen place as
  new content favoured "no move" (89.1%). Pricing it by the last view's
  tokens drifted (84.3%). The mean over known places with the outlier
  mixture gave 93.0%. A per-place weight by stillness alone did not help.
- **A noise per step** (`--adapt`: the view's own median mismatch,
  floored at b) gave 94.5% (night 76.9%, day 96.4%), but the player's
  codes then fell to U = 0.48–0.57 and only one of five passed U ≥ 0.5.
  In Crafter, where many moves are blocked, a fixed threshold on U is
  fragile (the next-highest code: 0.48, seen 78 times).
- **The controlled codes left out of the match** (`--world-only`, after
  the first pass) gave 92.9%. The agent's own tokens are not what hides
  Crafter's moves here; the cause is open. Card 102.8, reading the motion in
  pixels, reached 94.9%, with 89.9% at night.

## 8. Decision

**Revise** (overnight, 2026-10-12). Criterion 2 holds in every world: the
agent is found as the codes the actions move, with no table per action,
no centre and no body assumed. Criterion 3 holds in worlds 2, 3 and 4b.
Criterion 1 fails in Crafter (93.0% against 98%): its own motion at night
and on uniform ground. One matching noise b for day and night is the
likely cause (at night every place mismatches at day's scale); a noise
per step recovers part of it (94.5%). The next revision learns the noise
from what the agent sees, replaces the fixed U threshold by a test
against shuffled actions, and adds rotations so that MiniGrid's turning
view is read by the same rule.

## Appendix A. Breaking world 4

- **4a:** a 16 × 16 room seen whole. The player sprite moves one cell
  per arrow action unless blocked by a wall, and boxes are pushed as in
  breaking world 2.
- **4b:** the same rooms with no player. Each arrow action moves every
  blue tile one cell, where free. Red tiles never move.
- 500 episodes of 200 random actions each, training and test.
