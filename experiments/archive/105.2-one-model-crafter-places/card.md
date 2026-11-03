---
id: "105.2"
title: one transition model in Crafter — every action, tiles placed in the world's frame
rung: 1
serves: [P6, P3, P12]
status: done
verdict: revise
arch_version: 27
date: 2026-10-13
---

# 105.2: one transition model in Crafter

Series [105](../105-one-transition-model/card.md), approved for the overnight
session of 2026-10-13. Offline; no agent.

## 1. Question

Can 105.1's model, unchanged, predict every one of Crafter's 17 actions in one
model? That covers moves and being blocked by water, trees and stone; mining
and collecting; placing; crafting at a table; and doing nothing. The tokens
are placed in the world's frame, so the agent moves among them and is not
fixed at the centre of a view. Card 102.1 left moves out, and its role test
could be passed position by position. Rung 1; P6, P3, P12.

## 2. What changes

105.1's model on Crafter's tokens.
- **The 63 view tiles** (card 098.3's encoder) sit at world positions:
  103.4's learned origin plus the tile's place in the view. A tile is the
  same token at t and t + 1 when its world place is the same.
- **Tiles that leave the view** have no target. Tiles that enter it are not
  predicted (that is belief's work, 105.3).
- **The 18 inventory tiles** are tokens in the screen's frame. Positions
  are compared only within a frame.
- **Every awake step is used,** moves included. Sleeping steps are left out
  (sleep darkens the frame).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Tokens are tiles cut on the renderer's grid | yes | yes (cells) | no | declared adapter (105.3 uses objects) |
| The view's motion from perception (103.4) | yes | yes | yes, with odometry | declared |
| Two frames (world, screen) | yes | yes | yes (world, gripper) | declared, from perception |

## 3. Dependencies

- 105.1's model, kept or revised.
- 103.4's placing (kept): origins for 098.5's workers 0, 1, 8 and 9 exist.
  The crafting curriculum (102.1, worlds 0–9) is placed by the same tool
  before the run.
- 098.3's encoder.
- 102.1's role test and its numbers as the reference.

## 4. Data check

- **Random play:** 098.5's workers 0–1 for training and 8–9 for the test,
  about 24,000 awake steps each.
- **Crafting curriculum:** 102.1's worlds 0–7 (101,618 steps, 989 tools
  made) for training and 8–9 (25,467 steps, 236 tools) for the test.
- Move steps and blocked moves are counted before the run.

## 5. Feasibility gate

- **Upper bound:** the same model on the evaluator's kinds, counts and
  heading as one-hot tokens, in the world frame.
- **Trivial baseline:** "nothing changes".
- **Reference:** 102.1's first run, on non-move steps.

## 6. Success criteria and prediction

Scored by the evaluator on held-out worlds:
1. **Every action in one model.**
   - **Make:** of the tries that made a tool, the inventory is right in at
     least 90% (102.1: 96.2%; a 6-point margin).
   - **Do and place:** the scored next state is right in at least as many
     steps as under "nothing changes", plus at least 80% of the changes
     that happened.
2. **Moves in the same model.** The agent's next world place is right on at
   least 98% of move steps, blocked ones included. Perception's own label is
   right on 99.3%.
3. **A table at new places.** As in 102.1, the table is moved to each of 24
   places around the agent. The predicted inventory shows the tool at the 8
   near places and not at the 16 far ones, in at least 95% of the tests
   (102.1: 93.9%).

**Prediction:** 1 and 2 hold. 3 passes, because in the world's frame "near"
cannot be a fixed place on the screen.

**Decision rules:**
- **Keep** if 1–3 hold.
- **Revise** if one fails by less than its margin.
- **Stop** if 2 fails by more than 5 points.

**Budget:** placing the curriculum takes about 20 minutes over ten short
runs. Training takes about 20 minutes per run, capped.

## 7. Result

Run 2026-10-13 with arm Rb, 105.6's form ([results.json](results.json)).
- **Placing.** 103.4's placing, as card 103.5 fitted it, placed the
  curriculum's ten worlds right on 99.3–99.5% of steps.
- **Data.** 132,785 training and 69,685 test transitions, every awake step.
- **Speed.** Training ran at 7 ms a step, because the bias is shared: the
  position differences within the view are the same in every state.

**The gate,** with the evaluator's kinds as one-hot tokens and 30,000 steps:

| Measure | Gate | "Nothing changes" | Bar (test) |
|---|---|---|---|
| Makes that made a tool: whole scored state right | 57.6% | 0% | ≥ 90% |
| Changed tokens right: make, do, place | 96.7%, 97.5%, 99.5% | 0% | ≥ 80% |
| Whole scored state right: no-op, make, do, place, move | 45%, 47%, 7%, 43%, 47% | 99.8%, 98.8%, 88.4%, 92.2%, 97.1% | at least "nothing changes" |
| The agent's next place: moved / blocked | 100% / 67.3% | | ≥ 98% overall |
| Role test: made at all 8 near places and none of 16 far | 93.4% (196 tests) | | ≥ 95% |

- **Gate: fail.** The upper bound gets the changes right but breaks what
  does not change. On half of the no-op steps it predicts some unchanged
  token changing, almost always grass predicted to become a cow (3,753 of
  3,872 such tokens).
- **The cause is the loss, not the model's form.** 105.1's loss gives the
  tokens that change half the total weight. A change is rare, so it
  counts about 100 times as much as a token that stays. In world 1 every
  change followed from the state, and the weighting only sped up learning.
  In Crafter cows wander at random. Weighted squared error is minimised by
  a weighted mean, so the rare chance that a cow steps in outweighs the
  likely outcome that the grass stays grass. The model then predicts the
  cow confidently: distance 0.15 to the cow code, 1.27 to its own. Blocked
  moves (67%) show the same pull toward the usual outcome, the agent
  moving in.
- **The test was not run;** the token runs were stopped when the gate's
  cause was found.

## 8. Decision

**Revise** (2026-10-13). The gate failed for a reason outside what this
card tests: the loss's weighting, inherited from 105.1, favours change
wherever changes are random. Card [105.7](../105.7-plain-loss/card.md)
changes that one component. Every token with a target weighs the same, so
the model predicts the expected next vector. It reruns this card's gate
and test unchanged. The spread of outcomes stays with recall (105, section
2), not with a weight that tilts the point prediction.
