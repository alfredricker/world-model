---
id: "102.8"
title: the view's motion in pixels, then patches fixed in the world
rung: 1
serves: [P7, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 102.8: the view's motion in pixels

Drafted and run overnight on 2026-10-12, following card
[102.7](../102.7-tokens-off-the-grid/card.md)'s finding that the tile-grid
prior lives in one place: the view's motion is read only in whole tokens.

## 1. Question

Suppose the view's motion is read in pixels before the frame is cut into
tokens, and the patches are then cut on a lattice fixed in the world. Is
the motion read as well as on Crafter's own tiles? And does a world patch
keep its code across a move, whatever the patch size? If both, no token
needs to fit the world's grid, which Minecraft and robots require. Rung 1.
Serves P7 and P2.

## 2. What changes

One component: where the placing step reads the view's motion.

```
102.5:      shift in whole tokens = argmax  log p(shift | action) + match of token vectors to belief
this card:  shift in pixels (-9..9 each way) = argmax  log p(shift | action) - SSD / (2 sigma^2) - 3/2 x unseen pixels
              over the frame's world region against a pixel canvas of belief; p learned from own placements
            then P x P patches cut on a lattice fixed in the world (phase = minus the origin, modulo P), P = 8
```

- **Unseen pixels** cost what a matched pixel costs on average (3 channels
  × ½ nat), so a shift into the unseen is neither favoured nor punished.
- **The step length is not declared.** The candidates are every pixel
  shift up to 9, and the prior learns which ones an action makes.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The view's motion is a 2-D shift | yes | yes (none) | no (rotation, depth) | declared; rotations and scale are the next candidates |
| Tokens fit the world's grid | yes | yes | no | removed |
| Motion per step under 10 pixels | yes | yes | depends on the rate | declared (the candidate window) |

## 3. Dependencies

Card 102.5's placing rule; card 102.7's patch cutting and simplified
encoder recipe; card 098.5's Crafter play (worlds 0–2 training, 8 test).

## 4. Data check

24,204 test steps, 22% of them moves. Crafter's tiles are 7 pixels.

## 5. Feasibility gate

- **Upper bound:** 098.5's placing step, with the action's shift given and
  the agent's tile left out by name (98.1–98.5%).
- **Trivial baseline:** "the view never moves" (78.0%).

## 6. Success criteria and prediction

1. **Motion in pixels as well as in tiles.** The view's motion is read right
   on at least 98% of test steps (098.5's level), and at least 95% at night.
2. **Patches fixed in the world keep their code.** Across a move, 8-pixel
   world patches keep their code within 3 points of their rate across a
   still step.

**Prediction:** both hold, since a 7-pixel move is a clean pixel shift.

**Decision rules:** keep if both hold; revise if only night fails; stop if
pixel shifts read moves no better than whole tokens did (102.7's C: 0%).

## 7. Result

Run 2026-10-12 (`tools/card102.8/register.py`, 2.5 minutes, capped).

| | View's motion read right | Moves / non-moves | Day / night |
|---|---|---|---|
| 102.5, token placing (098.5's tokens) | 93.0% | 76.8% / 97.5% | 95.7% / 67.9% |
| 102.7, whole 8-pixel patches | 78.0% | 0% / 100% | |
| **This card, pixels** | **94.9%** | 78.7% / 99.5% | 95.5% / 89.9% |

- **The step length was learned.** The prior's likeliest shift per arrow is
  exactly ±7 pixels, the tile size, which nothing declared.
- **Patch size no longer matters for motion** (102.7's C read 0% of moves).
  Night is much better than on tokens (89.9% against 67.9%).
- **Criterion 1 fails** (94.9% against 98%). The misses are real moves read
  as "no move" (930 of 1,232 misreads). On uniform ground the only pixels
  that differ between the two hypotheses are the agent's own sprite, which
  stays at the view's centre and so argues that nothing moved. Two attempts
  did not help: scoring every candidate on the same central pixels (87.4%),
  and learning the prior from the posterior (EM, the same). The prior
  collapsed to "no move" in both, because the evidence itself favours it.
- **Criterion 2 fails:** world patches kept their code 82.0% of the time
  across a move, against 93.8% across a still step. Misplaced moves move the
  lattice, so part of the gap is criterion 1's.

## 8. Decision

**Revise** (overnight, 2026-10-12). Reading motion in pixels removes the
tile-grid prior and helps at night. What remains is not a grid prior: the
agent's own sprite argues against the view's motion. The revision leaves
the controlled tokens (102.5's contingency) out of the match, so that the
world frame is measured on the world alone. This is the general form of
098.5's "leave out the agent's tile".
