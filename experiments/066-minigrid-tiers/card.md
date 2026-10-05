---
id: "066"
title: CHARTER's three MiniGrid tiers, and version 14 on them
rung: 1
serves: [P18, P17, C1, C3]
status: draft
verdict:
arch_version: 14
date: 2026-10-05
---

# 066: CHARTER's three MiniGrid tiers, and version 14 on them

The user moved the standing evaluation to MiniGrid's own environments on
2026-10-05 (CHARTER, "Current world"). This card builds the harness and
measures version 14 on it; its scores are the "best version so far" that
later cards are compared against.

## 1. Question

Does version 14, unchanged, run on MiniGrid's DoorKey-8x8 (tier 1),
BlockedUnlockPickup (tier 2) and ObstructedMaze-Full (tier 3), and what
does it score? P18 (component successes in a published benchmark), P17
(costs on larger maps), C1 (the agent sees only MiniGrid's view), C3
(the harness's priors declared).

## 2. What changes

Nothing in the agent. The world:

| | Before | This card |
|---|---|---|
| Environment | our key, switch, either and both worlds; card 048's chained rooms | MiniGrid's own `DoorKey-8x8-v0`, `BlockedUnlockPickup-v0`, `ObstructedMaze-Full-v1`, stepped by MiniGrid |
| What the agent sees | MiniGrid's 7 × 7 view with occlusion, drawn from our simulator | the same view of MiniGrid's grid (checked against MiniGrid's own observation) |
| Objects | keys and doors in 4 colours, two balls, one box | keys, doors, balls and boxes in MiniGrid's 6 colours, appended to the tile catalogue (earlier codes keep their numbers) |
| Goal | the goal square, or a goal written in | tier 1 the goal square (the episode's end); tiers 2 and 3 holding the mission's object, written in as its tile (card 056's "has") |
| Memory | card 034's play in our worlds | random play in each tier: uniform actions, about 3.2 million steps, card 034's sampling, card 062's believed views |
| Token lattice | 12 tiles (placements within 6 of the start) | 12, 14 and 20 tiles for tiers 1–3, so that every place of the map is a placement from any start (a declared prior) |

CHARTER names `ObstructedMaze-Full-v0`; the harness uses `-v1`, which
MiniGrid publishes to fix v0's layouts where a ball covers the key. Card
052's generator appends six hues to MiniGrid's shared colour list; the
harness restores it, so the environments generate as published.

## 3. Dependencies

Version 14 (cards 057, 060–062 on card 051's planner); card 054's encoder
(seed 399), trained on card 052's generator, which draws MiniGrid's keys,
doors, balls and boxes in these six colours; card 056's goals over the
held tile; MiniGrid 3.1.0.

## 4. Data check

Random play per tier (`runs/066/memory_tier*.json`):

| Tier | Episodes | Steps | Rows kept | Tries | Doors opened | Keys out of boxes | Random successes |
|---|---|---|---|---|---|---|---|
| 1 | 5,000 | 3,164,788 | 451,658 | 254,165 | 1,565 | – | 167 (3.3%) |
| 2 | 5,555 | 3,196,379 | 490,153 | 291,128 | 416 | – | 25 (0.45%) |
| 3 | 888 | 3,130,903 | 514,034 | 318,328 | 10,571 | 2,889 | 52 (5.9%) |

## 5. Feasibility gate

- **The adapter:** in 300 random states of each tier, the agent's 7 × 7
  view equals MiniGrid's own observation, tile for tile and in what is
  visible (0 differences in 900 states; the agent's own place aside,
  where MiniGrid shows the carried thing). Only MiniGrid's six colours
  occur.
- **Upper bound:** every layout is solvable (MiniGrid; v1 for tier 3).
- **Trivial baseline:** random actions, as in section 4.
- **Smoke test:** tier 1, 20 episodes: 20 successes, 21.9 steps on
  average, 3 ms per step.

## 6. Success criteria and prediction

1. **The adapter** is exact (section 5).
2. **Tier 1:** version 14 reaches ≥ 99% over 200 episodes (CHARTER).
3. **Tiers 2 and 3:** reported on fixed seeds with success, steps, the
   share of random and exploring steps and time per step; no threshold.

**Prediction.** Tier 1 passes (DoorKey is our key world with an
unseen goal square). Tier 2 low: the ball in front of the door needs
clearing (card 058's open problem). Tier 3 near 0%, and slow: version
14's pair tables grow with the square of the lattice (3,364 placements,
about 11 million pairs), and System 1 is read far beyond the 12 tiles it
was trained on.

**Decision rules.** Keep (the harness and these scores become the
reference) if criteria 1 and 2 hold; revise the adapter if criterion 1
fails; if tier 1 fails, keep the harness and record version 14 as not
passing tier 1.

**Budget.** Tier 1, 200 episodes: about a minute. Tiers 2 and 3: set from
the smoke runs (section 7).
