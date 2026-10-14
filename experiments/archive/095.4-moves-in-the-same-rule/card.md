---
id: "095.4"
title: moves predicted by the same rule as interactions, on a world-fixed state
rung: 1
serves: [P6, P17, P3]
status: abandoned
verdict:
arch_version: 23
date: 2026-10-10
---

# 095.4: moves in the same rule (a question; drafted after 095.3)

A step of [card 095](../095-state-change-prediction/card.md)'s series.

## 1. Question

Can left, right and forward be predicted by 095.3's rule (which tokens
change, and what they become, by arm B's vote with arm A as its prior),
so that the fitted view transformations and forward's "draw" and
"undraw" kinds are retired? In the agent's own frame forward changes
18–62 places because the window moves; in a frame fixed to the world
(card 062's believed map, card 094's layout) it changes one token, the
agent's own place and heading. P6, P17, P3.

## Open before drafting

- What 095.3 leaves: whether the front-and-hand form (095.3's gate) has
  a counterpart for moves (the token ahead, and the agent's token).
- How much of a tier 3 step goes to walking; card 097.1 (walking as a
  skill) may take the same ground, and the two must not be built twice.

## Closed (2026-10-10, overnight): nothing for MiniGrid to test; moved to Crafter

The user asked (2026-10-10) for the cards 091–097 to be finished before
Crafter, unless saturated tiers leave nothing to evaluate. Here the
MiniGrid part is already done in substance, so there is nothing to
evaluate:
- **Forward's outcome** (moved, blocked, the episode ends) is already
  in arm C's two-part form: this token's own tries first, with card
  087's property network as the prior (version 18 onward). Admission
  would find only the token ahead (095.1: no other place matters for
  any action). Card 097.2's surprise check, which compares the pose,
  front and hand after each held step with the prediction, never fired
  on tiers 2 and 3 (tier 3: at least 2,596 checked steps).
- **The view's transformations** (turn, step) and draw and undraw (how
  the agent's own place looks) are perception's geometry, fitted from
  frames by card 061 with zero residual, not dynamics. A world-fixed
  state still needs them to place what the egocentric view shows.
- Card 091's router, kept "for forward" in 095.3, is not queried in
  version 23 (0 router queries in every tier): card 087's prior answers
  forward first.

In Crafter the view is centred on the player and moves are translations
(no turning); the same two pieces apply (fitted translations; forward's
outcome from the target place's token, now including water and lava).
That check is a step of the Crafter series.
