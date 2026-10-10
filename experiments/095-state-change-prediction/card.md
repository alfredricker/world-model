---
id: "095"
title: recall predicts the change in the whole state, by one learned rule (a series, 095.x)
rung: 1
serves: [P6, P3, P4, P10, P12, P17]
status: draft
verdict:
arch_version: 20
date: 2026-10-09
---

# 095: state-change prediction (the series)

Set with the user on 2026-10-09, after
[094.2](../094.2-router-on-admitted-conditions/card.md)'s trace and the
set-aside [094.3](../094.3-results-as-relations/card.md): "shouldn't our
recall be predicting overall *state* that proceeds an action? … Not
some engineered case per token like a prediction mechanism for the hand
and a prediction mechanism for the door." The user signed off on the
series replacing recall's two-slot output (CHARTER: replacing a
component). Each step is its own card, 095.1, 095.2, …, with one
component, a gate and a decision. This card holds the plan.

## 1. Why

Recall today predicts an action's effect through three engineered
mechanisms:
- **pick up, toggle and drop:** exactly two places, the token in front
  and the hand, each changed or not, and what it becomes;
- **moves:** fitted transformations of the view, and "draw" and
  "undraw" for the place stepped onto and off;
- **effects elsewhere in view:** none, or bolted on (card 092).

"Front" and "hand" are roles we wrote in, not roles the agent learned,
and a result is stored as a literal token (tier 2's last failures:
"the red box comes into the hand" when the agent faces the yellow one).
Crafter has effects on nearby tokens and on several inventory counts,
which two slots cannot hold.

## 2. The target

- **State:** the tokens working memory holds (card 094's object files,
  each with its place), the hand, and the agent's own place and heading
  as a token.
- **A stored try:** the state before, the action, and the set of
  changes: which tokens changed, and to what.
- **Prediction for (state, action):** for every token in the state,
  whether it changes and, if it does, what it becomes. One rule for
  every token:
  - **which tokens change:** learned from the token, the action and the
    token's relation to the agent and to the other tokens. A place is a
    candidate condition like any other, kept only where it predicts
    (the user's direction after 094.1: a token's place matters only once
    a condition on it is learned). In MiniGrid this should find "the
    token in front, and the hand" on its own;
  - **what it becomes:** card 038's learned rule, per part of the
    token's vector: keep, copy from another token (which one is learned:
    the hand takes the front's token on a pick up), shift by the stored
    change, or set. Relations, so a new colour carries.
- **Two steps, not three** (the user, 2026-10-09): a vote over similar
  stored instances (095.1's arm B; exact matches are simply its nearest
  neighbours, at full weight, not a separate lookup), with a learned
  prior (arm A) where the vote has little weight. No separate "same
  tokens" level, and no flat "all tries" frequency.
- **The planner reads the predicted state.** A goal is any target change
  in it (GOAL P12), not a condition written over the front and hand.

## 3. What the stored tries show (2026-10-09)

Places whose appearance changed, in the 13 × 13 window plus the hand
(memory 066, all stored tries):

| | Pick up | Drop | Toggle | Forward |
|---|---|---|---|---|
| Tier 1: front / hand / elsewhere | 0.34 / 0.34 / 0 | 0.33 / 0.33 / 0 | 0.04 / 0 / 0 | 0.19 / 0 / 0.50 (18.5 places) |
| Tier 2 | 0.47 / 0.47 / 0 | 0.46 / 0.46 / 0 | 0.01 / 0 / 0 | 0.22 / 0 / 0.52 (18.5) |
| Tier 3 | 0.51 / 0.51 / 0 | 0.50 / 0.50 / 0 | 0.26 / 0 / 0 | 0.23 / 0 / 0.51 (62) |

- Interactions never change anything but the front and the hand: the
  test of "which tokens change" in MiniGrid is that it finds these two
  without being told, and loses nothing.
- Forward changes 18–62 places because the window turns and moves with
  the agent. In a frame fixed to the world (card 062's believed map,
  card 094's object files), forward changes one token: the agent's own
  place. So moves join the series only on a world-fixed state.
- MiniGrid cannot test effects at a distance; Crafter can.

## 4. Steps

| Step | What | Depends on | Status |
|---|---|---|---|
| [095.1](../095.1-which-tokens-change/card.md) | Interactions over the whole state, offline: which tokens change and what they become, from stored tries; two arms (a network; exemplars, Hintzman), the stronger becomes 095.2's prior; judged against version 20's recall on held-out tries | Cards 038, 049, 094 | Draft |
| [095.2](../095.2-recall-at-scale/card.md) | Recall at scale, offline, on arm C: a learned gate (A answers the tokens it is sure of), a nearest-neighbour index over B's metric, and forgetting that prunes memory in RAM and on disk (done first, the user, 2026-10-09: "speed boost before doing a full agent eval") | 095.1 | Done: index and forgetting kept, gate dropped |
| [095.3](../095.3-state-change-in-the-agent/card.md) | Arm C, as 095.2 leaves it, in the agent for pick up, toggle and drop; the planner reads the predicted tokens at the places that change. Its criteria allow a stated margin of lower success in exchange for a large gain in speed and RAM (the user, 2026-10-09; an exception to "no tier worse", margin: tier 2 −3, tier 3 −2 seeds, only with steps 2× faster on tier 3) | 095.2 | Draft |
| 095.4 | Moves in the same rule, on a world-fixed state (forward changes the agent's place); draw and undraw retired | 095.3, card 094's layout | Not drafted; overlaps 097.1 (walking as a skill) |
| 095.5 | The planner's goals and needs as any change in the predicted state (P12), replacing conditions written over front and hand | 095.3 | Not drafted |
| 095.6 | One transition model: arm A becomes the encoder's transition head, trained jointly with the encoder on whole states (tokens with places), replacing the two-slot head (front and hand, card 054's recipe) that shaped the latent space and was then set aside; kept and used as recall's prior. Stored memory re-encoded and recall refitted after (card 052's staging) | 095.3, card 094's object files | Not drafted; with or just before card 096's encoder refinement (the user, 2026-10-09) |

## 5. Risks

- **Cost:** a prediction per token per imagined action, and a vote over
  a memory that grows with experience. 095.2 (gate, index, forgetting)
  is the step that answers it.
- **Size:** card 052 grew too large by doing everything at once. Each
  step changes one component and keeps the agent's tiers.
- **Places:** 094.1 found raw offsets make every try its own key. Here a
  place is one condition in a factored rule, admitted only if it
  predicts, not part of a key that must match.

## 6. Literature

- DOORMAX (Diuk et al. 2008) and Schema Networks: effects as changes to
  objects' attributes, conditioned on relations; `1110_2211` (Pasula,
  Zettlemoyer and Kaelbling 2007): rules name objects by their relation
  to the action's target (all in LITERATURE.md);
- interaction networks and graph networks (Battaglia et al. 2016,
  2018; not yet in papi): one function shared across objects and
  relations predicts every object's next state;
- object-centric world models (C-SWM, Kipf et al. 2020; slot-based
  dynamics), to read before 095.3 for how they choose which objects
  change.
