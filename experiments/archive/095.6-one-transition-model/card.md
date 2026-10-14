---
id: "095.6"
title: one transition model — arm A becomes the encoder's transition head
rung: 1
serves: [P6, P1, P4]
status: abandoned
verdict:
arch_version: 23
date: 2026-10-10
---

# 095.6: one transition model (a question; drafted after 095.3)

A step of [card 095](../095-state-change-prediction/card.md)'s series,
set with the user on 2026-10-09.

## 1. Question

If arm A (095.1's network over a state's tokens) becomes the encoder's
transition head, trained jointly with the encoder on whole states, in
place of card 054's two-slot head (front and hand) that shaped the
latent space and was then set aside, do the vectors and recall predict
at least as well, with one network for representation and prediction
(GOAL C2; C-SWM, Dreamer)? Stored memory is re-encoded and recall
refitted afterwards (card 052's staging). P6, P1, P4.

## Open before drafting

- 095.3's gate: the agent asks only about the front and the hand, so
  whether whole-state training buys anything in MiniGrid, or only in
  Crafter (effects on nearby tokens and inventory counts).
- Placement beside card 096's encoder refinement, so the encoder is
  retrained once.

## Closed (2026-10-10, overnight): moved to Crafter's encoder

The user asked (2026-10-10) for cards 091–097 to be finished before
Crafter, unless saturated tiers leave nothing to evaluate; this is such
a card. In MiniGrid arm C already predicts held-out tries at
99.84–100% (095.1) from card 070's encoder, every tier is at 100%, and
only the front and the hand ever change (095.3's gate). A jointly
trained encoder and transition head could at best tie, and the
encoder would be retrained again for Crafter's pixels (card 098),
which the user's ordering rule (no step rebuilt by the next) argues
against. The question moves unchanged to the Crafter series: card 098's
encoder trained with arm A as its transition head over Crafter's
tokens, where effects reach nearby tokens and inventory counts.
