---
id: "098"
title: Crafter — tokens learned from its pixels, the same thing the same code by day and night
rung: 1
serves: [P7, P1, P4, P18]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 098: Crafter tokens from pixels

Written overnight (2026-10-10) under the user's overnight authority, new
card 2 of the night; approved under it. The user's objective: "starting
Crafter evaluations this weekend". CHARTER enters Crafter once the rungs
MiniGrid can test have passed; this card and 098.1 are offline
evaluations in Crafter (the agent is not changed and not run there), so
they are read as the start of evaluation, not a change of direction.

## 1. Question

Every later Crafter card needs tokens: what each tile of the view is,
as a code (identity) and a vector (similarity), from pixels only (GOAL
C1). MiniGrid's tiles are drawn exactly; Crafter's are not: at night the
view is tinted, desaturated and overlaid with random noise growing
toward its edges, so the same grass tile never repeats exactly. Can an
encoder learned from the agent's own experience give the same code to
the same thing by day and by night, and different codes to different
things, judged against Crafter's semantic map (the evaluator's only)?
Rung 1 (its world grows); P7, P1, P4, P18.

## 2. What changes

Nothing in the agent. New: Crafter's view as tokens.

- **Units** (declared prior, as MiniGrid's tile grid): the 64 × 64 image
  is 9 × 9 tiles of 7 × 7 pixels; rows 0–6 are the world around the
  player (the player at column 4, row 3), rows 7–8 the inventory (one
  fixed slot per item, its icon and count).
- **Encoder, learned from continuity** (P7): a small convolutional
  network maps a tile to 32 numbers. Training pairs: the same place in
  two consecutive frames when the agent did not move (no move action, or
  a move that changed nothing), which show the same thing under slightly
  different light and noise, unless a creature moved there; contrastive
  loss (InfoNCE) against the other tiles of the batch. Slowly changing
  light is learned through chains of small changes (slow feature
  analysis, Wiskott and Sejnowski 2002; time-contrastive networks,
  Sermanet et al. 2018).
- **Codes:** prototypes of the embeddings, added greedily: a tile joins
  the nearest prototype within a radius set from the positive pairs'
  distances (the 99th percentile), else starts a new one (leader
  clustering).

## 3. Dependencies

Crafter (Hafner 2021, `crafter` 1.8); InfoNCE (van den Oord et al.
2018); card 054's encoder (identity up to noise) as the MiniGrid
analogue. The semantic map, player position and daylight are the
evaluator's.

## 4. Data check

Random play (uniform over 17 actions): 100,000 steps; per step the image,
the action, and for the evaluator the semantic map around the player,
the inventory and the daylight. Report: steps by daylight, distinct
semantic classes seen, positive pairs available, and the counts of the
interactions 098.1 needs (collect, place, make, eat, drink).

## 5. Feasibility gate

- **Upper bound:** exact pixels by day (daylight above 0.95): codes from
  identical tiles, purity against the semantic map (shows the classes
  are visually separable).
- **Trivial baseline:** k-means on raw tile pixels with as many codes as
  the encoder makes.

## 6. Success criteria and prediction

1. **Purity:** world tiles whose code's majority semantic class is their
   own: at least 95% overall and at least 90% at night (daylight below
   0.3), and above the k-means baseline.
2. **Few codes:** at most 4 codes per semantic class seen (appearance
   variants such as the player's four headings are allowed).
3. **Inventory:** a slot's code determines its item and count in at
   least 95% of slots.

**Prediction.** By day near 100%; at night 90–95%, losing at the noisy
edges of the view; inventory near 100% (drawn without light effects).

**Budget.** Collection 10 minutes, training 15, evaluation 5.

## First run (2026-10-10), then paused

Paused by the user (2026-10-10: finish the MiniGrid backlog before going
deep into Crafter). One training run (`tools/card098/collect.py`,
`encoder.py`; `runs/098/results.json`), kept for when the card resumes.

- **Data:** 100,000 random-play steps, 582 episodes (172 steps on
  average: random agents die fast), night (light below 0.3) 10% of
  frames; 17 semantic classes; 70,129 pairs without a move.
- **Exact pixels as codes** (the upper bound's form): 2,020 distinct
  world tiles by day, 128,880 at night (20,000 frames sampled): by
  night nothing repeats, so codes must be learned.
- **Encoder** (4,000 steps, 27 s): 85 codes; purity 96% by day, 74% at
  night, 93% overall; inventory 89%. Night purity falls with distance
  from the player: 100%, 83%, 80%, 73%, 67% at distances 0–4 (by day
  100–95%). Crafter's night noise grows toward the edges by design.
- **Likely defect:** the within-frame negatives use a pixel threshold,
  and at night two tiles of the same thing at different distances from
  the centre differ by noise beyond it, so the loss pushes them apart
  (grass owns 11 codes). To try when resumed: decide within-frame
  negatives on tiles at the same distance from the centre, or only in
  frames that are not noisy; and read uncertain tiles from belief.

## Resumed (2026-10-10, after the MiniGrid backlog): feasibility, diagnosis, revision

### Feasibility gate

- **Upper bound** (`tools/card098/ceiling.py`, `runs/098/ceiling.json`):
  a classifier of the encoder's form trained on the evaluator's labels
  (workers 0–7, judged on 8–9) is right on 99.8% of world tiles, 98.3%
  at night, 95.9% at the darkest outer ring. A tile still shows what it
  is at night: criterion 1 can be passed.
- **Trivial baseline**, k-means on raw tile pixels with the same number
  of codes (run only now): at 78 codes 96.6% overall, 85.0% at night,
  inventory 90.3%; at 409 codes 98.6%, 91.8%, 96.1%.

### Diagnosis of the first run

The suspected defect was wrong. Within one frame, the pixel thresholds
never called two tiles of the same class different at night (0%); they
called tiles of different classes the same: 97% of such pairs below
daylight 0.1, 69% at 0.1–0.3, and 17% even by day. Night darkens and
desaturates everything, so all distances shrink, and at night no pixel
threshold separates same from different (different-class median 0.0049
against same-class 99th percentile 0.0055). By day tiles of one class
are pixel-identical (99th percentile 0).

### Variants (the card's revision; each about 30 s of training)

Purity is the share of world tiles whose code's majority class is
theirs; "codes per class" counts the codes whose majority is the class
(the worst class is shown, almost always grass). Code radius: the
positive pairs' 99th percentile (as the first run), the distance that
best separates training's own same and different pairs ("boundary"),
or that boundary on noise-free pairs with only tiles drawn identically
when redrawn allowed to found a code ("exact founders").

| | Variant | Codes | Overall | Night | Day | Inventory | Codes per class (worst) |
|---|---|---|---|---|---|---|---|
| A | First run (pixel thresholds 0.02 / 0.08) | 85 | 92.9 | 73.9 | 96.2 | 89.4 | 31 (all codes) |
| B | Same thing only when pixel-identical | 149 | 96.2 | 86.8 | 98.6 | 92.2 | 31 |
| C | B; other places different unless the encoder finds them as near as true pairs (Huynh et al. 2022) | 218 | 99.0 | 93.5 | 99.8 | 85.8 | 94 |
| D | B; each frame's contrast normalised | 63 | 97.7 | 89.4 | 99.0 | 85.8 | 10 |
| E | Pixel judgments only between places drawn identically when redrawn, else C | 88 | 98.6 | 90.8 | 99.9 | 85.6 | 34 |
| E | boundary radius | 533 | 99.5 | 96.4 | 100 | 85.7 | 253 |
| G | E; a place that changed in a noise-free pair is a different thing | 486 (boundary) | 99.6 | 96.4 | 100 | 86.7 | 225 |
| H | G as a pair loss with a margin (Hadsell et al. 2006) | 409 (boundary) | 99.4 | 94.3 | 100 | 94.3 | 240 |
| H | exact founders | 78 | 97.5 | 81.7 | 100 | 90.3 | 36 |
| I | H with D's normalisation, exact founders | 253 | 98.1 | 85.5 | 100 | **99.1** | 105 |

(`runs/098/results_*.json`; all variants in `tools/card098/encoder.py`
behind `WM98_*` switches.)

What the variants established:
- **The encoder separates what it must.** E's embedding read by nearest
  neighbour (evaluator's probe) is right on 96.6% of night world tiles
  and 99.99% of inventory slots: the failures are in the codes.
- **Counts are never contrasted within a frame** (one slot per item), and
  a count changing between frames made "health 9" and "health 8" a
  positive pair. Treating a place that changed in an otherwise
  identical frame pair as a different thing, as a pair loss (H, I),
  fixes inventory (99.1% with I's codes).
- **The codes are not the same by day and by night.** Grass keeps 82%
  of its tiles in one day code and spreads the rest over about 35 codes
  by light level (dusk 0.44, 0.25, night 0.06–0.13); night purity is
  reached with night-only codes, not by night grass joining day grass.
  Still pairs link only consecutive frames, whose light differs by a
  fraction of a percent, so nothing ties the night appearance to the
  day one; tiles at the view's centre are noise-free at night and found
  night codes of their own.
- At equal numbers of codes, raw-pixel k-means is close: the encoder is
  better at night only with 400–500 codes (94.3% against 91.8%).

## Decision

**Revise.** Criteria 1–3 are not met together by any variant (the best
world purity, E or G with the boundary radius, needs 500 codes and gets
inventory wrong; the best inventory, I, gets 85.5% at night), and the
card's question, the same code by day and by night, is answered no.
Kept for the revision: judgments only where a place drew identically
when redrawn, and changes as a pair loss. The missing piece is
continuity across light: positives between the same world place seen
at different times, found through the agent's own moves (a successful
move shifts the view by one tile, which the agent can check from the
frames, as card 028 learned moves in MiniGrid), so that a tile seen at
dusk and at night is tied to the same tile seen by day. That is card
[098.2](../098.2-tokens-across-light/card.md) (098.1 stays the interactions card).
