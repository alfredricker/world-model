---
id: "098.2"
title: Crafter tokens across light — the same world place by day and by night, tied through the agent's own moves
rung: 1
serves: [P7, P1, P4, P18]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 098.2: tokens across light

The revision of [card 098](../098-crafter-tokens/card.md), written and
approved overnight (2026-10-10, overnight authority; new card 7 of the
night). Offline; the agent is not run in Crafter.

## 1. Question

Card 098's encoder, trained on frame pairs in which the agent did not
move, separates tiles at every light level but gives the same thing
different codes by day and by night (grass: 82% of its tiles in its day
code, the rest in about 35 codes by light level): consecutive frames
differ in light by a fraction of a percent, so nothing ties the night
appearance to the day one. If the agent ties a tile seen at dusk or at
night to the same world place seen by day, through its own moves, do
the codes become the same by day and by night? P7 (continuity), P1, P4,
P18.

## 2. What changes

One component, the encoder's training pairs. Kept from card 098: still
pairs judged only between places drawn identically when redrawn, and a
place that changed in an otherwise identical pair as a pair loss
(variant H); codes founded only by tiles drawn identically when
redrawn, radius from training's own same and different pairs.

New: **pairs across light through the agent's own moves.**
- **Moves learned from frames** (as card 028 in MiniGrid): for each
  action, the view shift (one tile or none) that best matches the next
  frame most often; random play gives left, right, up and down a
  one-tile shift (about 60% of the time; otherwise blocked), every other
  action none. Per step, the agent checks whether its action's shift
  happened (the shifted frame matches better than the unshifted).
- **Path integration**: summing the shifts within an episode gives each
  frame's offset, so a place in one frame corresponds to a place in any
  other frame of the episode whose view overlaps it.
- **Pairs**: a frame drawn without noise (its next still frame drew most
  places identically) as anchor, another frame of the episode whose
  light differs by at least 0.2 and whose view overlaps it in at least
  20 places as candidates. The anchor frame labels the other: a place
  corresponding to a tile identical to the anchor is the same thing, a
  place corresponding to a different tile is a different thing, places
  outside the overlap are unknown. Supervised contrastive loss, added
  to card 098's.

## 3. Dependencies

Card 098 (data, encoder, codes); card 028 (moves learned as view
shifts); time-contrastive networks (Sermanet et al. 2018: the same
scene from other viewpoints and times as positives); slow feature
analysis (Wiskott and Sejnowski 2002).

## 4. Data check

Card 098's 100,000 random-play steps. Report: moves that shifted the
view; anchor frames; pairs with light differing by 0.2 or more and at
least 20 overlapping places, by the darker frame's light.

## 5. Feasibility gate

- **Upper bound:** card 098's classifier on the evaluator's labels:
  98.3% at night.
- **Trivial baselines:** raw-pixel k-means at the same number of codes;
  card 098's variant H with the same code rule.

## 6. Success criteria and prediction

1. **Same code by day and by night (the question):** at least 90% of
   night world tiles (daylight below 0.3) carry a code that holds at
   least 1% of their class's day tiles (daylight above 0.7); variant H
   reported the same way.
2. **Card 098's criteria:** purity at least 95% overall and 90% at
   night, above k-means at the same number of codes; inventory at least
   95%; codes per class (by majority) at most 4.

**Prediction.** Criterion 1 rises well above variant H's, since night
tiles are trained toward their day appearance; night purity near the
classifier's at the centre and lower at the edges, where noise covers
most of the tile. Codes per class fall but may stay above 4 for grass,
whose night appearances vary most.

**Decision rules.** Keep (tokens for the Crafter series) if 1 and 2
hold; revise if 1 holds and 2 does not; stop if 1 fails (continuity
through moves does not tie the appearances).

**Budget.** About 2 minutes of training, 2 of evaluation.

## 7. Result

`tools/card098.2/across.py`, `runs/098.2/`.

**Data check.** The learned shifts are left (1, 0), right (−1, 0), up
(0, 1) and down (0, −1) for actions 1–4 and none for the rest; 14,150 of
23,599 move steps shifted the view (the rest were blocked). 64,515
anchor frames; 2,787 of 3,000 tried found a partner, whose darker frame
was below light 0.3 in 583, 0.3–0.5 in 629 and above 0.5 in 1,575.

| | Variant H (098) | First run | Both frames as candidates | **and the agent's own tile fixed** | k-means, raw pixels (57 codes) |
|---|---|---|---|---|---|
| Codes | 78 | 93 | 72 | **57** | 57 |
| 1. Night tiles in a code of their class's day tiles | 5.8% | 12.8% | 89.0% | **94.0%** | 7.0% |
| Purity overall / night / day | 97.5 / 81.7 / 100 | 98.6 / 90.2 / 99.8 | 99.1 / 94.0 / 99.9 | **99.2 / 94.7 / 99.9** | 95.8 / 82.0 / — |
| Inventory | 90.3% | 95.2% | 98.6% | **95.4%** | 90.2% |
| Codes per class, worst | 36 (grass) | 35 (grass) | 7 (grass) | **6 (cow); every other class 3 or fewer** | — |

(`results_v1.json`, `results_joint_selfbug.json`, `results_joint.json`.)

**First run.** The pairs across light improved criterion 1 only from
5.8% to 12.8%. The loss could not express what was asked: every
candidate in a softmax came from the same later frame, so anything
that frame's light adds to all of its tiles alike shifts every
similarity alike and cancels. The encoder kept tiles apart by relative
distance and never moved night grass onto day grass.

**Revision (the card's one): both frames' tiles as candidates in one
softmax.** A day grass anchor must now be nearer night grass than day
sand, so a light offset costs loss: 89.0%. One correction to the
implementation followed, not to the design: the agent's own tile was
treated as a world place, so every pair taught that the player at
night is the grass the player had stood beside by day (the centre at
night: 0%). The centre now corresponds to the centre (the agent
itself), and a world place the agent covered is unknown: 94.0%.

By ring at night (before that correction): 96% at rings 1–2, 89.5% at
ring 3, 82.9% at ring 4, where noise covers most of the tile (the
classifier's ceiling there: 95.9%).

1. **Same code by day and by night:** met, 94.0% (variant H 5.8%,
   k-means 7.0%).
2. **Card 098's criteria:** purity met (99.2% overall, 94.7% at night;
   k-means at 57 codes 95.8% and 82.0%); inventory met (95.4%); codes
   per class not met for cows (6; every other class 1–3). Cows keep
   codes by light level (0.97, 0.93, 0.85, 0.72, 0.53, 0.38), and one
   dusk code mixes cows, zombies and arrows: creatures move, so a world
   place seen at two times does not show the same creature, and the
   pairs mislabel them.

## 8. Decision

**Revise** (the card's rule: 1 holds, 2 does not). Continuity through
the agent's own moves ties appearances across light for everything
that stays put, and with both frames in one softmax the codes are the
same by day and by night. What is left is moving things: the next
revision ties a creature to itself across time by following it (its
tile moves by one place between frames) rather than by the world place.
The encoder (`runs/098.2/encoder_joint.pt`) and codes
(`codes_joint.npz`) are the best Crafter tokens so far and may be used
by the next Crafter cards, stated as such.
