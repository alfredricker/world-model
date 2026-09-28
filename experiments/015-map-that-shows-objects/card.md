---
id: "015"
title: a map that shows objects
rung: 0
serves: [P7, P4, P6]
status: done   # draft | approved | gated | running | done | abandoned
verdict: fail   # pass | fail | uninterpretable, when done
arch_version: 0
date: 2026-09-27
---

# 015: a map that shows objects

## 1. Question

Which self-supervised signal makes the encoder's spatial map show, tile by
tile, the objects that walking and acting depend on: reconstructing each
tile, or predicting which tiles an action changes? Before rung 1; serves P7
(learned units: what is at each place), P4 (keep what changes outcomes) and
P6 (consequences of actions). Card 014 needs it: its walking upper bound
stalls at 90–94% of moves closer, and a one-tile linear read-out of the
map finds an open door in only 12–14% of tiles (closed door 99.7%).

## 2. What changes

One component, the encoder's training signal, in two arms compared under
the same criteria (the user asked for both, 2026-09-27). Both start from
card 012 run 5's encoder and add one loss on its map (64 × 9 × 8, one cell
per tile):

- **Arm R, reconstruction:** each map cell predicts its own tile's pixels
  (8 × 8 × 3) through a 1 × 1 layer; mean squared error.
- **Arm C, change prediction:** from the map and the action taken, each
  cell predicts whether its tile's pixels differ in the next frame (a 3 × 3
  layer over the map with the action added at every cell; cross-entropy,
  changed tiles weighted up to balance them). A toggle changes a closed
  door, an open door, a switch or a vase; walking forward changes the
  tile in front only if it can be entered. So telling an open door from
  floor or a closed door is needed to predict the consequences.

The change label is computed from pixels (the two rendered frames), not
from simulator state. No object labels are used in training.

## 3. Dependencies

Card 012 run 5 (encoder, data policy); card 014's bench (object read-out,
full-map walking upper bound, `tools/card014/bench_vin.py`). Per-tile
reconstruction and next-step change prediction are standard
self-supervised signals; neither is a new method.

## 4. Data check

Card 012 run 5's key-world data (30k episodes, a third with play starts).
Reported: the share of frames with the door open, and the number of
transitions in which the door tile, switch or vase changes. Episodes with
the wall in column 5 are removed from training (arms and read-out) and
used only for testing.

## 5. Feasibility gate

- **Upper bound:** the same linear read-out trained on the encoder's input
  (the rendered tile itself, i.e. perfect per-tile information): must reach
  ≥ 99.9% for every class, otherwise the read-out, not the map, is at fault.
- **Trivial baseline:** run 5's encoder without any added loss: open door
  12%, switch 4–12%, vase 0% (card 014's read-out, unweighted classes).

Gate result: on the tiles' own pixels the read-out is 100% for every
class, on both wall columns. With classes weighted equally (the read-out
used for the criteria), run 5's encoder gives: floor 78%, wall 72%, goal
100%, closed door 100%, open door 95% (61% on the unseen wall column),
key 100%, switch 29–54%, vase 31%. The 12% above came partly from the
unweighted read-out ignoring the rare class.

## 6. Success criteria and prediction

For each arm (20k updates, same data, same seed):

1. **Objects:** one-tile linear read-out of the frozen map ≥ 99% for every
   class (floor, wall, goal, closed door, open door, key, switch off/on,
   vase), on held-out layouts with the wall in a column seen in training.
2. **Unseen geometry:** the same ≥ 99% on layouts with the wall in
   column 5 (never trained on).
3. **Use:** card 014's walking upper bound (every cell supervised, 8k
   updates) on the arm's encoder: moves closer ≥ 99% for both ways (run 5's
   encoder: 90% / 94%).

The better arm is the one meeting more criteria, then with the higher
walking score. Prediction: R passes criteria 1–2 easily (every tile's
pixels must be kept) but may not help walking more than it helps
appearance; C focuses the map on what actions change and may leave static
details (goal colour) weaker. Budget: about 10 minutes per arm, both arms
run in parallel.

## 7. Result

`tools/card015/bench_objects.py`, `runs/015_{R,C,input}.json`; key world,
20k updates per arm, same data and seed. Read-out on held-out layouts
(wall column seen / unseen):

| Class | Run 5 encoder | Arm R (reconstruction) | Arm C (change prediction) |
|---|---|---|---|
| floor | 78% / 75% | 97.8% / 97.8% | 99.0% / 98.8% |
| wall | 72% / 70% | 100% / 100% | 99.8% / 99.8% |
| goal | 100% / 89% | 100% / 100% | 100% / 100% |
| closed door | 100% / 100% | 100% / 100% | 100% / 99.9% |
| open door | 95% / 61% | 96.3% / 98.7% | 97.2% / 100% |
| key | 100% / 99% | 100% / 100% | 100% / 100% |
| switch off / on | 29% / 54% | 100% / 100% | 44% / 80% |
| vase | 31% / 52% | 100% / 100% | 86% / 91% |

| Walking upper bound (card 014, every cell supervised) | Goal square | Door with key |
|---|---|---|
| Run 5 encoder | 90.3% | 94.5% |
| Arm R | 84.1% | 92.0% |
| Arm C | 89.8% | 91.7% |

| Criterion | Arm R | Arm C |
|---|---|---|
| 1. Objects ≥ 99%, seen column | fail (floor 97.8%, open door 96.3%) | fail (switch, vase, open door) |
| 2. Objects ≥ 99%, unseen column | fail (floor 97.8%, open door 98.7%) | fail (switch, vase) |
| 3. Walking ≥ 99% | fail | fail |

Both signals make the map show objects far better, reconstruction
everything and change prediction what actions change (doors, not the
unused switch). Neither improves walking: the upper bound stays at
84–94% whatever the map shows. So what the map shows is not what limits
card 014's walking.

**Follow-up (the user asked, same day): training length.** Arm C's map,
walking upper bound with 30k updates instead of 8k (loss 0.0045 at 10k →
0.0003 at 30k): moves closer **99.97%** for both ways on held-out frames
(every distance band 99.7–100%, except 95.2% for the few walks of 13+
steps to the door), and 99.8% on the frames it was trained on. It was
training length, and the module generalises rather than memorises.
Control, run 5's encoder with no added loss, same 30k updates
(`runs/015_run5_trainfit.json`): 99.07% (goal square) and 99.83% (door)
on held-out frames, 99.65% / 99.97% on training frames. Arm C's map is
slightly better on the goal square (99.97%) but not needed to pass 99%.

## 8. Decision
