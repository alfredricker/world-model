---
id: "103.5"
title: card 098.6's files level on pixel sprites — earned existence, records by look, recognition, negative information
rung: 1
serves: [P7, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 103.5: card 098.6's files level on pixel sprites

Drafted and run on 2026-10-12, in series
[103](../103-tracking-with-general-priors/card.md), at the user's request
("ok, run 103.5"). It is card [103.2](../103.2-object-files-as-sprites/card.md)'s
revision.

## 1. Question

Card 103.2's sprite files separate a creature from the ground beneath it,
but they lacked card 098.6's bookkeeping. Suppose they get it:
- existence earned over views;
- records that learn which looks need memory of their own;
- recognition of such looks when first seen;
- a map of where an unseen file could be.

Does a creature then keep one identity better than 098.6's files did on
tile tokens (IDF1 0.717)? This must hold on the view placed by card 103.4,
with no tile, kind, grid or ground declared. Rung 1. Serves P7 (units
tracked across frames, from continuity) and P2.

## 2. What changes

One component: card 103.2's object files level. The sprite (template, mask,
world position), the search and the residual stay. Bookkeeping from 098.6
replaces "reported once it has moved".

```
103.2:      a file is reported once it has moved; claims by order of hits; kept as it was while unseen
this card:  existence r (IPDA). A placement is a hit with L = own / other:
              own   = p(d), the confirmed files' moves (before any: the agent's own placements)
              other = clutter, how often the template matches elsewhere in its gate in this frame
                      (where the canvas knows the world), or 1 - share (where it does not)
            a placement the world explains as well is not a detection; a file not found where it could be:
              r <- r (1 - q) / (1 - r q), q = P_D x P(in view)
            confirmed at 0.95, deleted below 0.05, tentative for 3 steps; only confirmed files are reported
            one placement for all files: by ln r + ln L, only where the file beats "not a file"
            records: one per file (look, steps seen, moves by itself), kept at its place after a tentative file
              ends; grouped by look; the share needing memory of their own by 098.6's EM
            births: a piece found at one displacement whose source has changed (the pair test): r0 = share
              updated by L; one that appeared: r0 = share; looks with share >= 0.05 searched in first-seen pixels
            an unseen file's map of places: moved by p(d) each step, lowered where looked for and not found
            cohesion: residual pixels near a moved file, shown by the canvas at its displacement, join its mask
```

**Declared (rule 8):**
- One displacement distribution for all files: 098.6's bank of predictors
  and its recall per look are not ported.
- No density of unseen objects.
- Recognition searches the 32 looks with the largest share.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A file moves as one 2-D shift per step | yes | yes | rigid parts in the image plane only | declared (cohesion), as 103.2 |
| Files are drawn over the world (layer order) | yes | yes | no | declared, as 103.2 |
| Before any file's move is seen, files move as the agent's view does | yes | yes | roughly | general (098.6's prior), corrected by experience |
| A look that has needed memory of its own will again | yes | yes | yes | general (098.6) |
| Tiles, kinds, ground, object size | not used | — | — | removed |

## 3. Dependencies

- Card 103.4's placing (kept): the learned arm's origins, workers 0, 1, 8
  and 9 (placing on 99.1–99.3% of each worker's steps; worker 9 is new).
- Card 103.1's camera pixels; card 103.2's sprite search and residual.
- Card 098.6's files level (gate passed on the true state, 0.934). Its
  methods are in LITERATURE: IPDA and PMBM existence, M-of-N confirmation,
  negative information, the beta-binomial share (Kemp, Perfors and
  Tenenbaum 2007). Clutter density per gate is the standard PDA quantity.
- Card 098.5's play and IDF1.

## 4. Data check

Training: workers 0–1 (60,000 steps). Test: workers 8–9 (55,655 creature
frames: cows, zombies, skeletons, arrows; 15% at night). On worker 1, 9% of
creature frames come before the creature's first move in view.

## 5. Feasibility gate

- **Upper bound:** 098.6's files on the true state, 0.934.
- **Reference:** 098.6 on tile tokens, 0.717.
- **Baselines:** 103.2 at 0.531 on the evaluator's origins and 0.411 on
  103.4's; identity by place, 0.009.
- **Development:** built on training play only, with the long-term memory
  learned on worker 0 and scored on worker 1, using the evaluator's origins.
  Each step below was found by a diagnostic of the losses:

| Build | IDF1 | ID recall / precision | Not on a creature |
|---|---|---|---|
| 098.6's existence, records, recognition | 0.558 | 0.573 / 0.543 | 36% |
| Pair test also checks that the source changed (night noise pieces) | 0.591 | 0.570 / 0.613 | 28% |
| Records kept at their place; staying is no evidence for a tentative file | 0.602 | 0.574 / 0.633 | 26% |
| Evidence against clutter in the gate, not the world's drift rate | 0.606 | 0.577 / 0.638 | 24% |
| A place the world explains as well is not a detection | 0.635 | 0.518 / 0.819 | 1.0% |
| Map of an unseen file's places; the world's move rate for scenery | **0.660** | 0.547 / 0.834 | 0.5% |

Two diagnostics drove the last three rows:
- **103.2's lost files** still matched their creature exactly in most loss
  events. Duplicates had claimed the pixels first.
- **The development run's false files** were tracks that had lost a
  creature at night. They kept matching dark ground that the canvas also
  explained.

## 6. Success criteria and prediction

1. **Identity on the learned view.** IDF1 over cows, zombies, skeletons and
   arrows, workers 8–9, on 103.4's origins, at least 0.717 (098.6 on tile
   tokens).
2. **What placing costs.** IDF1 on 103.4's origins within 0.05 of the
   evaluator's origins (103.2: 0.11).
3. **Files on scenery.** At most 1% of reported detections on no creature
   (098.6: 0.39% on static tokens; 103.2: 25%).

**Prediction:** on the evaluator's origins IDF1 0.64–0.68, as in
development. On 103.4's origins it is within 0.03 of that. So criterion 1
fails, while criteria 2 and 3 hold. Recall is the limit:
- still creatures get a file only once their look's share reaches 0.95,
  which is 098.6's rule;
- night.

**Decision rules:**
- **Keep** if all three hold.
- **Revise** if criteria 2 and 3 hold and IDF1 on the learned view is at
  least 0.63 (0.1 above 103.2), but below 0.717.
- **Stop** this form if IDF1 on the learned view is below 0.58.

**Budget:** each run is one worker, 5 minutes, capped. That is 4 training
runs and 4 test runs, about 25 minutes with two at once.

## 7. Result

Run 2026-10-12, workers 8–9, with the long-term memory learned on workers
0–1 of the same arm ([results.json](results.json)).

| Measure | Evaluator's origins | 103.4's origins | Bar |
|---|---|---|---|
| IDF1, movers pooled | 0.641 | **0.623** | 0.717 (098.6, tile tokens) |
| ID recall / ID precision | 0.528 / 0.814 | 0.520 / 0.779 | |
| Detections on no creature | 0.44% | 4.3% | at most 1% |
| ID recall: cow, zombie, skeleton, arrow (worker 8) | 0.55, 0.50, 0.51, 0.40 | 0.55, 0.48, 0.51, 0.40 | |
| Night creature frames with a file | 46% | 45% | |
| Still creatures (before a move) with a file | 7.3% | 7.4% | |

1. **Fail.** 0.623 on the learned view, against 0.717. The prediction
   (0.64–0.68 on the evaluator's view) was met at the bottom end.
2. **Pass.** The learned view costs 0.017 against the evaluator's.
3. **Fail on the learned view** (4.3%), pass on the evaluator's (0.44%).
   False files follow placing errors.

Where a creature's frames went (worker 8, learned view): 64.5% to its own
file, 12.3% to another file (identity broken and restarted), 10.0% to a
file still tentative, and 11.8% to nothing (7.8 points of it at night,
after moving). Fragmentation, slow confirmation and night are each about a
tenth.

## 8. Decision

**Stop** (2026-10-13). The rules did not foresee this case: IDF1 0.623 lies
between the stop line (0.58) and the revise line (0.63), and criterion 3
fails on the learned view. The user then set hand-built correspondence
aside. GOAL C1 now lets a declared, pretrained perception front end
supply objects' outlines and their correspondence ("we shouldn't need to
solve an agent completely end to end"). Card
[104](../104-sam-correspondence/card.md) gives that job to SAM 2.1. These
numbers, rerun on 104's new held-out play, are its baseline. Kept for
104: earned existence (tentative, then confirmed) and scoring by IDF1 with
detections on no creature.
