---
id: "116.1"
title: objects as slots at 448 pixels — the same slot model on a 32 × 32 patch grid that can show the buttons and cube
rung: 2
serves: [P7]
status: running
verdict:
arch_version: 27
date: 2026-11-01
---

# 116.1: objects as slots at 448 pixels

Card [116](../116-objects-as-slots/card.md) stopped at its gate. On the
14 × 14 grid the buttons cannot be bound (ceiling mBO 0.004) and the cube
barely (0.15). Its section 6 named this card in advance: re-encode the
frames finer. The approval of 116 (the user, 2026-11-01: "approved,
proceed") covers it. The gate measured 448 × 448 input (32 × 32 patches,
each exactly 2 × 2 frame pixels) as the best grid: ceiling 0.57 on both
buttons and 0.53 on the cube.

## 1. Question

Card 116's question, on the 32 × 32 grid. Rung 2, P7.

## 2. What changes

Only the encoder's input size: 448 × 448, giving 1,024 patch vectors per
frame instead of 196. The slot model, its losses and its settings are
card 116's, with one exception. The MLP decoder's width is set by the
smoke run, so that at least 20,000 steps fit in 30 minutes; the decoder
now runs over 5 times as many patches.

- **Training frames:** every fifth frame of training episodes 0–449
  (90,090 frames, about 71 GB in fp16), not 0–899, to keep encoding under
  10 minutes and the disk use halved. 90,000 frames are far more than
  the steps can use.
- **Held-out frames:** encoded only where scored: the 2,000 truth frames,
  each with its 4 model steps of history, and the 100 clips with a 4-step
  lead-in.
- Card 114's predictor keeps its own 14 × 14 features. The slots are a
  separate stream; card 117 pools the predictor's patches inside slot
  masks downsampled to 14 × 14.

Portability: card 116's table holds. The 448-pixel input is a declared
setting of the perception adapter.

## 3. Dependencies

Card 116's gate (truth, metrics, k-means code). DINOv2 ViT-S/14 accepts
448 × 448 input by interpolating its position embeddings, as SlotContrast
does at 336 and 518.

## 4. Data check

As card 116: the cube is visible in 94% of frames; each button is about
11 pixels and the cube 14.

## 5. Feasibility gate

1. **Ceiling at 32 × 32** on the 2,000 truth frames. The bars below are
   set from it.
2. **k-means at 32 × 32:** 8 clusters on the 448-pixel features. If its
   mean mBO is within 0.10 of the ceiling, criterion 2 cannot be passed,
   and the card says so before training.
3. **SAM 2.1-small automatic masks** on 300 of the truth frames:
   reported, not judged.
4. **Smoke run:** 2 minutes, to set the decoder width and check that the
   loss falls.

Result of the gate (2026-11-01):

| 2,000 truth frames | FG-ARI | Mean mBO | Arm | Cube | Button 0 | Button 1 | Drawer | Window |
|---|---|---|---|---|---|---|---|---|
| Ceiling, 32 × 32 | 0.81 | 0.70 | 0.88 | 0.54 | 0.58 | 0.57 | 0.77 | 0.86 |
| Bar for criterion 1 (0.7 × ceiling) | 0.6 | | 0.62 | 0.38 | 0.40 | 0.40 | 0.54 | 0.60 |
| k-means, 32 × 32 | 0.62 | 0.21 | 0.55 | 0.03 | 0.02 | 0.02 | 0.28 | 0.35 |
| SAM 2.1-small, automatic (300 frames, reported) | 0.26 | 0.40 | 0.27 | 0.64 | 0.40 | 0.49 | 0.31 | 0.30 |

- **k-means** is 0.49 below the ceiling on mean mBO, so criterion 2 (0.31
  or more) can be passed.
- **SAM 2** proposes 34 masks per frame (0.74 s each). It finds the small
  objects but cuts the large ones into pieces: what an untrained, supplied
  segmenter gives here.
- **Smoke runs:** the decoder at width 512 trains at 5.5 steps per second
  (10,000 steps in 30 minutes); at width 256, 11.7 (21,000 steps). Width
  256 is used, with batch 32 (the paper used 64), to fit 30 minutes.
- **Encoding at 448:** 90,090 training frames in 193 s, and 11,823
  held-out frames in 26 s; 75 GB.

## 6. Success criteria and prediction

On held-out validation frames and clips:
1. **Each object has a slot:** FG-ARI ≥ 0.6, and for every object, mBO at
   least 0.7 × its 32 × 32 ceiling.
2. **Better than clustering:** mean mBO at least 0.10 above k-means at
   the same grid. FG-ARI is reported; at 14 × 14 it could not separate
   any method from k-means.
3. **Kept over time:** on the clips, each object's best slot stays the
   same on at least 90% of the model steps it is visible.

**Prediction.**
- Criteria 2 and 3 pass.
- Criterion 1 is at risk for the cube when held (it may share the
  gripper's slot) and for the buttons, which still cover only about 3
  patches.
- The arm takes 2 slots.

**Budget:**
- Encoding: training frames about 8 minutes, held-out about 1 minute.
- Gate: 10 minutes.
- Training: 30 minutes, capped; longer runs handed to the user.
- Scoring: 5 minutes.

## 7. Result

**Interim: 29 minutes, 20,521 steps, not converged.** Reconstruction
error was still falling (1.04 at step 2,250, 0.69 at 4,750, 0.47 at
17,250), and the learning rate had only halved.

| 2,000 truth frames | FG-ARI | Mean mBO | Arm | Cube | Button 0 | Button 1 | Drawer | Window |
|---|---|---|---|---|---|---|---|---|
| Slots, 20k steps | 0.87 | 0.21 | 0.46 | 0.21 | 0.01 | 0.03 | 0.23 | 0.32 |
| Bar | 0.6 | 0.31 | 0.62 | 0.38 | 0.40 | 0.40 | 0.54 | 0.60 |

| Clips: the same slot kept | Arm | Cube | Button 0 | Button 1 | Drawer | Window |
|---|---|---|---|---|---|---|
| Slots, 20k steps (bar 90%) | 100% | 77% | 93% | 75% | 99% | 99% |

- **As interim results, all three criteria fail.**
  1. FG-ARI 0.87 passes, but no object reaches its mBO bar.
  2. Mean mBO 0.21 ties k-means.
  3. The cube and button 1 are kept under 90%.
- **What the slots do** (300 frames, `tools/card116/score116.py` and a
  diagnostic):
  - The slots split the scene by object: one slot is half arm, one
    holds a third of the window, two the drawer, a small one the cube.
  - But each also claims the table around its object: 52–83% of the
    pixels of the object slots are background. Three slots are pure
    background.
  - No slot takes a button.
  - Masks are soft: the strongest slot claims 0.60 of a patch on
    average.
  - So FG-ARI (agreement on foreground only) is high, while overlap
    with each object is low. Objects are separated from each other, not
    from the background.
- Per section 6, an unconverged run gets the paper's 100,000 steps,
  handed to the user as a command (about 2.4 hours at 11.7 steps per
  second):

  ```
  cd ~/Projects/world-model && OMP_NUM_THREADS=4 timeout 9600 systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% nice -n 10 \
    bin/prun python tools/card116/slots.py --minutes 145 --hidden 256 --out slots_long.pt \
    2>&1 | grep -v -i warn > runs/116/train_long.log
  ```

  It is then scored with `tools/card116/score116.py --model slots_long.pt`.

## 8. Decision

Pending the long run. If it still fails criterion 1 with slots that
spill onto the table, the next card distils SAM 2's masks into the slot
encoder, as WorldDP does (allowed by C1). The slots then learn object
extents from a supplied segmenter, and identity over time from their own
recurrence.
