---
id: "119"
title: variables from lasting change — a patch that switches between two steady states marks a condition change and where it is, with no object model
rung: 2
serves: [P7, P4]
status: done
verdict: fail
arch_version: 27
date: 2026-11-10
---

# 119: variables from lasting change

Roadmap step 2 (event segmentation), rung 2. Cards 116–116.3 tried to
find objects first, from appearance, and cards 117 and 118 wait on them.
This card removes that dependency (the user, 2026-11-10: objects and
conditions "ideally could be unified"). One definition serves both:
- **a variable** is a set of patches that change together, lastingly,
  between two steady states;
- **the object** is where that happens;
- **its conditions** are the steady states it switches between.

Infants group by common motion before appearance (Spelke 1990), and
grouping tiles that move together is the one object result this project
has (card 102.2). Cortex holds a steady pattern within an event and
shifts at its boundary (Baldassano et al. 2017).

## 1. Question

Do lasting changes per patch, with no slots, no segmenter and no model
of the arm, mark when and where the scene's conditions change? And do
the patches that change together group into the scene's objects? P7
(units from continuity and interaction), P4 (keep what changes
outcomes).

## 2. What changes

Card 117's boundary signal is read from patches directly, not from
card 116's slots. Nothing is trained.

```
117:  frames → slots (116) → self by influence → lasting change per slot
119:  frames → patches → lasting change per patch → patches that change together = a variable
```

- **Lasting change,** for patch p at frame τ: the squared distance
  between its mean vector over the 10 frames before τ − 5 and the 10
  after τ + 5, divided by the two windows' own variance plus a floor.
  - A patch the arm crosses and leaves scores low: before equals after.
  - A patch the arm is moving on scores low: its windows are unstable.
  - So the arm is discounted without being found first.
- **Boundaries:** card 115's rules (mean of the 4 highest patches, the
  highest within ± 5 frames, one threshold from episodes 900–999).
- **Scoring in time and place,** the top patch on the changing object:
  - *events:* card 115's list, one to one;
  - *transitions:* each run of motion is one switch from rest to rest,
    as is a lock flip. A boundary is right if it falls in a transition's
    interval on its object; a transition is found if any boundary does.
    This is the unit the definition predicts: a drawer slide of 9
    frames is one switch, not a start and a stop.
- **Main run, one component:** which features the signal reads (section
  6). The gate found that the frozen encoder's last layer does not
  support it.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A condition is steady between its changes | holds; day and night drift slowly | holds | holds | holds | general |
| A patch shows the same place over time (fixed camera) | no: needs patches placed by the pose | no: the view follows the agent; same fix | yes | fixed cameras yes, wrist cameras no | declared; card 105.10's placing by pose removes it |
| A moving object is seen as a change where it left and where it arrived | holds | holds | holds | holds | declared limit: the two are not yet bound into one object |
| Windows of 10 frames, gap of 5 | a setting | a setting | a setting | a setting | declared setting |

## 3. Dependencies

- Card 114's stored features and frames, card 115's masks, events and
  matching code. All exist; no rendering or encoding for the gate.
- Lasting change as the boundary signal: card 117's design, from the
  segmentation review (LITERATURE's current focus).

## 4. Data check

Validation episodes, frames where the signal exists (15 from either end
of an episode): 3,185 events; 1,912 transitions (lock 678, cube 572,
drawer 306, window 356). Episodes 900–999: 3,232 and 1,953.

## 5. Feasibility gate

Run 2026-11-10, `tools/card119/gate.py`, each run under 9 minutes and
capped (rule 5). Validation episodes, threshold from episodes 900–999.

| Signal | Features | Transitions F1 | Events F1 |
|---|---|---|---|
| Random frame and patch | | 0.09 | 0.06 |
| Plain change (card 115) | DINOv2, last layer | 0.45 | 0.31 |
| Plain change, true arm mask | DINOv2, last layer | 0.56 | 0.40 |
| Lasting change | DINOv2, last layer | 0.30 | 0.18 |
| Lasting change, true arm mask | DINOv2, last layer | 0.31 | 0.19 |
| Lasting change | raw pixels | **0.62** | 0.36 |
| Lasting change, true arm mask | raw pixels | 0.58 | 0.43 |

- **The harness reproduces card 115:** plain change scores 0.31 and
  0.40 on events, as there.
- **On raw pixels the definition holds for whatever moves.** With no
  mask, transitions found: drawer 99%, window 98%, cube 92%; precision
  0.52. The cube was found 6–15% of the time by every detector in card
  115.
- **On the frozen encoder's last layer it fails,** below plain change.
  The floor is not the cause: three floors gave 0.30, 0.20 and 0.18.
- **Probable cause, measured but not proven:** the encoder's patches
  are not local. At the median patch and frame, the lasting change is
  larger than the windows' variance (154 against 113). On pixels both
  are 0. A patch's vector shifts when the arm settles anywhere in the
  frame, since every layer attends over the whole image.
- **What pixels still miss:**
  - lock flips, 41% found (54–66% with the true arm mask): the arm sits
    on the button while pressing it, and a button is 2–3 patches;
  - 41% of boundaries have their top patch on the arm, where it comes to
    rest; with the true arm mask the cube falls to 13%, since a carried
    cube is under the arm.
- **Caveat:** the pixel floor was set at the 90th percentile of the
  variance (the median is 0); the 99th gives 0.59. Plain change is
  normalised per patch and lasting change is not.

**Gate result:** passed on pixels (upper bound 0.62 against random
0.09), failed on the stored features. The main run is therefore about
the features.

## 6. Success criteria and prediction

The same signal on the outputs of the same frozen encoder's blocks 2, 4,
8 and 12, so one encoder still serves every part (C2).
1. **Boundaries:** some block reaches transitions F1 ≥ 0.55 (pixels
   0.62; last block 0.30; random 0.09). Meeting it shows the definition
   works in a learned space, not only on pixels.
2. **Units:** on that block, patches are grouped by how often their
   lasting changes fall in the same boundary (episodes 0–899, no
   labels). For the drawer, the window and each button, the best group
   overlaps the object's true patches with IoU ≥ 0.5, and beats k-means
   on the same features. Meeting it shows objects come out of change,
   with no segmenter.

Lock-flip recall and the share of boundaries on the arm are reported,
not judged: both need the self (card 117's influence, per patch).

**Prediction.** Early blocks are more local, so criterion 1 passes at
block 2 or 4. Criterion 2 passes for the drawer and window and is the
risk for the buttons (2–3 patches). If no block passes criterion 1, the
frozen encoder is the wrong space for segmentation, which is a finding
about Roadmap step 1's choice.

**Budget:** encoding 200,200 frames with four taps, signals computed
per episode and not stored as features: about 10 minutes. Grouping and
scoring: 5 minutes. Run as two steps, each capped at 10 minutes.

## 7. Result

Run 2026-11-10 (approved by the user the same day). Encoding with four
taps took 72 s. Validation episodes, threshold from episodes 900–999,
floor at the median variance as in the gate's DINOv2 rows.

| Features | Transitions F1 | Precision | Recall | Drawer | Window | Lock | Cube |
|---|---|---|---|---|---|---|---|
| Block 2 | **0.63** | 0.74 | 0.55 | 99% | 89% | 53% | 13% |
| Block 4 | 0.56 | 0.63 | 0.50 | 98% | 58% | 53% | 15% |
| Block 8 | 0.47 | 0.45 | 0.50 | 92% | 53% | 46% | 29% |
| Block 12 (card 114's features) | 0.30 | 0.21 | 0.52 | 81% | 51% | 37% | 56% |
| Raw pixels (gate) | 0.62 | 0.52 | 0.76 | 99% | 98% | 41% | 92% |

Random scores 0.05–0.09. Block 12 recomputed here matches the stored
features' gate row.

1. **Boundaries: pass.** Block 2 reaches 0.63 against the bar of 0.55,
   level with pixels and above plain change with the true arm mask
   (0.56). The score falls with every deeper block. On block 2 the true
   arm mask changes nothing (0.63 against 0.63): the signal discounts
   the moving arm by itself, as designed. 26% of boundaries still sit
   where the arm came to rest.
2. **Units: fail.** Grouping on block 2 (696 label-free boundaries, 26
   patches in 8 groups), best group's IoU against k-means on the same
   features with 8 clusters:

   | Object (patches) | Groups from change | k-means |
   |---|---|---|
   | Drawer (17) | 0.17 | 0.30 |
   | Window (20) | 0.32 | 0.51 |
   | Button 0 (2) | 0.00 | 0.10 |
   | Button 1 (3) | 0.00 | 0.16 |

   - Every group lies on the drawer's or the window's side of the frame,
     and none on the table or the walls. But each object is split into
     four groups, and they cover only the strip the slide sweeps.
   - No button patch was grouped: at the label-free threshold too few
     boundaries include them.

**Deviation, decided before step 2:** groups were formed on episodes
900–999, not 0–899, since only held-out episodes have every frame
encoded within the budget.

**Not judged:** lock flips 53% on block 2; the cube 13% there against
92% on pixels, at a threshold that keeps fewer boundaries (1,429 against
3,493).

Verdict: **fail** (criterion 1 passed, criterion 2 failed).

## 8. Decision

**Revise.** The temporal half stands and the perceptual half does not.
Lasting change on an early block of the same frozen encoder finds when
and roughly where conditions change, with no slots, no segmenter and no
model of the arm, so card 117's boundaries no longer wait on card 116.
But "the object is where the change happens" is wrong as stated: a
slide changes only the strip it sweeps, so change gives a seed on the
object, not its extent. The deep blocks behave the other way round:
they lose where a change is (block 12: 0.30) but group by appearance
(k-means reaches 0.51 on the window). The next card should say how an
object's extent is grown from a change's seed, and whether the cube and
lock flips, both a few patches under the arm, need the higher-resolution
renders of card 116.3. For the user to choose.
