---
id: "118"
title: conditions as stable states — an object's stretches between boundaries cluster into codes that recur and match the scene's conditions
rung: 2
serves: [P7, P4]
status: approved
verdict:
arch_version: 27
date: 2026-11-01
---

# 118: conditions as stable states

The last step of event segmentation (Roadmap step 2), the one that
completes rung 2. It runs only if cards 116 and 117 are kept. Conditions
are the stable stretches between event boundaries (the user, 2026-11-01).
Here those stretches are turned into codes that recur across episodes.
For example: drawer open or shut, lock red or white, cube on the table or
held. Biological counterpart: the steady activity patterns that cortex
holds within an event recur when the same kind of event returns
(Baldassano et al. 2017), and SEM-2's event models are reused each time
their event recurs (Nguyen et al. 2024).

## 1. Question

Clustered without labels, do the stretches of each non-self slot between
card 117's boundaries give codes that name the scene's true condition
values, and do the same codes recur in episodes never seen? Rung 2, P7
(discover the units worth tracking), P4 (keep distinctions that change
outcomes).

## 2. What changes

Only a clustering step, on top of cards 116 and 117.

```
117:  slots → self (influence) → boundaries per non-self slot
118:  each stretch between two boundaries of a slot → its mean slot vector
      (middle half of the stretch) → k-means codes fitted on training episodes
      → applied to held-out stretches
```

- **Stretches:** at least 2 model steps long. The mean is taken over the
  middle half, away from the change at either end.
- **Codes:** k-means over all non-self stretches of training episodes
  0–899 (card 116's slots run on them too). k is chosen by silhouette
  between 4 and 32, so the number of codes is learned, not given. Codes
  are shared across slots, so "open" can be one code wherever it
  appears.
- **The truth** (scoring only), at a stretch's middle, for the object its
  slot covers (the true mask's majority):
  - drawer: open (beyond 4 cm) or shut;
  - window: open or shut;
  - each lock: red or white;
  - cube: on the table, held, or in the drawer.
- **Measure:** for each variable, the accuracy of the best map from codes
  to values, averaged over the values (balanced, so a rare value counts
  as much as a common one). Codes are fitted on training episodes,
  scored on validation.
- **Recurrence:** for each code that maps to a value, the share of
  validation episodes containing that value in which the code appears.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A condition is an object's stable state | holds (a door open, a furnace lit) | holds | holds (a cell's colour) | holds | general |
| One shared codebook across objects | holds | holds | holds | holds | general; per-object books are the alternative |
| k chosen by silhouette | a setting | a setting | a setting | a setting | declared setting |

## 3. Dependencies

- Cards 116 (slots) and 117 (self, boundaries), both kept.
- ASOT (`2404_01518`) for stable segments clustered into codes shared
  across videos; SEM-2 for reusable event models; FCDL (`2406_03234`) for
  codes as contexts. All in LITERATURE's segmentation review.

## 4. Data check

From the simulator, the gate counts each value's held-out stretches:
drawer open and shut, window open and shut, each lock red and white,
cube on the table, held, and in the drawer. A value needs at least 50
validation stretches to be judged; one with fewer is reported only.

## 5. Feasibility gate

Each step under 10 minutes, capped (rule 5).
1. **Upper bound:** the same clustering on true objects: DINOv2 vectors
   pooled in the true masks, over the true stable stretches (between the
   simulator's events). It shows whether the features separate the values
   at all. Card 114's probes suggest they do: the lock read at 100%, the
   drawer at R² 0.977.
2. **Baselines:**
   - **Frames, not stretches:** k-means on every frame's slot vectors,
     with no boundaries. It shows what the boundaries add.
   - **No objects:** k-means on each frame's mean DINOv2 vector, which
     shows what objects add.

Result of the gate: not yet run.

## 6. Success criteria and prediction

On validation episodes, codes from training episodes:
1. **Codes name the conditions:** balanced accuracy ≥ 0.9 for the drawer,
   the window and both locks. The cube is reported.
2. **Stretches beat frames:** balanced accuracy averaged over the four at
   least 0.05 above frame clustering, and above the no-objects baseline.
3. **Codes recur:** each code that maps to a value appears in at least
   80% of validation episodes containing that value.

Meeting all three completes rung 2: from pixels and actions alone, the
agent has objects, events and a reusable vocabulary of conditions. That
is the vocabulary that preconditions (rung 4) and backward and forward
search over conditions (rung 6) are written in.

**Prediction.**
- Locks pass easily (colour), and the drawer and window pass (position
  is large).
- Criterion 2 is the risk. In a slow, clean scene, frame clustering may
  match stretches, because most frames lie inside some stretch anyway.
  If so, the boundaries are not needed for naming conditions, only for
  marking when they change. That would be a finding about this world,
  and the card says so.
- The cube's "held" may sit in the self (card 117's prediction), leaving
  only "on the table" and "in the drawer".

**Budget:** slots on training episodes, 10 minutes (two capped halves);
clustering and scoring, 5 minutes. No training beyond k-means.

**What comes next, if kept:** preconditions (Roadmap step 4, rung 4).
For each code change on an object, the contrast between the self's
contacts that produced it and those that did not names the conditions it
needs. For example, a pull opens the drawer only when the lock's code is
white (card 111's contrast; influence contexts, FCDL).

## 7. Result

Appended after the run.

## 8. Decision

Appended after the run: keep, revise or stop.
