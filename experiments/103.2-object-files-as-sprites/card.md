---
id: "103.2"
title: object files as sprite layers — propagation and discovery on pixels, with existence
rung: 1
serves: [P7, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 103.2: object files as sprite layers

Drafted and run on 2026-10-12, in series
[103](../103-tracking-with-general-priors/card.md). The method took shape
over smoke tests on the first 3,000 steps of worker 8 (section 5). The
criteria were fixed before the full runs.

## 1. Question

Can an object file be a sprite: a template of its object's own pixels, a
mask, a world position and an existence count? Here a file is followed by
searching near where it was, and a new one is born only from what the world
and the existing files leave unexplained. If so, does it keep a creature's
identity better than card 098.6's files did on tile tokens? It must use no
tiles, no kinds and no ground, so the ground under a cow drops out of what
identifies it. Rung 1. Serves P7 (units tracked across frames, from
continuity) and P2.

## 2. What changes

One component: the object files level (cards 098.5–098.6). It now works on
pixels in the world frame, not on tile tokens.

```
098.6:      files on tile tokens; identity by the tile's code and vector; kinds from recall
this card:  each step, the frame placed in the world (origin known) is explained by
              the world: a pixel canvas of the episode (written where no file claims and the camera does not carry)
              the camera: card 103.1's learned pixels
              files: template x mask at a world position; each searches displacements up to 9 px with a learned
                     prior p(d), and is accepted if 60% of its visible mask matches; out of view it is kept as it was
            residual pixels no file claims: connected pieces; a piece whose content matches the canvas at one
            displacement (or two) came from there: a file is born and its source is revealed (the canvas learns the
            ground); at many displacements (uniform content) or none (appeared), it is the world's; an appeared
            piece is a tentative file, reported once it moves
```

- **A pixel differs** when a channel differs by more than 15 of 255 levels,
  or by four times the frame's noise if that is larger.
- **Sleep** is skipped and ends nothing: it moves nothing.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A file moves as one 2-D shift per step | yes | yes | rigid parts in the image plane only | declared (cohesion) |
| Displacements up to 9 px per step | yes | per grid scale | depends on the rate | declared (search window) |
| Files are drawn over the world (layer order) | yes (top-down) | yes | no: things go behind others | declared |
| Objects are what moves (reported once moved) | movers only | clicked things change in place | yes | declared; changed-in-place things stay the world's |
| Tiles, kinds, ground | not used | — | — | removed |

## 3. Dependencies

- Card 103.1's camera pixels (criterion 2 passed).
- Origins:
  - the gate uses the evaluator's;
  - the main arm uses card 103.1's robust placing, which failed its 98% bar
    at 94.7%. This arm measures what that costs.
- Card 098.5's play (workers 8–9) and IDF1.
- Propagation plus discovery with existence: SCALOR, SILOT, AXIOM, PMBM in
  LITERATURE.

## 4. Data check

Workers 8–9: 55,655 object frames of cows, zombies, skeletons and arrows in
view. Worker 8 alone: 27,364.

## 5. Feasibility gate

- **Upper bound:** 098.6's files on the true state, IDF1 0.934.
- **Reference:** 098.6 on tile tokens, 0.717.
- **Trivial baseline:** identity by place, 0.009.

**Smoke tests (3,000 steps):**
- The first version scored 0.46.
- Sleep had reset every file. Sleep moves nothing, so files now persist
  through it: 0.535.
- Two arms made it worse, so they are off:
  - a lost file searching the whole view (0.43): look-alike cows took one
    another's files;
  - birth pieces within 2 px grouped (0.46).

## 6. Success criteria and prediction

1. **Identity on Crafter (gate, evaluator's origins).** IDF1 over cows,
   zombies, skeletons and arrows above 0.717 (098.6 on tile tokens),
   workers 8–9.
2. **With learned origins.** Card 103.1's robust placing, worker 8: IDF1
   within 0.05 of the evaluator's origins on the same worker.

**Prediction:** criterion 1 fails, at about 0.5–0.6 (smoke test 0.535);
files break when a creature stands still long enough to be read as world,
and when look-alikes meet.

**Decision rules:** keep if both hold. Revise if criterion 1 fails while
files hold objects for long stretches (median life above 20 frames). Stop
if IDF1 is below 0.5.

## 7. Result

Run 2026-10-12 (`tools/card103.2/track.py`, under a minute per worker,
capped). Numbers in `results.json`.

| | IDF1 | ID recall / precision |
|---|---|---|
| 098.6, files on the true state (upper bound) | 0.934 | 0.924 / 0.945 |
| 098.6, files on tile tokens (reference) | **0.717** | 0.632 / 0.827 |
| **This card, evaluator's origins, workers 8–9** | **0.531** | 0.506 / 0.559 |
| This card, evaluator's origins, worker 8 | 0.522 | 0.507 / 0.537 |
| This card, card 103.1's learned origins, worker 8 | 0.303 | 0.479 / 0.222 |
| Identity by place (baseline) | 0.009 | |

- **Criterion 1 fails:** 0.531 against 0.717. ID recall by kind: cows
  0.50, zombies 0.61, skeletons 0.52, arrows 0.39.
- **Criterion 2 fails:** with learned origins, IDF1 falls by 0.22. A
  misplaced step leaves residuals everywhere, and they are born as files:
  59,040 detections against 25,849.
- **Where identity is lost** (workers 8–9, evaluator's origins; 55,655
  creature frames):
  - 65.7% of the frames have a file on the creature.
  - 12.7% have none because the creature has not moved since it came into
    view. "Objects by what they do" gives a still creature no file. 098.6's
    kinds, learned from recall, gave every cow one.
  - 21.6% have none after the creature has moved: files are lost and born
    again. Median life is 10 reported frames, with 1.6 files per creature
    followed.
  - 25% of reported detections sit on scenery or plants.
- **What the smoke tests showed:**
  - Resetting the files at sleep cost 0.07.
  - A view-wide search for lost files let look-alike cows take one
    another's files.

## 8. Decision

**Revise** (2026-10-12). The rules did not foresee this case: IDF1 is above
the stop line (0.5), but files live a median of 10 frames, not 20.

A sprite template does separate a creature from the ground beneath it,
which 098.6's tile codes could not. But this tracker has none of what made
098.6 work:
- kinds learned by recall, so a still creature of a moving kind has a file;
- existence earned over many views;
- the pair test that tells a hidden token from a gone one.

The revision puts sprite templates under 098.6's files level, in place of
tile codes and vectors: identity from the creature's own pixels, the file
logic from 098.6. Learned origins must first be right (card 103.3), since
misplacement alone took 0.22.
