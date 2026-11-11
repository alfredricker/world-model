---
id: "123"
title: regions of change — an event read as the connected regions that changed, each scored as an object
rung: 2
serves: [P7, P4]
status: draft
verdict:
arch_version: 27
date: 2026-11-10
---

# 123: regions of change

Roadmap step 2 (event segmentation), rung 2. Card
[122](../122-sharper-frames/card.md) ended with one reading of both its
failures: an event is not one top patch but a few separate regions that
change together (the button and the handle at a lock flip; where the
cube left and where it arrived). At 256 pixels a region of change may
itself be the object's extent when the whole object changes. The user
asked for this card (2026-11-10).

Biological counterpart: figure and ground are separated by common
fate, with regions that change together seen as one thing (Wertheimer
1923; Spelke 1990 for infants).

## 1. Question

Thresholded and split into connected regions, does the change between
just before and just after a touch give one region per changed thing,
whose extent matches the object? P7, P4.

## 2. What changes

One component: how card 120's signal is read. Nothing is trained.

```
120, 122:  signal per patch → its top patch (one place per event)
123:       signal per patch → threshold → connected regions (several places per event, each with an extent)
```

- **Signal:** card 122's, block 2, 32 × 32 patches, before and after
  each touch.
- **Threshold:** one number for every event, Otsu's split of the
  signal's logarithm pooled over all touches and patches of episodes
  900–999. No labels.
- **Region:** a 4-connected set of patches above it.
- **Scored** by the best region's IoU with the true patches of: the
  button at a lock flip; the cube where it was before the touch, and
  where it is after; the drawer and window (reported, since a slide
  changes only the strip it sweeps).

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A changed thing is one connected region | fails behind an occluder | holds | holds | fails behind the arm | declared limit |
| One threshold fits every kind of change | untested | untested | likely (flat colours) | fails here (section 5) | **fails**: found by this card |
| A touch brackets the event | no touch sense | none | none | yes | declared in card 120 |
| A patch shows the same place over time | no | no | yes | fixed cameras only | declared in card 119 |

## 3. Dependencies

Card 122's computed signals and masks (`runs/122/`); card 120's touches.
No rendering or encoding.

## 4. Data check

Card 122's: on the 32 × 32 grid a button covers 8 patches and the cube
8. Episodes 900–999 hold 753 lock flips and 549 cube moves inside a
touch; validation 722 and 558.

## 5. Feasibility gate

Run 2026-11-10 (`tools/card123/regions.py`, seconds). The pooled
threshold came out at 0.70, with 29% of patches above it.

| Best region's IoU with | Episodes 900–999 | Validation | Top patch alone |
|---|---|---|---|
| The button | **0.59** | **0.58** | 0.01 |
| The button, share of flips at 0.5 or more | 73% | 71% | |
| Where the cube was | 0.04 | 0.04 | 0.05 |
| Where the cube arrived | 0.04 | 0.04 | 0.04 |
| The drawer | 0.22 | 0.22 | 0.01 |
| The window | 0.32 | 0.31 | 0.01 |

- **The button comes out as its own region.** IoU 0.58, against 0.07
  grown by similarity from a true seed (card 122) and 0.06 for k-means.
  At 69–71% of lock flips the button and the handle are separate
  regions: the two places of one event.
- **The cube does not.** At this threshold where it left and where it
  arrived are merged into the region of the arm, which is next to both
  two frames before and after a touch.
- **No single threshold serves both.** Swept on either set, the
  button peaks at 0.59 near 0.7 and falls to 0.05 by 8.5; the cube
  peaks at 0.36–0.39 near 8.5–14 and is 0.04 at 0.7. A recoloured
  button is a weak change; a cube that has gone is a strong one.
- **Rules that set the threshold per event do worse** (episodes
  900–999): half of the event's top patch gives button 0.02 and cube
  0.08–0.09; a tenth gives 0.10 and 0.17–0.19; Otsu per event gives
  0.12 and 0.04.
- **Most regions are not objects.** A touch gives 17 regions on
  average. 25% lie mostly on the arm, 8% mostly on an object that
  changed, and the rest elsewhere.
- **The arm-masked bound did not work as a bound:** removing the arm's
  true patches also removes the button and the cube beside it (button
  0.16).

**Gate result:** passed for the button, failed for the cube, and
precision is poor. Validation was scored in the same run as episodes
900–999, before any criterion was written, so this card has a gate and
no judged result.

## 6. Success criteria and prediction

Not set. The gate shows the design as drafted cannot pass for the cube,
and a card whose test cannot be passed is not an experiment (AGENTS).
What a runnable version needs, for the user to choose:
- **The self first** (card 117, per patch, at 256 pixels). The arm is
  the common blocker: a quarter of the regions, the region the cube
  merges into, and the top patch in 23–32% of touches (cards 120, 122).
  With the arm known, the cube's two places can be split from it, and
  a threshold can be set on what is left.
- **Recurrence as the filter for noise** (card 118's idea, at the level
  of regions). A real variable's region comes back at every one of its
  events: the button's 8 patches at each of 700 flips. A noise region
  does not. Keeping regions that recur needs no threshold on strength.

## 7. Result

No main run.

## 8. Decision

For the user: revise toward one of the two routes above, or stop.
