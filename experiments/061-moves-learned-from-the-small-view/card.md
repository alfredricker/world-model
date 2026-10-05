---
id: "061"
title: moves learned from the small view
rung: 1
serves: [C2, C1, P2, P7]
status: done
verdict: pass
arch_version: 12
date: 2026-10-05
---

# 061: moves learned from the small view

Drafted and started overnight on 2026-10-05, after card 060 (AGENTS.md,
"Overnight sessions"; the user's objective: a view smaller than the
map). The numbers below were fixed before the main run; the smoke tests
that shaped the fit are reported in section 7.

## 1. Question

Version 12 acts on MiniGrid's 7 × 7 view with occlusion, but the memory
it starts with was learned from play seen through a 13 × 13 window that
holds the whole map (card 057's declared exception). Can how the moves
change where things are (card 044's transformations), and how a tile
looks once the agent steps off it ("undraw"), be learned from play seen
through the same 7 × 7 occluded view the agent acts with, and does the
agent then act as before? C2 (one representation for acting and
learning), C1 (nothing from a view the agent does not have), P2, P7.

## 2. What changes

One component: what the move model and undraw are learned from.

| | Version 12 | This card |
|---|---|---|
| Play seen through | the 13 × 13 window (whole map) | the 7 × 7 view with occlusion, as in acting |
| Move correspondences (card 028) | counted over all rows | counted only over rows where both places were observed; a correspondence is kept where it is exact in at least 50 such rows and the place shows more than one appearance there; then, since a move sends every where by one transformation (card 044), the correspondences are fitted by least squares weighted by their rows, dropping the worst until all fit within half a place |
| Undraw | the place behind the agent after a forward step | the same forward steps the other way round: the agent on X at the centre after, and X ahead before (nothing behind is in view) |
| What a place with no token shows | the commonest appearance of places entering the window | the same, over places entering the 7 × 7 view where observed |

Declared exception, left for card 062: pick up, toggle and drop still
store what was in view in the 13 × 13 window.

## 3. Dependencies

Version 12 (cards 057, 060); card 028's counted move maps; card 044's
transformations; card 054's encoders. Literature: robust regression by
trimming (least trimmed squares, Rousseeuw 1984; not in papi);
MiniGrid's visibility rule.

## 4. Data check

The same stored play as version 12: 505,537 transitions in the key
world; under the 7 × 7 occluded view 11.8% of the 13 × 13 window is
observed per step (smoke test). A turn keeps 16 places of the 7 × 7
view in view, forward 42.

## 5. Feasibility gate

- **Upper bound:** the full window's fit (version 12's memory).
- **Trivial baseline:** card 028's counted maps on the 7 × 7 views
  without the observed-only counting and the trimmed fit (smoke test:
  the turns' transformations are singular, so no placements exist).

## 6. Success criteria and prediction

On the four encoders of card 054 (seeds 399–402), in every familiar
world:
1. **Moves:** each move's transformation equals the full window's, with
   residual 0.
2. **Undraw and the unseen appearance** equal the full window's on
   every tile the agent steps off in the stored play; a tile it never
   steps off may differ and is reported with which is right.
3. **Acting** with this memory (version 12's agent, seed 399): familiar
   worlds 100% with steps within 1% of card 060's; chained rooms with
   one door 100%; no wrong remembered tile.

**Prediction.** All three hold: the transformations are rigid, and the
places that turns and forward share are enough to fix them.

**Decision rules.** Keep (version 13) if all three hold; one declared
revision if one fails; stop if a transformation is not recovered.

**Budget.** About 10 minutes.

## 7. Result

**Smoke tests first** (5 key-world layouts, seed 399), which shaped the
fit before the main run:
1. Card 028's counting with hidden places left out but no further
   filter: left and right came out singular (no placements). Places far
   ahead are observed almost only when they are wall, so a far place
   "predicted" a turned place exactly whenever both were observed.
2. A correspondence kept only where the place before shows more than
   one appearance in the rows where both were observed: turns right, but
   forward picked up a few wrong pairs (residual 3.45).
3. The trimmed fit (section 2): every move exact, residual 0, from 16
   (turns) and 35 (forward) correspondences. Undraw first left out
   forward steps where the centre looked the same before and after
   (floor to floor); fixed to use every forward step.

**Main run** (`runs/061/fam_{399..402}.json`, `chain_399.json`).

| | Seed 399 | 400 | 401 | 402 |
|---|---|---|---|---|
| 1. Transformations equal the full window's, residual 0 | ✓ | ✓ | ✓ | ✓ |
| 2. Unseen appearance equal | ✓ | ✓ | ✓ | ✓ |
| 2. Undraw equal on tiles the agent steps off | 4 of 4 | 4 of 4 | 4 of 4 | 4 of 4 |
| Move outcomes (moved or blocked) as the full window's, rows | all | all | all | all |

The fifth centre appearance, the agent on the goal square, differs. The
agent never steps off the goal (the episode ends), so the full window
never saw the goal behind the agent and its undraw was a vector imagined
by transport (handle 130, no tile). The small view reads the goal ahead
before the step: the goal tile itself. The small view is right there.

**3. Acting** (seed 399, version 12's agent): familiar worlds 100% with
18.13, 19.13, 17.13 and 23.47 steps, the same as card 060 to the
hundredth (0% difference); chained rooms 100% at 1.61 × the shortest
route (card 060: the same); no wrong remembered tile. The other three
encoders give the same.

All three criteria hold.

## 8. Decision

**Keep** (version 13: the move model and undraw learned from the small
view). How moves change where things are needs no view of the whole
map: the few places a move keeps in view are enough to fix one rigid
transformation, which then holds for every where, seen or not. The
remaining part of card 057's exception is what pick up, toggle and drop
store as "in view" (card 062).
