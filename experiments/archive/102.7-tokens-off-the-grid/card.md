---
id: "102.7"
title: tokens off the grid — patches that do not fit the world's tiles
rung: 1
serves: [P7, P2]
status: done
verdict: pass
arch_version: 27
date: 2026-10-12
---

# 102.7: tokens off the grid

Drafted and run overnight on 2026-10-12, under the user's objective: the
encoder and model as portable as possible to Minecraft and robotics,
"careful about strong grid priors". It adds a row to the series plan
([102](../102-minigrid-assumptions/card.md)).

## 1. Question

Every token so far is one of the world's own tiles. MiniGrid's adapter
reads the simulator's grid, and Crafter's cuts 7 × 7-pixel tiles at the
renderer's phase. Minecraft and a robot's camera have no tiles, and their
views move by amounts that no patch size fits.

So how much do the encoder, its codes and card 102.5's placing and
contingency depend on patches that fit the world's tiles? The patches are
cut at the right size but the wrong phase, and at the wrong size. This is
a diagnostic: it measures the grid prior before a card removes it. Rung 1.
Serves P7 (units from continuity) and P2 (belief needs the view's motion).

## 2. What changes

One component: the adapter's cut of a frame into tokens.

```
now:        tokens = Crafter's 7 x 7-pixel tiles at the renderer's phase (9 x 7 world tiles)
this card:  A  7 px from (0, 0)   the tiles (control)
            B  7 px from (3, 3)   the right size, the wrong phase (8 x 6 patches, each across four tiles)
            C  8 px from (0, 0)   the wrong size: the view moves 7 px, 7/8 of a patch (8 x 6)
            each patch resampled to 7 x 7 for the encoder; the same recipe, codes and placing for all
```

- **The encoder recipe** is card 098.3's, simplified alike for all three:
  - positives are places drawn identically within a frame, and the same
    place at the next frame when the agent did not move;
  - the rest are negatives;
  - the code radius is the distance that best separates those pairs.
- **Placing and contingency** are card 102.5's, unchanged, with candidate
  shifts in whole patches.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Tokens are the world's tiles | yes | yes (cells) | no | tested: B and C remove it |
| The view moves by whole tokens | yes | yes | no | tested: C breaks it |
| Patches are square, on a lattice | adapter's choice | adapter's choice | adapter's choice | declared |

## 3. Dependencies

- Card 098.3's encoder recipe and card 098.5's Crafter play (worlds 0–2
  for training, 8 for testing); card 102.5's placing and contingency.

## 4. Data check

The same play for all three forms: 90,000 training steps and 30,000 test
steps, about a quarter of them moves.

## 5. Feasibility gate

- **Upper bound:** form A, the tiles, is the control. Card 102.5 read the
  view's motion right on 93.0% of steps on 098.5's tokens.
- **Trivial baseline:** "the view never moves", which is right on every
  step that is not a move.

## 6. Success criteria and prediction

1. **Phase does not matter.** B reads the view's motion within 2 points of
   A, and finds the controlled codes as A does.
2. **Size does matter, and by how much.** C's motion read right, against
   A and against "never moves". It is reported, with no bar: C is expected
   to fail, and the size of the failure is the measure of the grid prior.

**Prediction.**
- B is close to A, since the view still moves by whole patches. It has
  more codes, because each patch mixes four tiles.
- C reads moves no better than "never moves": no whole-patch shift
  explains a 7-pixel move.

**Decision rules.**
- **Keep** (as a finding) if criterion 1 holds; the grid prior is then
  the patch size alone, and the next card makes shifts continuous.
- **Revise** if B already fails: then phase matters, and the encoder, not
  only the placing, depends on the grid.

**Budget.** About 5 minutes per form, capped.

## 7. Result

Run 2026-10-12 (`tools/card102.7/patches.py`, about 2 minutes per form,
capped).

| Form | Codes | View's motion read right | Moves / non-moves | Day / night | Controlled codes: found, evaluator's, both |
|---|---|---|---|---|---|
| A, the tiles | 1,226 | 95.5% | 80.2% / 99.8% | 96.9% / 82.3% | 121, 20, 12 |
| B, wrong phase | 2,654 | 94.4% | 74.8% / 99.9% | 95.4% / 85.4% | 592, 53, 29 |
| C, wrong size | 1,484 | 78.0% | **0%** / 100% | 77.7% / 80.6% | 706, 15, 6 |
| "The view never moves" | — | 78.0% | 0% / 100% | | |

- **Criterion 1, the motion part, holds.** B reads the view's motion within
  1.1 points of A. The phase of the cut does not matter while the view moves
  by whole patches.
- **Criterion 2: size is everything.** C reads no move at all; its result
  equals "the view never moves". No whole-patch shift explains a 7-pixel
  move, so the placing step keeps the view still.
- **Contingency did not hold, even on the tiles.** The simplified encoder
  makes about 1,200 codes (098.3's recipe: 85), and codes seen rarely
  reach U = 1 by chance (the highest other code in every form). Criterion 1's
  second half is therefore unreadable here: it needs 098.3's full recipe or
  a test against shuffled actions (card 102.5's revision).

## 8. Decision

**Keep, as a finding** (overnight, 2026-10-12). The tile-grid prior sits
in one place: the view's motion is read only in whole tokens. Cutting
patches at another phase costs about 1 point; cutting them at another size
loses every move. The next card reads the view's motion in pixels before
cutting, then cuts patches on a lattice fixed in the world frame. Tokens
then stay aligned with the world whatever the camera does, which is what
Minecraft and a robot need.
