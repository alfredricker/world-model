---
id: "042"
title: recall in two levels, codes for the same thing and vectors for similar things
rung: 0
serves: [P4, P10, P6, C2]
status: done
verdict: fail
arch_version: 5
date: 2026-10-01
---

# 042: recall in two levels, codes for the same thing and vectors for similar things

## 1. Question

Cards 038–039's planner works on encoder vectors and fails the familiar
worlds: the either and both worlds miss the goal bar in every seed. Recall
has one learned weighting of the vectors per action, so it cannot make a
key's colour count at doors without making it count everywhere (section 5).

Does recall in two levels bring every familiar world back while the planner
stays on vectors? The two levels are the same thing (equal codes) and
similar things (the vectors). Serves P4 (keep the distinctions that change
outcomes), P10 (retrieve the relevant experience), P6 (exact effects) and
C2 (one latent space: codes are the encoder's quantisation of the vectors).

## 2. What changes

One component: recall for pick up, toggle and drop. That covers the
category, the result, and the stored tries a condition draws on. Also the
result of draw and undraw, how the agent's own place looks.

```
card 039:  k_i(q) = exp(−d_λ(q, i))   one weighting λ of front, held and view vectors per action
           P = own tries + α · neighbours' vote;   outcome classes: which places change
card 042:  same thing:    k1_i(q) = [codes of front equal] [codes of held equal] exp(−chamfer_λ1v(views))
           similar thing: k2_i(q) = card 039's kernel over the vectors (λ2)
           P = (N + β s) / (N + β),  N = Σ k1_i n_i,  s = card 039's vote with k2
           outcome classes: which places change, and into which codes
```

- **Codes** come from card 038's arm A encoders (card 037's recipe). Each
  of 4 codebooks gives a tile's piece the nearest used code, or "new"
  (card 035), with fresh codes for new pieces (card 036). Nothing says
  what a code means.
- **A "new" code tuple** matches only its own vector, so a never-seen
  thing has no same-thing tries until it is tried (transfer by tries).
- **The fit.** λ2, the view weights λ1v ≥ λ2v and β are fitted together by
  leaving out one stored key at a time. Card 039 left out a whole (front,
  held) pair; the same-thing level needs that pair's other views.
- **Results.** Where the same-thing level carries the prediction (its
  tries outweigh β), the result is what the thing became. Otherwise it is
  imagined as in card 038. A condition's alternatives come from tries on
  the same thing, or on similar things when there are none.
- **Unchanged:** the planner (card 039: conditions from memory, the view
  as a set), walking, recall for the moves, and the encoders.
- **Declared exception** (CHARTER rule 8). The same-thing level reads all
  four codebooks for every action, so each thing is its own kind. "Each
  rule reads only the codebooks it needs" waits for the card that resumes
  colour transfer, paused by the user (2026-09-30). Tries stay stored one
  key each, and are pooled only when a query is made.

## 3. Dependencies

- Card 039's planner. Card 038's upper bound passed: 100% in the four
  familiar worlds on one-hot vectors.
- Arm A's encoders (card 038) and their codes:
  - card 035's reading and card 036's fresh codes passed;
  - every tile is named apart on seed 399.
- Methods: hierarchical back-off of own counts to a shared vote (MacKay
  and Peto 1995, already behind α); nearest-code quantisation
  (`1711_00937`); discrete latents for planning from pixels (`1705_00154`).

## 4. Data check

On seed 399, toggle has 612–890 stored keys per world, 71–72 of them at a
closed door holding a key; 15 distinct code tuples occur in front, and
toggle has 10 outcome classes.

Each door-and-key pair is stored in 4–8 views, so the same-thing level
has other views to learn from when one is left out.

## 5. Feasibility gate

- **Upper bound:** card 038's arm 1 (one-hot vectors), 100% in all four.
- **Trivial baseline:** card 038's arm A, the same encoders with one
  level. Criterion 1 passed in 0 of 3 seeds: either 66–90%, both 15–45%.
- **Why one level fails, and why vectors alone are not enough:** appendix A.
- **Shakedown** (this card, seed 399, 30 layouts per world, from
  `tools/card042/code_recall.py`, `runs/042_shakedown_399.json`):

| | key | switch | either | both |
|---|---|---|---|---|
| Goal reached | 30/30 | 30/30 | 30/30 | 30/30 |
| Steps / card 029's | 0.98 | 0.97 | 0.94 | 0.97 |
| Held-out effects, lowest action | 1.0 | 1.0 | 1.0 | 1.0 |
| Toggle at doors, held out | 100% | 98.6% | 97.2% | 100% |
| Seconds per layout | 0.59 | 0.59 | 0.65 | 0.38 |

- **New colours, same run.** Key world: goal 30/30 for both colours, but
  first-sight effects 0.0, as in two of card 038 arm A's three seeds.
  Switch world with a new-colour door: 1/30 (yellow), 5/30 (purple),
  against 6–7% in card 038.

## 6. Success criteria and prediction

Arm A, seeds 400–404, 500 layouts per familiar world.

1. **Familiar worlds.** Card 038's criterion 1 in all four worlds, in at
   least 4 of 5 seeds:
   - held-out effects ≥ 0.999 for every action;
   - goal in ≥ 99% of layouts;
   - mean steps within 5% of card 029's.

   This shows that identity from codes gives the vector planner what one
   weighting could not. The baseline is 0 of 3.
2. **Nothing lost on new colours.** Key world with yellow and with purple:
   goal in ≥ 99% of layouts, in at least 4 of 5 seeds; card 038 arm A had
   100%. Reported without a bar:
   - first-sight effects (card 038 arm A: 0.0–1.0);
   - the switch world with a new-colour door (6–7%).

   Colour transfer is paused.

**Prediction.**
- Both criteria pass in 5 of 5 seeds.
- First-sight effects fail where the vectors cannot tell a new colour from
  a familiar one, as in card 038 arm A.
- The new-colour switch world stays below 20%.

**Budget.** About 8 minutes per seed, so about 40 minutes for five. The
main run is handed to the user:

```
bin/prun python tools/card042/code_recall.py --arm A --seeds 400-404 --layouts-b 100 --out runs/042_armA.json
```

## 7. Result

Main run, arm A (`runs/042_armA.json`). The user stopped it after seed 402,
once criterion 1 could no longer pass (two of five seeds had failed).

| Seed | key | switch | either | both | Criterion 1 | Criterion 2 (new colours, key world) |
|---|---|---|---|---|---|---|
| 400 | 100% | 100% | 100% | 100% | pass | 100%, 100% |
| 401 | 100% | 100% | 87.8% | 100% | fail | 100%, 100% |
| 402 | 100% | 100% | 90.6% | 100% | fail | 100%, 100% |

- **Criterion 1: fail.** It passed in 1 of 3 seeds; card 038 arm A passed
  in 0 of 3.
  - Held-out effects were 1.0 for every action in every seed and world.
  - In key, switch and both, steps equalled card 029's (16.38, 17.27,
    21.77).
- **Criterion 2: on track** (100% in 3 of 3) but not finished.
- **Reported:**
  - first-sight effects of the new colours: 0.0, 0.0 and 1.0 by seed, as
    in card 038 arm A;
  - the switch world with a new-colour door: 11–61%, against 6–7% in
    card 038.
- **Why the either world fails** (seed 401, 8 of 60 layouts):
  - The agent picks up the wrong key and drops it again until the budget
    runs out.
  - The planner checks "holding the green key" against a stored view in
    which the green key also lies on the floor, a situation that cannot
    occur. The same-thing level has no tries for it.
  - The vector level then answers. Its held weight in this world is 0.02,
    so it gives the base rate (P(open) 0.55) instead of the 112 stored
    tries in which that key never opened that door.
  - With the held thing removed from such views (a test that found the
    empty hand by name), 0 of the 60 layouts fail.
  - The failure is in how the planner builds hypotheticals, not in recall.

## 8. Decision

**Revise** (the user, 2026-10-01). Keep the code level in recall: effects
were exact everywhere, and three of the four worlds matched card 029 in
every seed. The failure sits in the planner, which checks a needed part
against a situation that cannot occur. Card 043 makes the planner's
hypothetical situations consistent and reruns these criteria.

## Appendix A: why one level fails, and vectors alone

Drafting checks, 2026-09-30 to 2026-10-01; scripts in the session scratchpad.

**One level** (card 039's planner, arm B, seed 399, single stored keys
left out):
- At a door holding a key, recall's held-out prediction is right in 34%
  of key-world cases. It was a coin flip (log-likelihood −0.73), and
  these cases are 82% of the fit's remaining error.
- Adding weight on the two most colour-bearing dimensions fixes doors
  (−0.73 → −0.16) but harms everything else (−0.010 → −0.040).
- Acting reached the goal in key 17/30, switch 30/30, either 30/30 and
  both 12/30.

**Two levels on vectors alone** (a fitted sharp kernel as the same-thing
level):
- held-out door cases were right in 100% of every world;
- acting reached the goal in 30/30 in three worlds but 22/30 in either;
- identity rested on exact pixel repeats.
