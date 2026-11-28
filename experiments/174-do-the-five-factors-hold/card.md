---
id: "174"
title: do the five factors hold — card 173's label-free fit across state sizes and seeds, each variable read through its own factor, judged on fresh episodes
rung: 2
serves: [P7, P4, P1]
status: done
verdict: fail
arch_version: og5
date: 2026-11-28
---

# 174: do the five factors hold

OGBench `visual-scene`. Drafted at the user's go-ahead, 2026-11-28 ("sounds good", to: the ladder and three seeds as [card 173](../173-tight-box-sparse-change-factors/card.md) planned them, each variable read through its own factor, the bound dropped or rebuilt, the lock factors' movement located before a bar is set on it). Approved by the user the same day ("approved"). Section 4 was measured while drafting, with no training.

## 1. Question

Card 173's one label-free fit (8 coordinates, one seed) gave a factor per scene variable: cube, drawer, window and two locks. Its gate failed, so no second seed was fitted and the state's size was never chosen from data. Does the result hold over three seeds, at a size picked by the held-out loss, on episodes not yet looked at? Rung 2; Roadmap step 2. Serves P7, P4 and P1.

## 2. What changes

One component: the state's size is chosen from a ladder by the held-out loss, as card 173 declared and did not run. The loss, encoder, frozen pose term, moments and factor rule are card 173's, unchanged. og5 is untouched.

```
state      K numbers in [−1, 1] from one frame (card 172's encoder)
loss       InfoNCE over 17 moments, temperature 0.5, pose term frozen  +  1.0 × change between consecutive steps
size       K* = the smallest of {4, 6, 8, 12, 16} whose held-out loss is within 0.02 nats of the lowest (seed 0, 4,000 steps each)
factors    card 172's rule on 5-step changes, no labels
then       three seeds at K*
```

What differs from card 173 in how the result is judged, each for a stated reason:

- **Each variable is read through its own factor.** Card 173 measured reading by distance over the whole state, where stray movement in four other factors is added to every pair. Card 168's slots, the reference, are read slot by slot. Which factor is whose is named by truth, as for the slots.
- **Stray movement is measured per factor.** Card 173's measure summed the stray movement of all five factors and divided by the movement of one, so a state with five clean factors that each stray by 7% scored about 35%. The new measure asks of each factor alone: how much does it move when its own variable does not?
- **No bound is claimed for factoring.** Pairs chosen by truth hide the one-variable changes that sort variables onto coordinates (cards 172, 173). The evidence that criterion 2(b) can be passed is card 173's fit, which passed it for four of five. Card 173's bound is kept as the bound for reading and stray movement.
- **Fresh episodes.** The bars below were set knowing card 173's fit on the validation episodes. The criteria are therefore judged on 100 episodes never used: 900 to 999 of the play file (training used 0 to 449).
- **Half precision.** Card 173's fits ran with no limit on the GPU's memory and needed more than 8 of its 16 GB. This card's fits run the encoder in half precision: 6.7 GB at peak, measured. The gate checks that this reproduces card 173's fit.

Not in this card: any change to the loss or the encoder, values in the units a skill needs, the hidden cube, any use by the agent.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Near moments differ in few of the scene's variables | holds, except nightfall | the same | mostly | holds | general; scene-wide change is a declared exception |
| The agent's pose is given and explains its own body in view | holds (C1) | holds | no body: the pose term is empty | holds | general (C1) |
| A fixed camera: a query can learn where to look | no: the view moves | no | yes | fixed cameras only | declared, as in cards 119–173 |
| The scene's variables are few enough for a ladder of sizes | no: open terrain needs a state per region | no | varies | a table top, yes | declared limit; one fit per size |
| Most single steps change nothing in the scene | holds while walking | holds | no | holds | declared |
| A variable the body covers cannot be read from one frame | holds | holds | no body | holds | general; holding its value is the belief's work, not this state's |

## 3. Dependencies

- Card 172's features, encoder, pose term and scoring; card 173's fit and factor rule (`tools/card173/`); card 173's fits `k8_s0` and `bound8` in `runs/173/`.
- Zimmermann et al. 2021; SlowVAE (Klindt et al. 2021); Locatello et al. 2019 (seeds matter as much as methods; no label-free choice of settings is known to pick the factored model). All in LITERATURE.
- og5's belief keeps a variable's last value while it cannot be seen (card 169), which is where a covered button's state belongs.

## 4. Data check

Measured 2026-11-28 on the 100 validation episodes with card 173's fits; nothing trained (`tools/card174/locks.py`, `score174.py`; `runs/174/`).

**What moves the lock factors when no lock flips.** The hand on the button, for most of it.

| Change of a lock's factor over 5 steps, the lock's state the same at both ends | First lock | Second lock |
|---|---:|---:|
| Within 3 steps of a press (12% of windows) | 0.23 | 0.23 |
| Elsewhere | 0.007 | 0.041 |
| A real flip, for scale | 0.92 | 0.85 |

- In the 5% of frames where the hand is nearest a button, the lock's reading is off its usual value by 0.46 to 0.48; elsewhere by 0.04 to 0.15. Whether the hand covers the button or the press itself changes its look was not separated.
- The second lock's factor also drifts slowly: its change with the state constant grows from 0.03 over one step to 0.15 over 40. The pose and the rest of the scene together explain 17% of it, either alone under 4%. Not explained.

**The measures this card uses, on fits that exist.**

| Validation episodes | Card 173's label-free fit | Card 173's bound (K = 8) | Encoder with random weights |
|---|---:|---:|---:|
| Reading through own factor (AUROC), five variables | 0.970–0.990 | 0.982–1.000 | 0.52–0.63 |
| Share of a variable's change in its own factor | 73–96% | 51–88% | 16–19% |
| A factor's movement when its variable does not differ, against when only it does | 2.4–7.5% | 0.0–6.8% | 62–105% |
| The same, lock factors, in windows within 3 steps of a press | 10.7%, 12.1% | 0.0%, 0.4% | 88%, 91% |

Every window is kept in the third row, presses included; the fourth row is reported beside it.

**Memory.** A fit in half precision peaks at 6.7 GB on the GPU and 5.4 GB of RAM; in full precision it does not fit in 8 GB. Scoring peaks at 0.85 GB on the GPU. Every run is capped at 8 GB of RAM and half the GPU's memory, and prints its peaks.

## 5. Feasibility gate

- **Fresh episodes built:** 100 episodes, about one minute, 2.6 GB, written with pauses.
- **Trivial baseline** (the encoder with random weights, K = 8) and **reference** (card 168's slots, each variable on its own slot) scored on the fresh episodes.
- **Bound for reading and stray movement:** card 173's `bound8` scored on the fresh episodes; no new fit. It must reach criterion 1 and 2(c).
- **Replication in half precision, the go or no-go:** K = 8, seed 0, 4,000 steps. On the validation episodes it must give what card 173's fit gave: each of the five variables its own factor, at least four with 80% of their change there, and reading through the own factor at 0.95 or more for all five. If it does not, half precision changed the result; the card stops and the user decides whether fits may use more of the GPU.

Result of the gate: **the replication step failed by its own terms; the six fits of the main run have not been made.** The user decides what follows, as the step says.

| Gate step | Result | Verdict |
|---|---|---|
| Fresh episodes built | 100 episodes, 74 s, 2.6 GB | done |
| Trivial baseline on fresh episodes | reading 0.53–0.68; share 17–19%; a factor's stray movement 61–105% | far from every bar |
| Reference: card 168's slots on fresh episodes | reading 0.984–1.000 | |
| Bound (`bound8`) on fresh episodes | reading 0.985–1.000; stray movement 0.0–6.4% | reaches criteria 1 and 2(c) |
| Card 173's own fit on fresh episodes (no bar) | reading 0.982–0.990; share 74, 96, 94, 92, 82%; stray 2.2–7.9% | as on validation |
| Replication in half precision: each variable its own factor | five factors, each variable a different one | pass |
| Replication: reading through own factor at 0.95 for all five | 0.968–0.989 | pass |
| Replication: 80% of change in own factor for four of five | 79.1, 95.2, 92.9, 90.4, **57.0%** (cube, drawer, window, lock, lock) | **fail: three of five** |

- **The refit is a different solution, not a copy.** Same seed, but the variables landed on other coordinates (the cube on the coordinate that held a lock in card 173's fit). Small differences in arithmetic grow over 4,000 steps, so a refit behaves as a second draw. I wrote the step as if half precision would give the same fit; that was a mistake in the gate, and it cannot tell precision from ordinary run-to-run difference. A full-precision refit could, and needs about 12 GB of the GPU.
- **What the second draw shows:** the same five factors and the same reading; the cube a hair under the bar; the second lock at 57%, with 22% of its change in the window's factor and 12% in the other lock's. Stray movement per factor 2.7–9.3%. Held-out loss 0.958 (card 173's fit 0.944).
- On the fresh episodes the same refit gives 81.3, 95.3, 92.8, 89.6 and 55.2%: four of five, the cube a hair over the bar instead of under. The step was declared on the validation episodes and stays failed; the cube sits on the line and the second lock is the real shortfall.
- Memory, measured: a fit peaks at 6.71 GB on the GPU, scoring at 0.85 GB.

**The card's one change of design** (the user, 2026-11-28: "do 1"): the main run goes ahead in half precision although the replication step failed. The refit `k8_s0h` stands as the ladder's K = 8 fit and as a first draw of the question the card asks. Criteria and bars are unchanged.

## 6. Success criteria and prediction

Only if the gate passes: four more sizes at seed 0 (K = 8 is the gate's fit), K* by the rule, two more seeds at K*. Six fits of about 5 minutes. Judged on the 100 fresh episodes, the mean over three seeds, the spread reported. Truth for scoring only.

1. **Each variable can be read through its own factor:** AUROC at least **0.95** for all five (card 168's slots on validation: 0.976–1.000).
2. **Each variable has its own factor.** (a) Each of the five is the best match of a different factor, in at least two seeds of three. (b) Where only one variable differs over 5 steps, at least **80%** of the state's change is in its factor, for at least four of five. (c) Each variable's factor moves, where that variable does not differ, under **10%** of what it moves where only that variable does, for all five.
3. **The count does not depend on the seed:** the three seeds at K* give the same number of factors.

Reported, not judged: K* and the held-out loss, number of factors and criteria at every size; reading by whole-state distance; the lock factors near a press; card 141's frames before a touch.

Prediction. Replication in half precision: likely (four in five). The held-out loss I expect to keep falling with size, so K* is 12 or 16, and I do not know whether factoring survives there; that is the main risk. Three seeds agreeing on five factors at K*: about even. All three criteria: about one in four. If K* is large and factoring fails there while K = 8 holds, the finding is that the held-out loss does not choose the size, which Locatello et al. would predict.

Budget: seven fits of about 5 minutes and scoring, each its own capped run. Nothing over 10 minutes.

## 7. Result

**Fail: the factoring holds in two seeds of three, and the cube is lost in the third.** Seven fits of 4,000 steps in half precision (about 4 minutes each, 6.7 GB of the GPU at peak). Judged on the 100 fresh episodes; the validation episodes agree within 2 points on every line. Numbers in [results.json](results.json).

**The size.** The held-out loss was 1.075, 0.931, 0.958, 0.934 and 0.926 nats for 4, 6, 8, 12 and 16 coordinates, so the rule picked **6**. The loss does not fall smoothly (8 is worse than 6), and two fits of the same size differ by as much as the 0.02-nat margin, so the choice is within noise. It is still a sound one: at 4 the cube has no factor, and at 12 and 16 every variable has its own factor but the cube's change is spread thin (48–51% in its factor, stray movement 13–17%).

**Three seeds at 6 coordinates, fresh episodes.**

| | Seed 0 | Seed 1 | Seed 2 | Mean | Bar |
|---|---:|---:|---:|---:|---|
| Factors found | 5 | 5 | 4 | | the same in all three |
| Each variable its own factor | yes | yes | **no** | | two seeds of three |
| Reading through own factor: cube | 0.977 | 0.983 | **0.872** | **0.944** | 0.95 |
| Reading: drawer, window, lock, lock (range) | 0.973–0.991 | 0.978–0.988 | 0.977–0.995 | 0.976–0.991 | 0.95 |
| Share of change in own factor: cube | 79% | 89% | **52%** | **73%** | 80% for four of five |
| Share: drawer, window, lock, lock | 96, 93, 87, 83% | 95, 94, 92, 80% | 97, 92, 80, 86% | 96, 93, 86, 83% | |
| A factor's stray movement: cube | 5.0% | 5.7% | **110%** | **40%** | under 10% |
| Stray movement: the other four (range) | 3.2–7.8% | 2.3–5.2% | 3.0–11.4% | 3.8–6.6% | under 10% |
| Held-out loss (validation, no labels) | 0.931 | 0.954 | 0.989 | | |

Card 168's rule-built slots read the same pairs at 0.984 (cube) and 0.999–1.000; the encoder with random weights at 0.53–0.68, with 17–19% shares and 61–105% stray movement.

| Criterion | Verdict |
|---|---|
| 1. Each variable read through its own factor at 0.95 | **Fail** on the cube (0.944); the other four pass |
| 2(a). Each variable its own factor in two seeds of three | **Pass** |
| 2(b). 80% of change in own factor for four of five | **Pass** (the cube is the fifth, at 73%) |
| 2(c). Each factor's stray movement under 10% | **Fail** on the cube (40%); the other four pass |
| 3. The same number of factors in every seed | **Fail** (5, 5, 4) |

- **Seeds 0 and 1 meet every bar by themselves**, on episodes that neither the fits nor the choice of bars had seen. Read through its own factor, each variable is told within 2 points of the rule-built slots.
- **Seed 2 gave the drawer three coordinates and the cube none.** The cube's change is split between the window's coordinate (52%) and a lock's (32%), so "the cube's factor" there is the window's, which moves more when the cube is still than when it moves.
- **The cube is the weak variable in every fit of the card.** It never passes 89%, and it is the first to go when coordinates are too few (size 4), too many (12 and 16), or taken by another variable (seed 2). It is the one variable with more than one number to it, and no fit gave it two.
- **The held-out loss ranks the three seeds in the order of their factoring** (0.931, 0.954, 0.989). With three seeds this is a hint, not a finding. If it holds, the bad seed can be told from the good ones with no label.
- **Counting every fit of cards 173 and 174:** each variable had its own factor in six of eight (not at size 4, not in seed 2).
- **The lock factors near a press** move 10–14% of a flip in every seed, as section 4 found. The whole-state distance reads worse than the own factor everywhere (cube 0.75–0.94, drawer 0.88–0.91, window 0.80–0.84).
- Memory: fits peaked at 6.69–6.73 GB on the GPU, scoring at 0.85 GB, the feature build at 1.2 GB. The fitting process reported 12 GB of RAM at peak, which counts the mapped feature files; no run was stopped by its cap.

## 8. Decision

**Revise.** A state trained with no rule, touch, label or decoder gives a factor per scene variable in two seeds of three, read nearly as well as og5's rule-built slots, and holds on fresh episodes; but one seed in three loses the cube, so it cannot yet replace the rules. og5 stays. The next card should work on the cube and on telling a bad fit from a good one: whether the held-out loss picks the factored seed over more seeds, and whether the cube keeps a factor when the fit runs longer than four minutes or the cube's two table axes are both asked for. The gate's replication step was mis-specified by me and was set aside by the user's decision; it is recorded in section 5.
