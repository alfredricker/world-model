---
id: "173"
title: tight box, sparse change — card 172's loss with the temperature fixed, a cost on change between consecutive steps, a state size chosen on held-out episodes, and factors read from changes that last five steps
rung: 2
serves: [P7, P4, P1]
status: done
verdict: fail
arch_version: og5
date: 2026-11-27
---

# 173: tight box, sparse change

OGBench `visual-scene`. The design was given by the user on 2026-11-27 as a written proposal and approved in the same message ("card 173 is approved, run it as written"). This card writes that proposal down with exact thresholds; where the proposal said "e.g." its value is taken. It follows [card 172](../172-factors-from-sparse-change-over-time/card.md), which showed that a state trained only to pick the near moment of an episode reads every scene variable but does not put them on separate coordinates.

## 1. Question

Card 172 gave three likely causes for the missing factoring: more coordinates than the scene has variables, a learned temperature that shrank the state and undid the penalty on spare coordinates, and factors read from single steps that are mostly jitter. If all three are removed, do the scene's variables (cube, drawer, window, two locks) line up with separate factors of the state? Rung 2; Roadmap step 2. Serves P7 (units from continuity), P4 and P1.

## 2. What changes

One component: the objective that shapes the factoring, and how the state's size and its factors are obtained. The encoder, the frozen pose term `g`, the choice of moments and the stored features are card 172's. og5 is untouched.

```
172:  score(a, b) = −Σ_k |s_a[k] − s_b[k]| / τ + g(pose_a, pose_b)     τ learned; K = 32
      loss        = InfoNCE over 17 moments + 0.01 × Σ_k spread of s[k]
173:  the same score with τ = 0.5 fixed
      loss        = InfoNCE over 17 moments + 1.0 × mean of Σ_k |s_{t+1}[k] − s_t[k]|       (consecutive steps)
      K           chosen from {4, 6, 8, 12, 16}: the smallest whose held-out InfoNCE loss is within 0.02 nats of the lowest
      factors     card 172's rule, read from changes over 5 steps
```

- **Temperature fixed at 0.5, spread penalty dropped.** The state is `tanh(·)`, so each number lies in [−1, 1]. With the temperature fixed the state cannot shrink to make every distance small; to tell false moments apart it has to spread into the box. A box with a sum-of-absolute-differences distance is the condition under which a contrastive loss recovers the generating variables up to order and sign (Zimmermann et al. 2021, LITERATURE).
- **A cost on change between consecutive steps**, the sum of absolute changes per number, weight 1.0 (SlowVAE, Klindt et al. 2021, LITERATURE). In 68% of consecutive steps no scene variable changes (card 172, section 4), so in those steps the cost falls on jitter alone. A change spread over two numbers by a rotation costs √2 times what it costs on one.
- **The size of the state is chosen, not declared.** Five fits with K = 4, 6, 8, 12 and 16, the same seed and the same 4,000 steps. The held-out loss is the InfoNCE loss on 40,000 moments of the 100 validation episodes, drawn the same way for every fit. K* is the smallest K within 0.02 nats of the lowest loss. No label is used.
- **Factors are read from changes that last.** Card 172's rule and its four constants, applied to the change between moments 5 steps apart on 100 training episodes, no labels: a coordinate is unused if its standard deviation is under 0.02; it moves in a window if its change is over five times its median change and over 0.02; two coordinates are tied if the windows in which they move overlap by half or more; a factor is a connected group of tied, used coordinates. Over 5 steps, 66% of pairs differ in exactly one variable and 3.5% in two or more, against 30% and 1% over one step (card 172's `datacheck.json`). No variable differs in more than a third of 5-step pairs (cube 32.5%, slides 11–12%, locks 9%), so a coordinate's median change is still its change at rest.

Three things I expect to matter, stated before the run. None changes the design.

- **The new cost partly repeats a term the loss already has.** InfoNCE's pull on the true pair is its distance divided by the temperature: a cost of 2.0 per unit of change between moments up to 5 steps apart. The new term adds 1.0 on moments 1 step apart. Its effect is to shift the balance toward holding still, not to add a new kind of pressure.
- **The held-out loss may fall with every added coordinate.** A larger box holds more to tell moments apart by. If so, the rule picks 12 or 16 and the first of card 172's causes is not removed.
- **The 0.02-nat margin is compared across single fits.** The spread over seeds is measured only at K*, after the choice.

Not in this card: values in the units a skill needs, kinds, the hidden cube, the transition head, any use by the agent. Declared settings: temperature 0.5, weight 1.0, 4,000 steps, the ladder, the 0.02-nat margin, the 5-step window, and the factor rule's four constants.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Near moments differ in few of the scene's variables | holds, except nightfall | the same | mostly | holds | general; scene-wide change is a declared exception |
| The agent's pose is given and explains its own body in view | holds (C1) | holds | no body: `g` is empty | holds | general (C1) |
| No contact sense | none needed | none needed | none needed | none needed | general |
| A fixed camera: a query can learn where to look | no: the view moves | no | yes | fixed cameras only | declared, as in cards 119–172 |
| The scene's variables are few enough for a ladder of sizes to be fitted | no: open terrain needs a state per region | no | varies | a table top, yes | **declared limit**; the ladder costs one fit per size |
| Most single steps change nothing in the scene | holds while walking | holds | no: every move changes the grid | holds | declared; the cost on consecutive steps leans on it |
| The window is 5 steps of this world's clock | setting | setting | setting | setting | declared setting, from card 172's count |

## 3. Dependencies

- Card 172's features (`runs/172/`, 14.5 GB), encoder (`tools/card172/factors.py: Factors`), frozen pose term (`runs/172/g.pt`; alone it picks the true moment 53.9% of the time, loss 1.43 of 2.83 nats) and scoring (`tools/card172/score.py`).
- Card 172's trivial baselines and reference under the same scoring: best trivial baseline 0.56–0.67, card 168's slots 0.976–1.0, on the validation episodes.
- Zimmermann et al. 2021; SlowVAE (Klindt et al. 2021); Locatello et al. 2019 (no label-free choice of settings is known to pick the factored model, and seeds matter as much as methods). All in LITERATURE; no new source.

## 4. Data check

No new data. The counts this card leans on are card 172's (`runs/172/datacheck.json`, 450 training episodes): the share of pairs differing in none, one, or more variables at 1 and 5 steps, and each variable's share at 5 steps, quoted in section 2. Smoke test of the code: 150 steps in 11 seconds, so 4,000 steps take about 5 minutes.

## 5. Feasibility gate

Three fits of 4,000 steps, each its own capped run of about 5 minutes. Scored on the 100 validation episodes.

1. **Upper bound**, at K = 6 and K = 8: pairs chosen by the true scene state (the true moment shares the scene's state, twelve false ones differ in one variable where the episode has them), with **no pose term in the score**, the same fixed temperature and cost on change. Card 172's bound kept a pose term fitted to other pairs, which muddied it. The bound passes if either size reaches criterion 1 (0.95 for all five variables) and criterion 2(a) and 2(b) (each variable its own factor; 80% for at least four). If neither does, the encoder or the box cannot factor this scene and the card stops.
2. **Label-free short check**, K = 8, seed 0: every one of the five variables at least 10 points of AUROC above the best trivial baseline (so at least 0.660, 0.722, 0.701, 0.742 and 0.765 for cube, drawer, window and the two locks), and over 5-step windows the state's change where nothing in the scene differs under **20%** of its change where one variable does. Card 172's single-step figure (74.9%) is reported beside it.

Trivial baselines, the reference and random scoring are card 172's and are not rerun: the scoring of criterion 1 is unchanged.

Result of the gate: **failed on both steps. The ladder and the three seeds were not run and no criterion is judged.** Validation episodes; all numbers in [results.json](results.json).

| Gate step | Result | Verdict |
|---|---|---|
| Bound, K = 6: reading (0.95 for all five) | 0.993 to 1.000 | reached |
| Bound, K = 6: own factor each, 80% for four | six factors, each variable its own; 43, 54, 48, 88, 62% | **not reached** (one of five) |
| Bound, K = 8: reading | 0.987 to 1.000 | reached |
| Bound, K = 8: own factor each, 80% for four | seven factors, each variable its own; 51, 61, 51, 88, 84% | **not reached** (two of five) |
| Label-free, K = 8: each variable 10 points over the best trivial baseline | 0.910, 0.866, 0.827, 0.971, 0.968 against bars of 0.660 to 0.765 | pass |
| Label-free, K = 8: change where nothing differs under 20% of change where one variable does | 36.6% (single steps 57.2%; card 172's 74.9%) | **fail** |

- **The bound was not a bound for factoring, and the stop rule's inference was wrong.** The label-free fit factored better than both bounds (next section). In the bound a true pair shares the whole scene, so the loss's pull on true pairs never sees one variable change by itself; only the weaker cost on consecutive steps does. The sparse change that sorts variables onto coordinates is in the label-free pairs, half of which differ in one variable. The bound tests reading, as card 172's did. The gate fails all the same, on the second step.

## 6. Success criteria and prediction

Only if the gate passes: the ladder (K = 8 reuses the gate's fit), then three seeds at K* (seed 0 reuses the ladder's fit). The mean over seeds is judged and the spread reported. Truth is used for scoring only.

1. **Each variable can be read.** AUROC at least **0.95** for all five variables on the validation episodes, defined as in card 172. Card 141's test frames (the frame before a touch) are reported beside card 168's slots, which reach only 0.86–0.89 on the locks there; no bar is set on a set the reference does not pass.
2. **Each variable has its own factor, and the arm has none.** Over 5-step windows of the validation episodes: (a) each of the five variables is the best match of a different factor, in at least two seeds of three; (b) where exactly one variable differs, at least **80%** of the state's change is in that variable's factor, for at least four of the five; (c) where nothing differs, the state's change is under **10%** of its change where one variable does.

Reported, not judged: K*, the held-out loss at each size, the number of factors per seed, the single-step version of 2(c).

Prediction. The bound should pass criterion 1; I give it about two in five to pass criterion 2. For the label-free check I expect the readings to hold or improve on card 172's, and give about one in three that change with nothing differing falls under 20%: card 172 did not find what that jitter is, and this card removes its room to hide, not its cause. All criteria passing: about one in five.

Budget: gate 3 fits, main run 6 fits, each about 5 minutes in its own capped run. Nothing over 10 minutes.

## 7. Result

**Fail at the gate.** Three fits of 4,000 steps (about 5 minutes each) were made and scored; a diagnostic with no training followed (`tools/card173/diag173.py`). Everything below about the label-free state is one fit with one seed.

**The label-free fit found five factors, one per scene variable.** It emptied one of its eight coordinates and the rule grouped the other seven into five factors. Each variable's best factor is a different one.

| | Cube | Drawer | Window | Lock | Lock |
|---|---:|---:|---:|---:|---:|
| Share of the state's change in the variable's factor when only it differs (bar 80%) | 73.4% | 95.9% | 94.4% | 92.1% | 83.4% |
| The same, card 172 (32 coordinates, single steps) | 13.2% | 16.7% | 15.0% | 19.4% | 7.8% |
| Reading by the variable's own factor, validation (AUROC) | 0.983 | 0.981 | 0.970 | 0.990 | 0.983 |
| Card 168's slots, own slot, validation | 0.976 | 1.000 | 0.998 | 1.000 | 1.000 |
| Reading by own factor, card 141's test frames | 0.987 | 0.973 | 0.976 | 0.940 | 0.937 |
| Card 168's slots, own slot, test frames | 0.994 | 0.979 | 0.959 | 0.862 | 0.888 |
| Reading by the whole state, validation (the gate's and criterion 1's measure) | 0.910 | 0.866 | 0.827 | 0.971 | 0.968 |

- **Factoring, by this fit, is there.** Four of five variables pass the 80% bar that card 172's fit missed by 60 points and that neither bound reaches. A straight-line read-out from one factor alone recovers the drawer (R² 0.88), the window (0.95) and each lock (0.96, 0.89); from every other factor it recovers nothing (under 0.05).
- **Read through its own factor, a variable is told about as well as by card 168's rule-built slots**, and on the frames just before a touch better for the window and both locks. Which factor is whose is named by truth here, as it is for the slots.
- **The whole-state distance reads worse than card 172's did for the cube and drawer** (0.910 and 0.866 against 0.954 and 0.902), because of the next point.
- **The state still moves when nothing in the scene does: 36.6% of its change when a variable does** (bar 20%). Two thirds of that movement is in the two lock factors: 0.09 and 0.08 a window, where a real flip moves them 0.85 to 0.92. It is as large when the cube and slides are still to the millimetre (37.5%) as when they creep under a centimetre (32.8%), and it barely follows how far the arm moved (correlation 0.11). The pose explains under 5% of any factor. I do not know what moves the lock factors; the arm passing over the buttons is the obvious guess and was not tested.
- **The cube has one number, not two.** A straight-line read-out recovers its position along one axis of the table (R² 0.90) and not the other (below zero), and a fifth of its change sits in a lock's factor. The rule did not form the two-coordinate factor the proposal expected.
- **The spare coordinate emptied without a penalty.** With the temperature fixed, one coordinate's spread fell to zero and 21% of readings sit at the box's edge. The held-out loss is 0.944 nats (pose term alone 1.440, chance 2.833).
- **Not tested:** whether another seed gives the same five factors, and whether the held-out loss picks a size near 8. Both were the main run.

Of the three points I raised in section 2, the first held in a way I did not foresee (the true-pair term is what factors; it is what the bound lacked), and the other two were not reached.

## 8. Decision

**Revise.** The card fails its gate as written: change with nothing differing is 36.6% against a bar of 20%, and the bound I was to trust did not reach the factoring bar. og5 and its rules stay. But the one label-free fit is the first here to give a factor per scene variable with no rule, touch, label or decoder, read about as well as the rule-built slots. The next card should ask whether that holds: the ladder and three seeds as this card planned them, with two things set first from what was learned. Criterion 1 should read each variable through its own factor, as the slots are read, with the whole-state distance reported. And the gate's bound should be dropped for factoring or rebuilt with true pairs that differ in one variable. The movement in the lock factors needs its cause found before a bar is set on it.
