---
id: "172"
title: factors from sparse change over time — a scene state whose coordinates are found from how the senses change, with no touch signal, no slot count and no decoder
rung: 2
serves: [P7, P4, P1]
status: done
verdict: fail
arch_version: og5
date: 2026-11-27
---

# 172: factors from sparse change over time

OGBench `visual-scene`. Drafted at the user's request, 2026-11-27 ("draft the card that learns the factoring from sensory changes (not strictly touch)"; the same day: "we should work to remove the hand set rules as well"). Approved by the user the same day ("approved"), after it was revised after a review the user passed on: the pose term is fitted first and frozen, the choice of moments is fixed from a count of the data, the state is bounded, and the rule that reads factors off it is written down. Section 4's numbers come from stored truth and the arm's pose; no frame was read.

og5's scene model (`SceneWNet`, [card 168](../168-consequence-trained-scene-world-model/card.md)) reads five slots. What a slot is, and what its value is, are set by rules in `discover_slots`: k-means on where the hand made contact, the 48 patches a touch there changes most, the first SVD direction of their change as a slide's value, a Fisher contrast and 2-means as a lock's state. This card replaces that factoring with a learned one and judges it by itself, before anything reads it.

**This has been tried here four times.** Each failed for a reason the design has to answer:

| Card | Signal | What went wrong |
|---|---|---|
| [116.1](../116.1-slots-at-448/card.md) | Slots that redraw the present frame | The error is summed over patches: background kept, small objects skipped |
| [119](../119-variables-from-lasting-change/card.md) | Patches that change together are one variable | A slide changes different patches at each position, so it split into four groups covering only the strip it sweeps |
| [124](../124-tokens-from-consequences/card.md) | Tokens that predict the next frame, with a cost on how many tokens change | One token changing is cheaper than two: 65% of the cube went into the arm's token |
| [145](../145-a-state-that-changes-only-when-the-scene-does/card.md) | One gated vector that redraws every frame given the arm's pose | Variables were learned in order of picture area (drawer, window, locks); the cube never. The user's rule: no decoder as the training signal |

## 1. Question

If a frame is mapped to a short list of bounded numbers, trained only to tell which moment of an episode is near in time, with a change between near moments costed by its absolute size per number, do those numbers line up one to one with the scene's variables (cube, drawer, window, two locks), with the count found and not declared? Rung 2; Roadmap step 2 (conditions are each object's stable states between its boundaries). Serves P7 (units from continuity, not a supplied vocabulary), P4 and P1.

## 2. What changes

One component: what defines the coordinates of the scene state. No planner, skill or transition head changes; og5 is untouched.

```
og5:  contact points → k-means → places → 48 most-changed patches → SVD direction (slide), 2-means (lock)   [rules, from touches]
172:  frame t → frozen patch features, each with its grid position → K learned queries pool the patches
              → s_t = tanh(·), K numbers in [−1, 1]                                                          [one network]
      true pair:      (t, t′), t′ drawn within 5 steps of t (a step is 5 frames)
      false moments:  16 — the 12 moments 6 to 20 steps from t whose pose is nearest t's, and 4 from other episodes
      score(a, b)   = −Σ_k |s_a[k] − s_b[k]| / τ + g(pose_a, pose_b)       g fitted first by itself, then frozen
      loss          = pick t′ out of the 17 (InfoNCE)  +  0.01 × Σ_k spread of s[k] over the batch
```

- **The loss is over moments, not patches.** A variable counts by how often it makes two moments differ, not by its area. Under card 145's loss the cube carried 0.12 of the error and was never learned; here it is the variable that most often tells a false moment from the true one (section 4).
- **Absolute size per number, in a box.** With the state in a bounded box and the distance a sum of absolute differences, a contrastive loss recovers the generating variables up to order and sign (Zimmermann et al. 2021, checked in `papi`; LITERATURE). The theorem assumes a uniform spread and a matching form of change, which real play only approximates. `τ` is one learned number; it has a meaning because the state is bounded. Card 124's fault cannot occur in this form: two independent quantities do not fit in one number, so merging them is not cheaper.
- **Which moments.** Near moments must share most of the scene, and the false moments must differ from the true one in little, or the small variables earn nothing. Moments 6 to 20 steps away differ from the true one in exactly one variable 48% of the time, against 25% for moments drawn from the whole episode. Moments from other episodes differ in everything; four are kept so that readings agree across episodes.
- **The arm is given, not found, and my first argument for that was wrong.** I wrote that a trace of the arm in `s` "adds distance to every true pair and buys nothing". It buys a lot: near moments have near poses, and the pose alone picks the true moment 24% of the time (chance 5.9%). Three things now keep it out. The false moments are the ones nearest in pose. `g`, a small network that sees only the two poses (GOAL C1), is fitted alone on the same pairs and frozen, so `s` is trained only where the pose leaves doubt. And the spread penalty charges for every coordinate in use. Criterion 2(c) measures whether this worked.
- **No touch.** Frames are every fifth step of recorded play; the contact value is not read. In this data 39% of the drawer's steps of change and 53% of the window's happen with no contact.
- **The count is found, by a rule fixed here.** K is a ceiling (32). On training frames, with no labels:
  - a coordinate is *unused* if its standard deviation over frames is under 0.02;
  - a coordinate *moves* between two consecutive steps if it changes by more than five times its median change, and by more than 0.02 (two steps in three change nothing in the scene, so the median is the coordinate's noise);
  - two coordinates are *tied* if the steps in which they move overlap by half or more (intersection over union);
  - a *factor* is a connected group of tied, used coordinates.
- **Where a thing is.** As in card 168, each patch carries its grid position and each query's pooled centre is appended, so a query that finds the cube by its look can report where it is.
- **A frame at a time.** `s` is a function of one frame, so a goal frame is read the same way. The cube shut in the drawer is in no frame and is outside this card.

In the brain's terms: representations that hold still while the view changes are learned from what follows what in time (Földiák 1991; Li and DiCarlo 2008; slow feature analysis, Wiskott and Sejnowski 2002), and sensory change the body causes itself is predicted from the motor command and cancelled (von Holst and Mittelstaedt 1950). These are analogies, not evidence.

Not in this card: values in the units a skill needs, kinds, the hidden cube, the transition head, any use by the agent. Declared settings: the 5-step window, the 6-to-20-step ring, 16 false moments, the penalty's weight, and the four constants of the factor rule.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Near moments differ in few of the scene's variables | holds, except nightfall | the same | mostly | holds | general; scene-wide change is a declared exception |
| The agent's pose is given and explains its own body in view | holds (C1) | holds | no body: `g` is empty | holds | general (C1) |
| No contact sense | none needed | none needed | none needed | none needed | **general**; removes card 120's assumption |
| A fixed camera: a query can learn where to look | no: the view moves, the pose must place it | no | yes | fixed cameras only | declared, as in cards 119–168 |
| At most K numbers describe the scene | too few for open terrain | too few | varies | enough for a table top | declared ceiling; criterion 3 tests that the count found does not follow it |
| The window and ring are in steps of this world's clock | settings | settings | settings | settings | declared settings, chosen from section 4's count |

## 3. Dependencies

- Frozen DINOv2 patch features cut to 64 components keep every variable: a labelled probe reads the cube to 0.19 cm, slides to 0.11–0.13 cm and locks at 98% (cards 141, 145).
- [Card 141](../141-representation-test-bed/card.md)'s test bed and [card 145](../145-a-state-that-changes-only-when-the-scene-does/card.md)'s per-step data builder and scoring (`tools/card145/build.py`, `shared.py`).
- Card 168's attention pooling over patches (the encoder's form, kept at 2 of 3).
- Established methods, all in LITERATURE: Zimmermann et al. 2021 (entered and checked for this card); SlowVAE (Klindt et al. 2021: an absolute-value prior on change identifies factors up to permutation, with "no help for factors that never change within a pair"); Locatello et al. 2020 (pairs that share all but a few factors identify them); SlotContrast (a contrastive loss across time on frozen DINOv2 features, no labels).
- Locatello et al. 2019 (LITERATURE): seeds matter as much as methods. So: fixed settings, three seeds, no choice of seed or setting by the test.

## 4. Data check

Counted 2026-11-27 from stored truth and the pose, 450 training episodes of 201 steps (`tools/card172/datacheck.py`, `runs/172/datacheck.json`). A variable "differs" if the cube or a slide is over 1 cm away, or a lock is in the other state.

| Gap between two moments | No variable differs | One | Two or more |
|---|---:|---:|---:|
| 1 step | 68% | 30% | 1% |
| 5 steps | 31% | 66% | 3% |
| 10 steps | 10% | 71% | 20% |
| 20 steps | 5% | 42% | 53% |

How the false moments are chosen, with the true one within 5 steps (it shares the whole scene with `t` in 49% of pairs):

| False moments drawn from | Pose alone picks the true one (chance 5.9%) | Differ from the true one in exactly one variable | Of those: cube / drawer / window / lock / lock |
|---|---:|---:|---|
| Anywhere in the episode | 28.5% | 25% | 41 / 12 / 17 / 16 / 14% |
| 6 to 20 steps away | 23.5% | 49% | 39 / 12 / 15 / 17 / 17% |
| 6 to 20 steps away, nearest in pose (this card) | 18.6% | 48% | 45 / 11 / 14 / 15 / 15% |

- **The locks are not starved inside an episode.** A lock flips in only 1.7% of steps, but its state lasts, and each lock is the telling variable as often as a slide is.
- **The pose is a real cue and choosing by pose only halves its lead over chance.** A window of 20 steps brings it to chance (6.0%) but then only 20% of true pairs share the scene. Hence `g` fitted first and frozen.
- **The arm moves in 99.7% of steps** (card 145).

Card 145's stored features were deleted. They are rebuilt with card 168's components (`runs/132/pca.pt`), so that card 141's test frames and og5 read the same input: 450 training and 100 validation episodes, about 12 GB, written with pauses.

## 5. Feasibility gate

Before the main run, each step under 10 minutes and capped.

- **`g` alone**, reported first: how often the frozen pose term picks the true moment. If it is above 50%, the pose settles most pairs, `s` has too little left to learn from, and the ring is widened to 10–40 steps before anything else is run (the one change of design this card may make).
- **Upper bound, shows the bars can be reached:** the same encoder and loss with pairs chosen by the true state (a true pair shares the scene's state, a false one differs in exactly one variable). Truth is used here and in scoring only. It must reach criterion 1's bars for all five variables and criterion 2's for at least four. This bound gives every variable equal weight, so it tests the encoder and not whether the locks get enough signal; the short fit tests that.
- **Reference built with rules:** card 168's `SceneWNet` slots under the same scoring.
- **Trivial baselines:** (i) the pose alone; (ii) the 32 leading principal components of the frame's mean patch vector, nothing trained; (iii) the encoder with random weights.
- **Short fit, 5 minutes, one seed, the go or no-go:** the cube and at least one lock are told apart at 10 points of AUROC or more above the best trivial baseline, and in steps where only the arm moves the state changes by under a fifth of what it changes when a variable does.
- **Random scoring under the matching rule** is measured first (LESSONS, card 120's note).

Result of the gate, before the main run: **failed. The six fits of the main run were not made and no criterion is judged.** Numbers on the 100 validation episodes unless said; all in [results.json](results.json).

| Gate step | Result | Verdict |
|---|---|---|
| `g` alone picks the true moment | 53.9% (chance 5.9%) | over the 50% line; see below |
| Upper bound: criterion 1 for all five | 0.96 to 0.999 on validation; 0.69 to 0.84 on the test frames | reached on one set of two |
| Upper bound: criterion 2 for four | each variable's change is 9–27% in its best coordinate; 30 factors | **not reached**: the bars were not shown to be passable |
| Short fit: cube and a lock 10 points over the best trivial baseline | cube 0.954, locks 0.899 and 0.918, against 0.56–0.67 | pass |
| Short fit: change in arm-only steps under a fifth of change when a variable does | 74.9% | **fail** |

- **The contingency for `g` was wrong and I did not apply it.** The card said to widen the ring to 10–40 steps if `g` passed 50%. Measured before any fit: that raises `g` to 61.9%, and three other choices of window and ring give 51–70%. A network that sees both poses finds nearness in time whatever the moments are, and the simple nearest-pose count in section 4 (18.6%) understated it. `g` still leaves half the loss (1.43 of 2.83 nats), so the design was left as declared. This is a departure from the card as approved.
- **Fixed before any fit was scored:** criterion 1 uses the distance over the whole state, on pairs up to 40 steps apart, and is read on each set separately.

## 6. Success criteria and prediction

100 validation episodes and card 141's 300 test episodes; three seeds, the mean judged and the spread reported; truth for scoring only. The number of training steps is fixed from the gate's throughput and is the same for every fit.

1. **Each variable can be read.** For pairs of frames that differ in one variable, the distance in the state separates them from pairs that do not differ: AUROC at least **0.95** for the drawer and the window differing by 4–10 cm, for each lock, and for the cube on the table differing by over 4 cm. These are card 145's bars; its decoder-trained state reached drawer 0.999, window 0.958, the drawer's lock 0.991, the window's lock nothing and the cube 0.49. Card 168's slots are scored beside it.
2. **Each variable has its own factor, and the arm has none.** Factors are read by section 2's rule. (a) Each of the five variables is the best match of a different factor. (b) When exactly one variable changes, at least **80%** of the state's change is in that variable's factor, for at least four of the five (card 145's precision was 0.27–0.35). (c) In steps where only the arm moves, the state's change is under **10%** of its change when a variable does.
3. **The count is found, not set.** With the ceiling at 32 and at 64 the number of factors is the same, and criteria 1 and 2 stay within 3 points.

Prediction. Honest odds are under one in two for all three. Criterion 1 is the likeliest: the drawer and window should pass, the cube is the point of the card and I expect it to clear 0.90, and the locks are the risk, being two or three patches that the arm covers while pressing. Criterion 2(b) is the hardest; I expect three or four variables near 60–80%, with the cube's position spread over more coordinates than it needs. Criterion 3 depends on the spread penalty emptying the spare coordinates, which nothing here has shown.

Budget (as drafted): rebuilding the features about 15 minutes in paced runs under 10 minutes each; the gate four fits of 5 minutes; the main run six fits of 10 minutes (two ceilings, three seeds), each its own capped run. Nothing over 10 minutes.

## 7. Result

**Fail at the gate.** One label-free fit of 5 minutes (4,038 steps, ceiling 32, one seed), one upper-bound fit, three baselines and the reference were scored; nothing else was trained. The features took 6 minutes to rebuild and are kept in `runs/172/` (14.5 GB) for the next card.

How well a pair that differs in one variable is told from a pair that differs in none (AUROC, distance over the whole state; bar 0.95):

| | Cube | Drawer | Window | Lock | Lock |
|---|---:|---:|---:|---:|---:|
| **Validation episodes** | | | | | |
| Label-free fit, 5 minutes | 0.954 | 0.902 | 0.826 | 0.899 | 0.918 |
| Upper bound (pairs chosen by truth) | 0.982 | 0.980 | 0.963 | 0.999 | 0.999 |
| Best trivial baseline (pose, principal components, random encoder) | 0.560 | 0.622 | 0.601 | 0.642 | 0.665 |
| Card 168's slots, each variable on its own slot | 0.976 | 1.000 | 0.998 | 1.000 | 1.000 |
| **Card 141's test frames (the frame before a touch)** | | | | | |
| Label-free fit, 5 minutes | 0.944 | 0.854 | 0.790 | 0.746 | 0.802 |
| Upper bound | 0.840 | 0.688 | 0.716 | 0.797 | 0.825 |
| Card 168's slots | 0.994 | 0.979 | 0.959 | 0.862 | 0.888 |

| Criterion | Verdict |
|---|---|
| 1. Each variable can be read (0.95) | **Not judged.** In the short fit the cube reaches the bar on validation; the others are at 0.83–0.92. |
| 2. Each variable has its own factor, the arm none | **Not judged, and not shown to be passable.** All 32 coordinates are in use and none are tied. A variable's change is 8–19% in its best coordinate (bar 80%); the state changes 75% as much in arm-only steps as when a variable changes (bar 10%). |
| 3. The count is found | **Not judged.** No coordinate emptied. |

What the runs show.

- **A loss over moments learns what a loss over patches skipped.** With no decoder, no touch and no label, five minutes of training gives a state from which a straight-line read-out recovers the drawer (R² 0.90), the window (0.95), both locks (0.92, 0.90) and the cube's position on the table (0.83 one way, 0.49 the other), on held-out episodes. The cube is the best-told variable (0.954). Card 145's state, trained for an hour to redraw every patch, scored 0.49 on the cube, which is chance.
- **The arm's pose is close to absent from the state.** The pose explains 0.7% of the state's variance, and the state predicts no pose number better than R² 0.31. The frozen pose term did that part of its work.
- **The factoring did not appear, and the bound says this form cannot give it.** The state uses about 12 directions spread over all 32 coordinates. Even with pairs chosen by the true state, a variable's change is 9–27% in its best coordinate. Unmixing the state's step-to-step changes into 8, 12 or 16 independent directions does not find the variables either (13–46% for the label-free fit, 16–55% for the bound; `tools/card172/rotate.py`).
- **Why, as far as I can tell.** Three things, none of them tested apart:
  - The theorem this card leans on needs as many coordinates as there are variables. With 32 for about seven, a variable can be copied or spread over many coordinates at no cost.
  - The spread penalty that was meant to empty spare coordinates was undone by the learned `τ`: `τ` fell to 0.20 and every coordinate shrank to a spread of 0.03–0.15, so the box was never touched. The review's third point was right and my fix did not hold.
  - The rule reads factors from single steps, and a single step's change is mostly jitter: a lock's flip moves the state only 1.4–1.6 times as much as a step in which nothing in the scene changes. I do not know what the jitter is. It is not the pose in any straight-line sense.
- **Frames just before a touch are read worse than frames in general** (locks 0.75–0.80 against 0.90–0.92), for the bound too. The arm is at the object in those frames. Card 168's slots are also weakest there on the locks (0.86–0.89).
- **Card 168's rule-built slots read every variable better than this state does** (0.96–1.00 against 0.75–0.95). The label-free state is not a replacement yet.

## 8. Decision

**Revise.** The card asked whether the scene's variables would line up on the coordinates of a state trained on nearness in time, and they did not: the gate's own upper bound could not do it, so the main run was not made. What it did show is the half that cards 116.1 to 145 could not get: a state trained by a loss over moments, with no decoder, no contact signal and no label, holds every variable of the scene after five minutes, the cube first, with the arm left out. Reading is there; factoring is not. og5 and its rules stay. The next card should keep this loss and change how factors are obtained. The candidates I see: a state no larger than the scene needs, with its size chosen by the loss on held-out episodes and `τ` fixed, so that the theorem's condition holds; and factors read from changes that last over a window of steps, as in card 119, not from single steps. Before that, the rule for when `g` settles too much, and the choice of whole-state distance, need to be set from data and not from my guess.
