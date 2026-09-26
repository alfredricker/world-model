---
id: "001"
title: architecture selection
rung: 0
serves: [P12, P6, P19]
status: draft   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-26
---

# 001: architecture selection

## 1. Question

Which architecture best predicts how close a shown goal condition is, from
each state and after each action, on the rung-1 goal-conditioned fork? This
is the first part of GOAL.md's main insight: every later part (inferring a
goal's conditions, pursuing them as subgoals) is built on this estimate.
This is the one-time selection screen in CHARTER.md, run before rung 1. The
winner becomes arch_version 1. It serves the predictive side of P12 and P6
under C6; P19 is left to rung 1, because this screen balances the data
(section 4).

## 2. What changes

There is no model (arch_version 0). Every candidate has three parts on one
learned state (C2): an **encoder** from a 42×42×3 frame to a latent z; a
**transition** T(z, a); and a **reachability head** d(z, g) ≥ 0, the
predicted fewest steps from z to goal g, built as a quasimetric so one-way
changes cost more than their reverse. An action is scored by d(T(z, a), g),
lower being better. This is QRL's structure (`2304_01203`). The hub H is
the simplest version; each other candidate changes one part of it.

| | Candidate | Changes from H | Question it answers | Source (papi) |
|---|---|---|---|---|
| H | Hub | — | Baseline | `2304_01203`, `2208_08133`, `2402_15567`, `ogbench`, `leworldmodel` |
| A | + reconstruction | Adds a pixel decoder loss on z | Does explaining pixels help represent conditions? | `dreamerv3` |
| C | Subgoal head | Adds a head that proposes a latent subgoal on the way to g; actions are scored by d(T(z, a), subgoal) | Does an explicit subgoal level help, as the main insight's hierarchy claims? | `hiql` |
| D | Factored state, partial goals | Grid of cells + one vector; T moves cells and writes sparse events; goals weight the factors their examples agree on | Do conditions need a factored state? | `latent-actions`, `schema-networks-zero-shot-transfer-with-a-generative-causal`, `1711_00937`, `disco-rl` |
| E | No transition | Q(z, a, g) replaces d(T(z, a), g) | Is an explicit model of consequences needed? | `universal-value-function-approximators`, `hiql`, `ogbench` |

Held equal: the encoder trunk (3 convolution layers), at most 2M parameters
(measured), the same data and number of updates (fixed by a throughput
profile before approval), 2 seeds each. The appendix gives each one's losses
and main risk.

**Goals** (revised after the gate, user decision 2026-09-26): a goal is a
condition supplied as a task, as C1 allows until P20: 4 example frames
where it holds, from other training episodes, with a success signal from
the evaluator's labels. The model is never told which other conditions
lead to it (key before door); that is what it must learn. At test the 4
examples come from other development worlds. Hindsight goals (single later
frames) are kept for the state-to-state distance. Discovering conditions
without supplied goals is deferred to a later card (appendix).
**Diagnostics** (appendix) include two condition diagnostics for rung 2 and
card 002: the condition gap on matched state pairs, and the one-way gap.
**Round 2** (CHARTER): at most three combinations justified by the
diagnostics, for example QRL's transition loss in the learned quasimetric.

## 3. Dependencies

- **Chained-rooms port:** passed 2026-09-26 (9 tests; hashes match the old
  repo).
- **Fork evaluator:** `src/worldmodel/envs/rooms_goals.py`, 16 tests pass
  (rules match the real environment step by step, hand-counted distances,
  state setting round-trips, shuffled rankings score at chance). Distances
  ignore the 512-step truncation.
- **Matched condition pairs** for the condition gap: a pair builder with
  its own tests, before approval.
- **Methods:** every part is from a paper in papi. Untested anywhere:
  expectile regression with a quasimetric head (nearest: QRL's MountainCar
  table), D's hindsight goal sets, and C's subgoals chosen from random-action
  data (HIQL used offline datasets of directed behaviour). The screen tests
  all three.

## 4. Data check

Training data: the ported collection, 4000 episodes, 1.7M transitions,
uniform opaque actions repeated for geometric lengths. Interaction counts
in the old collection with the same seeds: key pickup 4052, unlock 307,
other door opens about 1300, box open 823, switch press 970.

**Balanced sampling (declared, screen only):** half of each batch is
anchored just before an interaction, chosen with evaluator labels the
learner never sees. Label-free sampling is card 002.

**Supplied goals (declared):** per training frame, the evaluator's labels
say which of 7 conditions hold (a red, green or blue key held; a red,
green, blue or grey door open) and whether they are in view. Example
pools are the frames where a condition holds and is visible.

**Evaluation states.** Random walks almost never reach the rare
situations (8 distinct unlock-decisive states in 120 development episodes),
so fork states are drawn from every reachable state of 120 development
worlds (up to 826k each, searched completely), stratified by goal, best
action and object in front (`runs/sampled_dev`). Scored strata:

| Stratum (goal ← best action at object) | Chosen | Worlds |
|---|---|---|
| door open ← toggle locked door holding its key | 300 | 120 |
| door open ← pick up the matching key | 300 | 120 |
| door open ← open the box holding the key | 300 | 68 |
| door open ← toggle a closed plain door | 300 | 120 |
| key held ← pick up the key | 300 | 120 |
| key held ← open its box | 300 | 88 |
| movement-decisive, both goal kinds | 2000 | 120 |

Reported, not scored: switches (rung 2), timed doors, picking up a box to
clear a path. Goal example pools: 135–237 frames per goal.

## 5. Feasibility gate

- **Upper bound:** the hub on the egocentric simulator view (MiniGrid's
  symbolic encoding of the same 7×7 view the frames show, plus the carried
  object) instead of pixels. A whole-world grid was tried first and is not
  an upper bound: a step changes one cell and T learned nothing. It must reach
  0.9 top-1 on the scored interaction strata; its rank correlation with
  true steps-to-goal over fork states, called ρ_U, is written here and sets
  criterion 2's floor. The same model scored with
  pooled 4-example goals from other worlds must stay within 0.1 of it with
  exact single-state goals, or goal pooling is revised first.
- **Trivial baselines:** random ranking (chance per fork from its ties);
  a fixed action preference; the goal-swapped control.

Result of the gate, before the main run: **not passed** (2026-09-26,
seed 0, 30k updates each; `runs/gate_try1`–`3`, `runs/profile_frames`).
Chance is 0.2 on the scored strata; always-toggle scores 0.667.

| Run | Top-1 pooled | Top-1 exact | Top-1 true successors | Movement exact | Rank corr. exact / pooled |
|---|---|---|---|---|---|
| Simulator state (allocentric grid) | 0.30 | 0.36 | 0.59 | 0.21 | 0.78 / 0.01 |
| + QRL transition loss, local penalty | 0.18 | 0.16 | 0.57 | 0.29 | 0.69 / 0.04 |
| Frames (hub H as specified) | 0.21 | 0.48 | 0.63 | 0.54 | 0.57 / 0.05 |

Three findings. (1) Goals pooled from other worlds carry almost no
signal in any run: trained only on single-state goals, the head measures
distance to states, not to conditions. (2) On the allocentric grid, T
learns nothing (predicting "no change" is as good, shuffle ratio 1.0), so
it is not an upper bound; on frames T learns (shuffle ratio 50–200).
(3) QRL's transition loss without QRL's push-up objective shrinks
one-step distances to about 0.02 and ties all actions. The one-way gap on
frames: box open 5.9, pickup 2.5, unlock 1.4, forward move 1.6 (a move
takes more steps to undo when facing matters), plain toggles 0.1.

Second round (egocentric simulator view; `runs/gate_ego*`, `gate_cond_*`):

| Run | Top-1 pooled (swapped) | Top-1 exact | Movement pooled | Rank corr. pooled |
|---|---|---|---|---|
| Egocentric view, hub as specified | 0.31 (–) | 0.54 | 0.46 | 0.05 |
| + label-free goal sets, 100k updates | 0.29 (0.18) | 0.59 | 0.49 | 0.03 |
| + supplied condition goals, asym rule | 0.62 (0.42) | 0.60 | 0.43 | 0.19 |
| + supplied condition goals, pooled rule | 0.65 (0.44) | 0.58 | 0.44 | 0.18 |

Supplied goals make the goal matter (0.2 above the swapped control), but
only for one-step conditions: key held ← pick up 0.88; door open ← toggle
the locked door 0.66, the plain door 0.70; the two-step chains (door ←
pick up its key; anything through a box) stay near 0.5, the swapped
level. Holding the key is predicted closer to "door open" in 74–80% of
matched pairs, but by about 0.1 step against a true 7. Movement stays
below 0.55 even on true successors: the head's one-step precision is a
separate limit. Curves were flat from 10k to 30k updates.

## 6. Success criteria and prediction

**Metrics:** (a) mean top-1 over the six scored interaction strata (the
best-ranked action is among the truly best); (b) Spearman rank correlation
between predicted d(z, g) and true steps-to-goal over all fork states and
their goals, unreachable pairs ranked last. Both are means of 2 seeds.

| # | Criterion | Threshold | Compared against | Why |
|---|---|---|---|---|
| 1 | The screen is informative | Best candidate ≥ 0.3 above the goal-swapped control | Goal-swapped control | Otherwise no candidate has the capability; the list is revised |
| 2 | Winner | Highest interaction top-1 among candidates with rank correlation ≥ 0.8 × ρ_U, movement-decisive top-1 ≥ 0.8 and no-effect interactions ranked best on ≤ 0.1 of forks | The other four; the upper bound | Choosing well at interactions is the target; knowing how close the goal is from every state is what later rungs build on; the rest guard against movement loss and false changes |
| 3 | Tie-break | Within 0.05: higher rank correlation, then fewer priors | Tied candidates | Prefer the one that knows better how soon, and assumes less |

**Prediction.** H does better on key goals than door goals, where two
conditions chain. D beats H on door goals if its transport learns. E is
close to H on top-1 (OGBench's Q-function expectile method:
cube-single-noisy 99) but has lower rank correlation, since it has no
separate distance between states. A adds little. C matches H on key goals
and beats it on door goals behind a box, the longest chain (HIQL's
hierarchy helped most at long horizons), if subgoals can be chosen from
random-action data. On the condition gap, D and C separate the key
condition from irrelevant differences better than H.

**Budget.** Gate: 1 run. Screen: 5 candidates × 2 seeds = 10 runs; round 2
at most 6 more. Run length is set by the throughput profile to what gives
an informative result (user, 2026-09-26), not capped at 10 minutes; runs
over 30 minutes are handed to the user as commands.

## 7. Result

## 8. Decision

---

## Appendix: candidates, goals and diagnostics

Sizes are starting points, fitted under the 2M cap.

**H, hub.** z: 192-d (pooled trunk, projector with BatchNorm). T: action
embedding through zero-initialised AdaLN in a small MLP; loss: squared
error to the encoded next frame plus SIGReg on every encoding (LeWM). d:
MRN construction on (z, g) in one space, a quasimetric by its Proposition 1
(IQE, `2211_15120`, is the better-evidenced alternative). d is trained
action-free by lower-expectile regression toward 1 + d_target(z_next, g),
with an EMA target network (HILP; OGBench's GCIVL held up better than QRL on
noisy and pixel data), on hindsight and far goals. Risk: SIGReg
under-spreads on low-diversity frames (LeWM lost to PLDM on TwoRoom).

**A, + reconstruction.** H plus a convolutional decoder from z with a pixel
loss. Risk: a changed door or carried key is a few pixels of loss.

**C, subgoal head.** H plus a head s(z, g) that outputs a latent of the
same size as z: a subgoal on the way to g. Actions are scored by
d(T(z, a), s(z, g)) instead of d(T(z, a), g). s is trained as HIQL's
high-level policy: regress toward the encoding of the state k = 8 steps
later (stop-gradient), weighted by exp(β · (d(z, g) − d(z_{t+k}, g))),
capped, so futures that brought the goal closer count most. HIQL used
k = 3 on pixel Procgen Maze and 25–50 elsewhere; its goal representation
φ([g, s]) is not taken, since z is already shared (C2). Risks: in
random-action data few k-step futures move toward g, so the subgoal
regresses to an average future; the subgoal can be too close for d to
separate actions.

**D, factored state with partial goals.** z: 8×8 grid of 24-d cells plus
one 32-d vector (where a carried object can live). T: next = move_a(z) +
gate ⊙ write(e); move_a is a learned action-conditioned transport of cells;
e is one code per cell and one for the vector from a 32-entry codebook,
code 0 meaning "no change" (VQ-VAE), predicted by a prior p(e | z, a) and
trained against a posterior that sees the next frame, with a penalty on the
rate of non-null codes. Goals: sets of 4 frames spread far apart in a later
stretch of the same trajectory, so that what they share is what lasted
rather than room or heading; per-factor weights from how much the examples
agree (DisCo RL's precision), applied inside the head so it stays a
quasimetric. Same SIGReg and d training as H. Risks: a poor transport
floods the event codes; codes collapse to "always null"; agreement weights
from 4 examples are noisy (DisCo used 30–50).

**E, no transition.** Q(z, a, g), steps-to-goal after action a, trained by
Q-learning toward 1 + min over a' of Q_target at the next state (QRL's
MountainCar baseline); z is trained only through Q. Risk: rare interactions
give Q little signal without a transition model to share across goals.

**Goals.** A condition goal is 4 example frames where it holds; its
target is 0 where it holds, else min(steps until it next holds in the
episode, 1 + d_target(z', G)). State goals are the next frame (15%), a
later frame of the same trajectory (45%; HER's "future" relabelling,
`1707_01495`) or any frame (40%; HILP, LEXA). Label-free goal sets (4
frames spread over a later stretch of the same episode) were tried at the
gate and left pooled goals at chance even after 100k updates
(`runs/gate_ego3`); discovering conditions is deferred. At test a goal is
4 example frames from other worlds. The distance to a set of states is the minimum
over its members (MRN, QRL), so the goal side is pooled by a coordinate-wise
maximum on the head's asymmetric features: this is a lower bound on the
distance to every example, and coordinates on which the examples disagree
stop counting. Only D also trains on goal sets, which is part of what it
tests.

**Diagnostics** per run: score per stratum; true versus shuffled action;
latent spread (collapse); for D, event codes firing on interactions versus
movement; for C, the rank correlation of d(subgoal, g) with true steps.
Two condition diagnostics, reported for rung 2 and card 002, not scored:
- **Condition gap:** pairs of states the evaluator sets up identically
  except for one condition (holding the key, the box open). The predicted
  gap in distance to the goal is compared with the true gap, and with pairs
  that differ in something irrelevant (another room's door).
- **One-way gap:** d(z', z) − d(z, z') across a transition z → z', for
  pickups, unlocks and box opens (which cannot be undone; there is no drop
  action) against plain door toggles and moves (which can). A reachability
  measure can only single out conditions that cannot be undone: a change
  undone in one step moves every distance by at most 1.
