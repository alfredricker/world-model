---
id: "001"
title: architecture selection
rung: 0
serves: [P6, P12, P19]
status: draft   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-25
---

# 001: architecture selection

## 1. Question

Which architecture best predicts how each action changes how soon a shown
goal condition can be reached, on the rung-1 goal-conditioned fork? This is
the one-time selection screen in CHARTER.md, run before rung 1. The winner
becomes arch_version 1. It serves P6 and the predictive side of P12 under
C6; P19 is left to rung 1, because this screen balances the data (section 4).

## 2. What changes

There is no model (arch_version 0). Every candidate has the same three
parts, all on one learned state (C2):

- an **encoder** from a 42×42×3 frame to a latent z; a goal is shown as a
  set of example frames, encoded by the same encoder and pooled into g;
- a **transition** T(z, a) giving the latent after action a;
- a **reachability head** d(z, g) ≥ 0, the predicted fewest steps from z to
  the goal; directed (a quasimetric), so one-way changes such as picking up
  a key cost more than their reverse.

An action is scored by d(T(z, a), g): the lower, the better. The hub,
candidate H, is the simplest version; each other candidate changes exactly
one part of it, so its score relative to H diagnoses that part.

| | Candidate | Changes from H | Question it answers | Source (papi) |
|---|---|---|---|---|
| H | Hub | — | Baseline: latent prediction + SIGReg for T; d learned action-free by expectile regression on hindsight goals, quasimetric head | `leworldmodel`, `2402_15567`, `2304_01203`, `2208_08133` |
| A | + pixel reconstruction | Adds a decoder loss on z | Does explaining pixels help represent conditions? | `dreamerv3`, `1811_04551` |
| B | Contrastive transition | InfoNCE replaces squared error + SIGReg for T | Does discrimination beat regression for consequences? | `1911_12247`, `1807_03748` |
| D | Factored state, partial goals | z is a grid of cells plus one non-spatial vector; T moves cells for turns and forward and writes sparse discrete events; g keeps only the factors the goal examples agree on, and d ignores the rest | Do conditions need a factored state in which a goal constrains only part of it? | `latent-actions`, `schema-networks-zero-shot-transfer-with-a-generative-causal`, `1711_00937` |
| E | No transition | A direct head Q(z, a, g) replaces d(T(z, a), g) | Is an explicit model of consequences needed at all, or is a goal-conditioned value enough? | `universal-value-function-approximators`, `hiql` |

The appendix gives each candidate's sizes, losses and main risk. Held equal
across all five: the encoder trunk (3 convolution layers), at most 2M
parameters (measured), the same data and number of updates (fixed by a
throughput profile before approval), 2 seeds each.

**Goals in training, without labels.** A training goal is a set of 4
frames drawn from a later stretch of the same trajectory (hindsight). What
the frames share is what lasted, such as a carried key or an opened door;
where the agent stands and looks varies. This teaches goals as partial
conditions without ever naming one.

**Round 2** (CHARTER): at most three combinations of round-1 parts, each
justified by the diagnostics below, for example D's factored state with B's
transition loss. Written into this card before it runs.

**Diagnostics** logged for every run: the score on forks decided by
movement versus by interactions (a representation problem shows on
interactions first); the rank correlation of d with true steps-to-goal;
action sensitivity (true versus shuffled action); latent spread (collapse);
for D, how often event codes fire on real interactions versus on movement.

## 3. Dependencies

- **Chained-rooms port, collection and fork probes:** must pass its tests
  and the byte-identical collection check. Not yet run (`src/` is empty).
- **Goal-conditioned fork evaluator (new):** breadth-first search over a
  copy of the simulator for the fewest steps to each goal condition after
  each action; builds goal example sets from other development episodes.
  Unit tests first: search distances match hand-counted cases; a fork where
  pickup is the only way to the goal ranks pickup first; a shuffled ranking
  scores at chance. Timed doors make the state time-dependent; goals whose
  shortest path passes a timed door are reported, not scored.
- **Methods:** every mechanism is from a paper in papi (table above). D and
  H combine established parts in a new way, and that combination is what
  this screen tests; rule 7 is met by the parts.

## 4. Data check

Training data: the ported collection, 4000 episodes, 1.7M transitions,
uniform opaque actions repeated for geometric lengths. Interaction counts
in the old collection with the same seeds: key pickup 4052, unlock 307,
other door opens about 1300, box open 823, switch press 970.

**Balanced sampling (declared, screen only).** Half of each batch is
anchored within a few steps before an interaction, chosen with the
evaluator's labels; the learner never sees them. Without this, every
candidate would fail on interaction-decisive forks and the screen could not
separate them. Learning from rare events without labels is card 002.

Evaluation: development ordinary-start probes (120 episodes), goals from
{holding key of colour c, door of colour c open, box open}. The number of
interaction-decisive and movement-decisive forks per goal type is measured
once the evaluator exists and written here before approval. Switch doors
are rung 2: goals that need a switch are reported, not scored.

## 5. Feasibility gate

- **Upper bound:** the hub's heads trained on the evaluator's simulator
  state (egocentric tile grid, carried object, door states) instead of
  frames. It must reach 0.9 top-1 on interaction-decisive forks, or the
  test is revised before any candidate runs.
- **Trivial baselines:** random ranking (chance per fork, from its ties);
  a fixed action preference (the most often best action overall); the
  goal-swapped control (each candidate shown a different goal's examples).

Result of the gate, before the main run:

## 6. Success criteria and prediction

**Metric:** top-1 on interaction-decisive forks: the candidate's
best-ranked action is among the truly best. Mean of 2 seeds.

| # | Criterion | Threshold | Compared against | Why |
|---|---|---|---|---|
| 1 | The screen is informative | Best candidate ≥ 0.3 above the goal-swapped control | Goal-swapped control | Otherwise no candidate has the capability; the list is revised, not a winner picked |
| 2 | Winner | Highest interaction-decisive top-1, with movement-decisive top-1 ≥ 0.8 and no-effect interactions ranked best on ≤ 0.1 of forks | The other four | Rare interactions are the target; the rest guard against buying them with movement or false changes |
| 3 | Tie-break | Within 0.05 of the best: higher rank correlation of d with true steps, then fewer built-in priors | Tied candidates | Prefer the one that also knows how soon, and assumes less |

**Prediction.** D wins on interaction-decisive forks if its movement
transport learns; if not, ego-motion floods its event codes and it trails
H. E scores well on movement and poorly on rare interactions, which get too
little signal without a transition model. A adds little over H. B is close
to H. All candidates are weak on "door open" goals with the key out of
view, which need memory (rung 2).

**Budget.** Gate: 1 run. Screen: 5 candidates × 2 seeds = 10 runs, each
under 10 minutes on the RTX 5070 Ti, about 2 hours; round 2 at most 6 more.

## 7. Result

## 8. Decision

---

## Appendix: the candidates

Sizes are starting points, fitted under the 2M cap.

**H, hub.** z: 192-d vector (pooled trunk, projector with BatchNorm). T:
action embedding through zero-initialised AdaLN in a small MLP; loss:
squared error to the encoded next frame, plus SIGReg on every encoding
(LeWM; no EMA or stop-gradient). g: mean of the encoded example frames. d:
metric-residual quasimetric head (MRN); trained action-free by expectile
regression toward 1 + d(z_next, g) on hindsight goals, with other
trajectories' frames as far goals (HILP, QRL). Scoring: d(T(z, a), g).
Risk: SIGReg under-spreads on low-diversity frames (LeWM lost to PLDM on
TwoRoom for this reason).

**A, + reconstruction.** H plus a convolutional decoder from z with a pixel
loss. Risk: a changed door or carried key is a few pixels and gets little
gradient.

**B, contrastive transition.** H, but T is trained by InfoNCE: T(z, a) must
pick the encoded next frame from the batch's next frames and from the
current frame (C-SWM objective). Risk: keeps whatever tells frames apart,
mostly viewpoint.

**D, factored state with partial goals.** z: an 8×8 grid of 24-d cells plus
one 32-d non-spatial vector (where a carried object can live). T: next =
move_a(z) + gate ⊙ write(e): move_a is a learned action-conditioned
transport of cell contents; e is one discrete event code per cell and one
for the non-spatial vector, from a 32-entry codebook whose code 0 means "no
change" (VQ-VAE); a prior p(e | z, a) predicts codes, a posterior that also
sees the next frame trains it; a penalty on the rate of non-null codes.
Persistence is the default. g: per factor, the mean over examples and a
weight that is high where the examples agree; d reads only weighted
factors. Same SIGReg and d training as H. Risks: a poor transport floods
the event codes; codes may collapse to "always null" (tracked with
evaluator labels).

**E, no transition.** Q(z, a, g) predicts steps-to-goal after action a
directly, trained by expectile regression toward 1 + min over a' of Q at
the next step. z is trained only through Q. Risk: rare interactions give
Q little signal without a transition model to share across goals.
