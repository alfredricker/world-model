---
id: "006"
title: conditions in the learned state
rung: 0
serves: [P12, P4, P7, C4]
status: done   # draft | approved | gated | running | done | abandoned
verdict: fail
arch_version: 0
date: 2026-09-26
---

# 006: conditions in the learned state

## 1. Question

Card 005's network predicts from pixels when an action achieves a goal,
but the conditions were read back in the simulator's words ("facing the
door", "holding the matching key"). Does the network's own internal state
have parts that act as those conditions: found without the simulator's
variables, and such that switching one part off switches the predicted
success off? This is step 2 of card 003's appendix and its open point (b):
"remove one condition, keep the rest" needs a state with separable parts.
It serves P12 (conditions and subgoals from the learner's own state), P4
and P7 (units that matter, discovered), and C4 (a condition must change the
prediction, not just be decodable).

## 2. What changes

No training of the network: card 005's final network
(`runs/005_data10k_long`) is frozen. New: a way to split its internal
state into candidate parts, and card 003's condition finder run on those
parts instead of the simulator's variables.

- **Internal state.** The 256 numbers z the encoder produces for a frame;
  both heads (achievement and walk) read only z.
- **Parts, candidate: a sparse dictionary.** A sparse autoencoder
  (Cunningham et al. 2023) learns 1024 directions such that each z is
  rebuilt from a few of them. A part is "direction i is active". Fitted
  on z of 1M training frames; three sparsity strengths are tried and the
  sparsest one that passes the fidelity check (section 5) is kept, decided
  before seeing any rule.
- **Parts, baseline: raw units.** "Unit i of z is above zero". Expected to
  fail if the network spreads each fact across many units.
- **Finding conditions.** Card 005's finder (FOIL gain, removal scores)
  with parts as the atoms. Its labels are the supplied success signals, as
  the agent has them: did this toggle unlock the door, did this pickup give
  the key that opens it; for walking to the goal square, the network's own
  walk value (W > 0.1).
- **Switching a part off.** Subtract the direction's contribution from z
  (its activation times the direction) and feed the result to the heads.
- **Evaluation only:** the simulator's variables name what a found part
  means (step 3 below); they are never used to find the parts.

Approved: the user said "proceed with implementation and running"
(2026-09-26), after card 005 had closed. Code:
`src/worldmodel/latent_conditions.py`.

## 3. Dependencies

Card 005 (final network passed all three criteria). Sparse autoencoders are
an established way to split a learned vector into separate, readable
directions (LITERATURE; in papi, text still to be added).

## 4. Data check

Training frames: card 005's 6.1M (10k episodes); the dictionary is fitted
on 1M of them. Condition rows: every toggle and pickup in training (about
1.2M each) and a quarter of all frames for walking. Test: card 005's 500
new layouts.

## 5. Feasibility gate

- **Upper bound:** a linear read-out of z predicts each needed simulator
  variable (facing the door, holding the matching key, door open, facing
  the matching key, empty hands) with area under the curve ≥ 0.95 on new
  layouts. If not, the facts are not in z in a form parts can capture, and
  the card stops there.
- **Dictionary fidelity:** feeding the rebuilt z to the heads keeps card
  005's criterion-1 cells within 2 points of the original.
- **Trivial baseline:** raw units as parts.

## 6. Success criteria and prediction

On new layouts, with dictionary parts:

1. **Small rules.** For unlocking, the rule or rules covering ≥ 90% of
   successes each have at most 3 parts, achieve ≥ 0.9, and every part has
   removal score ≥ 0.9. Several rules, one per colour, count: the relation
   "same colour" is rung 6, not this card.
2. **Parts act as conditions.** In test frames where the toggle succeeds,
   switching off one rule part drops the predicted success below 0.1 in ≥
   90% of frames; switching off an active part not in the rule leaves it ≥
   0.9 in ≥ 90%. Each rule part matches one simulator variable (area ≥ 0.95
   on new layouts), e.g. "facing the door", "holding the red key".
3. **A part becomes a subgoal.** The part that means "holding the key"
   switches on at ≥ 95% of pickups of that key and at few other steps
   (≥ 95% of its switch-ons are such pickups). With "this part switches
   on" as the goal, and no supplied label, the finder gives pickup + a
   facing-that-key part + an empty-hands part.

Also reported: the same for raw units; the walk-to-goal rule (expected:
one "door open" part); switching a part on in a near miss (facing the door,
empty hands) and whether predicted success rises.

Prediction: the gate passes (card 005's accuracy implies the facts are in
z). Raw units fail criterion 1 (rules of many units). The dictionary finds
per-colour rules; criterion 2 is the risk: a direction switched off may
leave z in a region the heads never saw, and "holding the key" may be
split over several directions. If criterion 2 fails, the next card builds
separable parts into the network itself (for example a sparse or factored
state trained end to end) rather than finding them afterwards. Runtime:
each step under 10 minutes.

## 7. Result

**First pass** (`runs/006_main`). Gate passed: a linear read-out of z
gives area 1.0 for facing the door, holding the matching key, door open and
empty hands, 0.9995 for facing the matching key. Dictionary fidelity
failed at all three strengths I had set (L1 0.003, 0.01, 0.03: 17.6, 5.4
and 0.2 active parts per frame): the toggle-with-matching-key cell fell
from 100% to 96%, 0% and 0%. The rare frames that matter (facing the
locked door with the key) are too rare in ordinary frames for a sparse
dictionary to spend parts on them. Second pass: weaker strengths (0.0003,
0.001, 0.003) and 20k fitting steps instead of 8k; the fidelity check and
all criteria unchanged.

**Second pass** (`runs/006_weak` stopped by me after 40 minutes stuck in
criterion 3, where fragmented rules made the finder produce rule after
rule; the analysis then saved each stage and reported at most 6 rules per
goal; `runs/006_l1e-3` reran it with the dictionary the sweep had chosen).
Dictionary kept: L1 0.001, 36.5 active parts per frame, 0.9% of z's
variance unexplained, all criterion-1 cells within 1 point. Numbers:
[results.json](results.json).

| | Dictionary parts | Raw units |
|---|---|---|
| Unlock: rules needed, share of test unlocks each | 6+, 14–36% each | 6+, 7–48% each |
| Parts per rule | 3–6 | 3–6 |
| Removal scores ≥ 0.9 | 5 of 21 | 12 of 31 |
| Best match of a rule part to a simulator variable | 0.92 (front=goal); "facing the door" 0.81; "holding the matching key" ≤ 0.69 | "facing a key" 0.97 (as a unit being off); "holding the matching key" 0.77 |
| Switching one rule part off: predicted success drops below 0.1 | 0% for 7 of 9 parts, at most 59% | only 3 "on" units could be switched: 0% |
| Switching another active part off: stays ≥ 0.9 | 97–100% | 98–100% |
| "Holding the key" part as a subgoal | part 967: on at 53% of matching-key pickups; 5% of its switch-ons are such pickups; the finder's rules for it are turns | not run |
| Walk to goal square | one "door open" part (0.89) + a positional part, 96% | not run |

| Criterion | Verdict |
|---|---|
| 1. Small rules | fail: no rule covers more than 36% of unlocks |
| 2. Parts act as conditions | fail: switching a part off almost never changes the prediction; no part matches a variable at 0.95 |
| 3. A part becomes a subgoal | fail |

What it shows: the facts are in z (a linear read-out gets them at 0.9995–1.0)
but spread over many redundant directions. The achievement head reads them
redundantly, so removing any one direction leaves its answer unchanged;
a dictionary fitted afterwards cuts z into pieces that do not line up with
the conditions. The prediction named this as the risk.

## 8. Decision

**Revise.** Conditions cannot be read off the internal state of a network
trained only to predict: it holds each fact redundantly. Corrected after
discussion with the user (2026-09-26): the theory does not need a
separable internal state. It needs each condition to be a separate
detector the agent can check, pursue and compare with, and card 005's
network supports such detectors (linear read-out 0.9995–1.0). A separable
state would buy later properties (counterfactual imagination, P3, P8), not
discovery. The open problem is discovery without a supplied vocabulary:
[card 007](../007-condition-discovery/card.md).
