---
id: "002"
title: rare-event sampling
rung: 1
serves: [P12, P6, P19]
status: draft   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 1
date: 2026-09-26
---

# 002: rare-event sampling

## 1. Question

Without labels, can the learner chosen by card 001 find its own rare
interactions (pickups, unlocks, door and box toggles) often enough to rank
actions by how they change steps-to-goal on the goal-conditioned fork, and
how many examples of each does it need? These interactions are how the
conditions of a goal get met, so this asks which label-free signal finds
them: any change, or a change that cannot be undone. Rung 1, properties
P12 (predictive side), P6 and P19. Depends on card 001 having a winner.

## 2. What changes

The architecture is card 001's winner, unchanged (arch_version 1). Card 001
chose training anchors with the evaluator's labels; this card removes the
labels. Only the sampler changes. Three arms:

- **A, uniform:** every anchor transition equally often.
- **B, null-margin priority:** each transition gets a priority, the model's
  own "nothing happened" error: how much closer its transition prediction
  T(z, a) is to the unchanged current latent than to the real next latent,
  in the model's own units, plus a floor of 0.01. Unmeasured transitions get
  the highest priority. No importance correction.
- **C, one-way priority:** each transition z → z' gets the priority
  max(0, d(z', z) − d(z, z')), clipped, from the model's own reachability
  head, plus the same floor. This is P19's new signal: a change that makes
  the previous state hard to get back to has met or broken a condition.
  Pickups, unlocks and box opens cannot be undone; plain door toggles can,
  so this arm should miss them. Unmeasured transitions get the highest
  priority, as in B.

An anchor is used by every loss: the transition loss on the anchor itself,
and the reachability loss on hindsight pairs and goal sets that start at
the anchor. Priorities sit on transitions, not on longer stretches, because
one rare transition barely moves a stretch's summed priority (LESSONS.md,
sampling). The margin is measured one step ahead, where the change first
appears. The one-way priority measures reachability back to where the
agent was, not toward the goal: toward any goal, a single step moves the
distance by at most 1, so conditions cannot be seen that way. Arm B needs
card 001's transition T and arm C its distance between states; if the
winner is E, which has neither, both are redefined before approval.

## 3. Dependencies

- **Card 001's winner, its training recipe and the goal-conditioned fork
  evaluator:** passed card 001's screen.
- **Card 001's one-way gap diagnostic:** arm C is run only if the winner's
  one-way gap is larger at interactions that cannot be undone than at
  toggles and moves. If it is not, arm C is dropped and the card says so.
- **Prioritised replay:** Schaul et al., 2016 (`1511_05952`). The
  null-margin and one-way priorities are our variants and are what arms B
  and C test.

## 4. Data check

Training data: 4000 episodes, 1.7M transitions, uniform opaque actions
repeated for geometric lengths; half are declared play starts. Interaction
counts in the old collection with the same seeds: key pickup 4052, unlock
307, other door opens about 1300, switch press 970, box open 823.
Old-repo diagnostics: 0.5% of all transitions carry a null margin, against
47–63% of interaction transitions (LESSONS.md, suggestive). No one-way
measure has been tried here.

**Event-count curve.** For each N in {10, 30, 100, 300, all}, keep N
occurrences of each interaction type (key pickup, unlock, plain door open,
box open; switch doors excluded as rung 2). Each removed occurrence splits
its episode there: no transition, hindsight pair or goal set may span it,
so its effect is never seen. Total data size is nearly the same at every N.
The evaluator's labels choose the occurrences; the learner never sees them.
N = 300 is the largest value every type allows.

Evaluation: card 001's stratified development fork states and goal example
pools (`runs/sampled_dev`), the same set and counts.

## 5. Feasibility gate

- **Upper bound:** card 001's winner trained with label-balanced anchors
  at N = 300. Its mean top-1 over card 001's six scored interaction strata,
  called U, is written here
  before approval. It must be at least 0.2 above the goal-swapped control,
  or this card waits.
- **Trivial baselines:** random ranking, fixed action preference, and the
  goal-swapped control, as in card 001.

Result of the gate, before the main run:

## 6. Success criteria and prediction

**Metric:** goal-conditioned fork top-1 (CHARTER rung 1): the best-ranked
action is among the truly best.

| # | Metric | Threshold | Compared against | Why this shows the capability |
|---|---|---|---|---|
| 1 | Mean top-1 over the scored interaction strata at N = 300, mean of 3 seeds | ≥ 0.8 × U | Label-balanced upper bound U | Finds rare events without labels nearly as well as with them |
| 2 | Same forks | ≥ 0.3 above the goal-swapped control | Same model shown another goal | The ranking depends on the goal, not on a fixed action preference |
| 3 | Movement-decisive top-1; no-effect interactions ranked best | ≥ 0.8; ≤ 0.1 of forks | Card 001's winner | Interactions are not bought with movement or false changes |

The rung passes if any arm meets all three. The curve (all N, both arms)
is reported per P19, with each arm's share of sampled anchors per
interaction type and the rank correlation of predicted and true
steps-to-goal (a CHARTER rung-1 criterion, but not this card's pass
criterion, which is about finding events). Results are reported per
goal type (key held, door open, box open); play-start probes and goals that
need a switch are reported, not scored.

**Prediction.** Arm B passes at N = 300; arm A falls short on door goals,
as the old learner fell short on toggles under uniform sampling (0.24 at
one step). Arm C matches B on key goals, box goals and door goals behind
a locked door, and is worse than B on plain doors. All arms score low at
N = 10 and 30. A slow learner needing hundreds
of examples is the expected baseline, and the reason for a later card on an
episodic store.

**Budget.** Gate: reuses card 001's runs where possible, else 1 run. Curve:
5 values of N × 3 arms, seed 0, plus seeds 1 and 2 at N = 300 for all
arms: 21 runs. Updates per run as in card 001; runs over 30 minutes are
handed to the user.

## 7. Result

## 8. Decision
