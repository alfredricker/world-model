---
id: "056"
title: goals from example frames
rung: 1
serves: [P12, P16, C1, P4, P17]
status: approved
verdict:
arch_version: 10
date: 2026-10-05
---

# 056: goals from example frames

Drafted and started overnight on 2026-10-05 under the user's standing
instruction for the night ("then the next cards should be the goals from
example frames"; AGENTS.md, "Overnight sessions"). The numbers below were
fixed before any run.

## 1. Question

Shown five frames from other episodes in which a goal holds (the agent
holding a given key, a given door open, the switch on), can version 10's
agent work out which condition the frames share, using only its own
perception and its own memory of play, and then reach that condition
through its chain of conditions about as well as when the condition is
written in by hand? Rung 1 (CHARTER: "the learner is shown a goal
condition ... as a set of example frames from other episodes in which it
holds"), first of its two criteria; P12 (a goal is a condition in the
learned state), P16 (goals inferred from examples), C1 (goals supplied
only as example observations; what they require is learned).

## 2. What changes

One component: the planner's top condition.

| | Version 10 | This card |
|---|---|---|
| Top condition | "the episode ends" (forward onto a tile predicted to end it) | a goal inferred from K = 5 goal frames: a conjunction of features in the agent's own codes, "the hand holds h" or "some tile in view shows u" |
| How it is chosen | supplied by the world | Bayesian concept learning with the size principle (below) |

**Inference.** Each frame is read through the agent's perception (card
054's encoder, identity up to noise). The candidates are the features
present in every goal frame. A hypothesis g (a set of features) scores
log P(g | frames) = −|g| log N + K log(1 / p(g)), where N is the number
of candidates, so that naming a feature costs its description length,
and p(g) is the share of the agent's own experience frames (its stored
random play, weighted as stored) in which every feature of g holds,
smoothed as (n + 1)/(total + 2). The second term is the size principle
(Tenenbaum and Griffiths 2001): examples chosen because they satisfy g
are more likely under a g that is rare in ordinary experience. Features
are added greedily while the score rises. Because p(g) is conditional
on the features already chosen, a feature that merely comes with the
goal (the key in hand when a door is open in the key world) adds
almost nothing and is left out. The rates are computed from the stored
frames when a goal is posed; nothing is tallied in advance (CHARTER:
experiences are kept, not reduced to tallies).

**Planning.** New conditions ("has", h) and ("shows", u), and ("all",
g) for a conjunction, whose achievers are its unmet features. A
feature's achievers are version 10's achievers for a part condition: any
pick up, drop or toggle on a tile in view whose predicted effect makes
the feature true, with recall's conditions beneath it, then walking.
When the agent believes its goal holds, it idles (turns), so that a
wrongly inferred goal cannot succeed by random actions.

Nothing else changes: encoder, recall, admission, walking, threats and
commitment are version 10's. Declared limits: goals of absence ("the
vase is gone") and of place ("a door at this place open") cannot be
stated with these features; neither is tested.

## 3. Dependencies

- Card 054's encoder with identity up to noise (seed by seed; if 054
  stops, version 10's own encoder).
- Version 10 (cards 049–051): recall, admission, the planner, its
  familiar worlds and test layouts.
- Literature: Tenenbaum and Griffiths 2001 (`tenenbaum-griffiths-
  generalization`; the size principle under strong sampling);
  Eysenbach et al. 2021 (`2103_12656`; goals given as examples,
  contrasted with the agent's own experience); card 049's admission
  (gain against description length).

## 4. Data check

Goal frames come from random play with play starts on 200 fresh layouts
per world (seed 9000 on), states where the simulator says the goal holds,
for each colour; the pools' sizes are reported. The agent's experience
is its stored play (about 505,000 frames per world). Test: version 10's
30 test layouts per familiar world (key, switch, either, both), four goals
each: hold the door's key, hold the other key, the door open, the switch
on (the colours are the layout's).

## 5. Feasibility gate

- **Upper bound:** the same agent with the goal written in (the evaluator
  names the target key's, open door's or switch's code).
- **Trivial baseline:** the goal-swapped control: the same agent shown
  another goal's frames (hold the door's key ↔ hold the other key; door
  open ↔ switch on), scored on the true goal.

The gate passes if the upper bound reaches ≥ 95% in every world on seed
399.

## 6. Success criteria and prediction

Encoders of seeds 399–402; each criterion in at least 3 of 4.
1. **The inferred goal is the true one:** on 2,000 states per world from
   random play on the test layouts, the inferred condition holds exactly
   when the simulator's goal holds, balanced accuracy ≥ 95% for every
   goal and world, averaged over 20 draws of five frames.
2. **The agent reaches it:** success ≥ 95% in every world (four goals
   pooled), within 3 points of the upper bound; the goal-swapped control
   at most half the success.
3. **Rung 1's interaction-decisive forks:** on states from random play
   where the evaluator's search says every fastest action toward the goal
   is a pick up, drop or toggle, the agent's action is one of them in
   ≥ 90%, and ≥ 40 points above the goal-swapped control at the same
   states.

Reported: K = 1, 2, 3, 5, 10 for criterion 1, and the rule without base
rates (every feature common to the frames); steps against the shortest
route; how often the chain for "door open" passes through a condition on
the hand (the agent discovering that the key is needed); idle steps.
Rung 1's second criterion (how soon) needs the planner to estimate a
whole chain's length; it is not tested here.

**Prediction.** Inference near exact: identity codes separate key
colours and door states, and the conditional rates drop companions.
Acting near 100% for keys and the switch; "door open" goes through the
key, the switch or both, as version 10 does for the episode's end.
Forks ≥ 95%; the control near 0 on keys, higher for door ↔ switch in the
switch world (turning the switch on is a fastest action toward the door).

**Decision rules.** Keep if all three hold in 3 of 4 seeds (version 11:
goals from frames). One declared revision if one fails. Stop if acting
falls below 80% of the upper bound.

**Budget.** About 15 minutes per seed, seeds in parallel.

## 7. Result

## 8. Decision
