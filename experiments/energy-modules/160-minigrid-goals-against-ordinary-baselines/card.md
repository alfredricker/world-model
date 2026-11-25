---
id: "160"
title: goals in MiniGrid, against ordinary learned baselines
rung: 5
serves: [P12, P6, P19]
status: done
verdict: fail
arch_version: og4
date: 2026-11-21
---

# 160: goals in MiniGrid, against ordinary learned baselines

The user, 2026-11-21: "let's run an honest minigrid evaluation". GOAL.md
now names MiniGrid as the first benchmark. Until this card the energy
module line (cards 146 to 159) had been run only in a sprite world I
built, with no baseline. Runs take under ten minutes each. Biological
counterpart of the one new part (below): prefrontal goal maintenance
with backward chaining over known contingencies, and hippocampal
closeness to a place for walking.

**Terms.** A **token** is one grid cell's content with its place. A
**code** is the name the agent gives a distinct tile (the dictionary
grows as tiles are seen; a code's meaning is never told). An **event**
is a change of a token's code. A token **out of view** is one the agent
saw and no longer sees; it is remembered with the action at which it
went. A **statement** (GOAL P12) is an effect with its conditions: the
action, where the agent must be, what must be out of view, what must
not be. **PPO** is proximal policy optimisation (Schulman and others
2017), the usual model-free baseline for MiniGrid.

## 1. Question

Given only reward-free random play, can the sparse transition energy of
card [158](../158-events-as-changes-of-an-objects-latent-state/card.md),
with conditions read from it and chained backward, reach goals it was
never rewarded for in new MiniGrid layouts, and does it beat an
ordinary learned baseline? CHARTER rung 5 in MiniGrid (P12), with P6.

## 2. What changes

One new component: **acting**, by chaining statements backward from a
goal. The rest is card 158's form carried to MiniGrid.

```
perception (declared, GOAL C1)   the full grid as tiles, one token per cell; the agent's pose; its own action.
                                 The tile under the agent and what it carries are not shown (as in MiniGrid's full view).
state kept by the modules        each cell's last seen code; the tokens out of view, each with the action at which it went
transition (card 158's form)     p(next code | code, action, where the agent is relative to the token, tokens out of view)
                                 = softmax of a bias plus one sparse coupling per input token (L1, 1 nat as in 158):
                                 a set of couplings shared by all codes, toward any change, plus each code's own
                                 sparse difference; the same form for the agent's own move, read from the codes of
                                 the eight cells around it
statements (card 158)            from the commonest experienced success of an effect: its action; the places the model
                                 allows; a token out of view is a condition if the model says the effect fails without
                                 it (or with it) and the experience holds at least 3 tries of the effect in that case
goal                             shown as 20 example success steps from other layouts; the goal is the event seen there
NEW acting                       the goal's statement; an unmet condition becomes the goal of the effect that makes it
                                 true; "be at this place" is walked by closeness propagated over the learned move
                                 effect; a blocked walk asks for the effect that makes the blocking token passable;
                                 an effect done only to free a condition is checked forward once so it breaks no other
```

Not carried over from the sprite cards: binding from motion (tokens are
supplied here), dynamics classes (153.1), and the inference of a hidden
state by expectation-maximisation (in MiniGrid's full view a token going
out of view is seen, so nothing needs inferring).

**Supplied, not learned** (C1): cells as tokens; the full view; the
window of relative places (two cells around the token); the two levels
of memory of a token out of view (its code with the action, the action
alone); the chaining procedure itself. **Learned:** codes, every
effect, which cell blocks a move, every statement, the goal.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Tokens with exact outlines are supplied | adapter | adapter | no (card 159) | declared |
| The whole scene is in view | no | yes | no | declared; partial view needs belief (P2) |
| Effects are local to the agent and deterministic | mostly | no | no | declared |
| A token is identified by its code | no (many trees) | no | no | declared |
| Discrete actions and places | yes | yes | no | declared |
| Walking by propagation over a learned move effect | yes | partly | no (needs a learned closeness) | declared |

## 3. Dependencies

Card 158's transition form and read-out (kept; failed 1 of 3 on object
extents, which are exact here). Card 159's relative place. PPO and
MiniGrid: LITERATURE.md, "Baselines". The chaining is new and is this
card's component.

## 4. Data check

Experience: uniform random actions, no reward, in DoorKey-5x5,
DoorKey-6x6, Unlock, UnlockPickup and BlockedUnlockPickup (as first
written), the same number of steps in each. The number is set here, before any model is
fitted, so that each effect in MiniGrid's rules occurs often enough.

Result (100,000 steps a world, experience 1). With the five worlds, 36
of the 60 effects in MiniGrid's rules were experienced at least 3 times:
boxes lie behind the locked door (picked up once in 500,000 steps), and
closed doors were toggled 0 to 8 times a colour. Two worlds were added
before any model was fitted, PutNear-6x6-N2 (boxes within reach) and
MultiRoom-N2-S4 (closed doors). With seven worlds all 60 are experienced
at least 3 times in each of the three experiences; the rarest is
unlocking a door of one colour, 4 to 12 times. The experience is
700,000 steps.

## 5. Feasibility gate

- **Upper bound:** the same agent with MiniGrid's rules written by hand
  in place of the learned energy, on 50 layouts of each test task except
  KeyCorridor (seeds 2000 to 2049, never the test layouts).
- **Trivial baseline:** random actions.
- **Baseline check:** my PPO must reach 90% on DoorKey-5x5 and
  DoorKey-6x6, as published implementations do, before it is used.

Result: passes. On 50 layouts each (seeds 2000 to 2049) the agent told
the rules solves DoorKey-5x5, 6x6, 8x8, 16x16, Unlock, UnlockPickup and
BlockedUnlockPickup 50 of 50; random actions solve 4, 4, 0, 0, 4, 0 and
0. PPO reaches 100% on DoorKey-5x5 by 100,000 frames and on DoorKey-6x6
by 200,000, in both views.

**Development** (experience 1; the three development tasks; before the
test). Fixed here, then frozen (`runs/160/frozen.txt`):
- The chain puts a thing down only once nothing stands between the
  agent and the place it needs; otherwise it dropped and retook the key
  without end. Found with the rules agent in the gate, so the chaining
  was debugged on the test families other than KeyCorridor; what the
  test asks is whether the learned energy can replace the rules.
- The shared couplings of card 158 were restored after a fit with
  couplings per code only named 45 of 60 statements; with them 52.
- Statements are scored on meaning: a statement is right if it allows
  its effect in exactly the combinations of action, place and thing
  carried in which MiniGrid's rules do, over the places experienced.
- PPO's longer run is twice the steps, not three times, to keep each
  run under ten minutes.
- On the development tasks the learned agent solves 100 of 100 each.

## 6. Success criteria and prediction

**Protocol.** Development on DoorKey-5x5, DoorKey-6x6 and Unlock
(layouts 500 to 599). Test, run once with the code frozen, 100 layouts
each (seeds 1000 to 1099): DoorKey-8x8 and DoorKey-16x16 (sizes never
experienced), UnlockPickup and BlockedUnlockPickup (new layouts), and
KeyCorridorS3R3 (a family never experienced and never run before the
test, gate included). Three experiences collected independently; every
number is given for each. Success is MiniGrid's own, within its own step
limit. PPO trains on each test task with MiniGrid's reward, with the
standard 7 × 7 view and with the full view, three seeds; its better
view counts.

1. **Acting (P12, C5).** Success averaged over the five test tasks is at
   least 80%, and on no task more than 5 points below PPO at the same
   number of environment steps. Meeting it shows conditions pursued as
   unrewarded subgoals in new layouts.
2. **Naming conditions (P12).** Of the effects experienced at least 3
   times, at least 90% of statements equal the statement read the same
   way from MiniGrid's rules, in each experience. Reported beside a
   multilayer perceptron on the same inputs, read the same way.
3. **Effects in new layouts (P6).** On test-task trajectories, the F1 of
   predicted events (right token, step and outcome) is at least 0.95 and
   no more than 2 points below the perceptron's.

Also reported: the agent run on the perceptron; PPO at twice the
steps; steps per success; the share of steps with no plan; time per
step (P17); success against the size of the experience (P19).

**Prediction.** Criteria 2 and 3 pass for common effects and may miss
on rare door colours. Criterion 1: DoorKey passes; BlockedUnlockPickup
and KeyCorridor are uncertain, because the chaining has never been run
and KeyCorridor is blind. PPO should solve DoorKey-8x8 at twice the
steps and none of the others.

## 7. Result

Success, percent of 100 new layouts; the three experiences side by side
where they differ (numbers in [results.json](results.json)).

| | DoorKey-8x8 | DoorKey-16x16 | UnlockPickup | BlockedUnlockPickup | KeyCorridorS3R3 | mean |
|---|---|---|---|---|---|---|
| Random actions | 2 | 1 | 0 | 0 | 0 | 0.6 |
| PPO, 700,000 steps, better view, 3 seeds | 6.0 | 0.3 | 0.7 | 0.7 | 0.3 | 1.6 |
| PPO, 1,400,000 steps, standard view, 2 seeds | 7.0 | 1.0 | 0 | 0 | 0 | 1.6 |
| **Energy model, frozen test** | 100, 100, 97 | 100, 100, 99 | 10, 7, 9 | 9, 11, 9 | 16, 0, 0 | **44.5** |
| Perceptron in the same agent, frozen test | 100 | 100 | 41 | 45, 46, 46 | 86, 69, 72 | 72.5 |
| Agent told MiniGrid's rules (upper bound) | 100 | 100 | 100 | 100 | 100 | 100 |
| After the test: energy model, one planner fix | 100 | 100 | 41, 47, 38 | 35, 48, 36 | 22, 0, 1 | 57.9 |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Mean success at least 80%, no task below PPO | 44.5% (47.0, 43.6, 42.8); above PPO on every task | **fail** |
| 2. At least 90% of statements right in each experience | 52, 55 and 54 of 60 (87%, 92%, 90%); the perceptron 52, 53, 54 | **fail** |
| 3. F1 of predicted events at least 0.95, within 2 points of the perceptron | 100 on four tasks, 98.6 on KeyCorridor (100, 98.5, 97.3); the perceptron 78.3 there | **pass** |

**What the test shows.**
- **DoorKey is solved, in sizes never experienced.** 99.0% and 99.7%, in
  17 and 33 steps where the plan held (the rules agent: 16.5 and 33.0).
  PPO with the reward on the task itself reaches 6% and 0.3% in as many
  steps, and 7% and 1% in twice as many. The key was never rewarded.
- **The three other tasks fail**, with no plan on 60 to 99% of steps.
- **The ordinary model did better in the same agent** (72.5 against
  44.5). Sparse couplings are not what makes the agent work here.
- **The chaining itself is sound:** told the rules, it solves all five
  tasks 100 of 100, KeyCorridor included, which it had never been run on.
- **Compute (P17):** 1 to 55 ms a step; fitting the energy 23 to 51 s,
  the perceptron 202 to 414 s. Of 267,279 couplings 101 to 111 are not
  zero, plus 20 to 23 shared.

**Why the three tasks fail** (checked after the test, on layouts outside
it; `dev6.py` to `dev11.py`).
1. **The same fact enters the energy twice, and L1 splits the credit.**
   When a door opens, the model sees the door's new code and also "the
   closed door went out of view by toggle". In play the two always occur
   together. The learned move effect lets the agent through an open door
   of five colours of six only when the memory of opening it is there
   (all three experiences; yellow, seen 185 to 230 times, is the
   exception). The planner imagined the door open without that memory,
   so no path existed. My first explanation, too few walks through open
   doors, was wrong: play made 7 to 24 a colour heading east, all of
   which moved.
2. **"Hands must be empty" is missing for 5 to 8 objects of 24.** A
   pickup with something already held was tried 0 to 2 times for those
   codes. The condition is learned per code; nothing carries it from the
   keys and balls that have it to the boxes that do not. This is what
   card 153.1's dynamics classes were for, and they were not carried
   over.
3. **When a try fails twice the agent acts at random,** and a random
   toggle destroys the box.

With cause 1 removed (the planner imagines an effect's trace in memory
too; `FIX=1`, not part of the test) the energy model reaches 42% and
40% on the two pickup tasks, the perceptron's level, and causes 2 and 3
remain. KeyCorridor stays at 8%: not explained.

**Experience (P19).** With 87,500, 175,000 and 350,000 steps the energy
model solves DoorKey-8x8 7%, 16% and 7% (700,000: 100%). In the
350,000-step experience the yellow open door is also passable only with
the memory of opening it. The result at 700,000 steps sits just past a
cliff; experience 3 already shows it (78% of DoorKey-8x8 steps had no
plan, and random actions carried it to 97%).

**Limits of this evaluation.**
- Tokens, the full view and the chaining procedure are supplied (C1).
  The memory is not hidden state to infer: MiniGrid shows a token leaving.
- I chose the seven play worlds so that every effect occurs; two were
  added after the first count.
- The chaining was debugged with the rules agent on four of the five
  test families (other layouts). Only KeyCorridor was blind.
- PPO's second test is at twice the steps, on two seeds, in the standard
  view only; the full view on DoorKey-16x16 reached no test in ten
  minutes.
- Criterion 2 was scored on meaning, a change from the first wording,
  made before the test.
- After the freeze, `agent.py` gained the `FIX` switch (off in the test)
  and `evaluate.py` a correction to the events scorer, which had crashed;
  the other five files match `runs/160/frozen.txt`.

## 8. Decision

**Revise.** The new component, acting by chaining learned statements,
is kept: it is the first time this line reaches goals in a real
benchmark, and with true rules it solves every task. What fails is the
learned energy underneath it, in two ways the test made concrete: a
token that changes in place must not also be counted as a token out of
view (the front end's correspondence can say which is which), and a
condition must be shared among codes that behave alike (card 153.1's
classes). The ordinary perceptron doing better in the same agent means
the sparse form has not yet earned its place in MiniGrid; the next card
should carry the perceptron as its comparison again. Not adopted;
og4 unchanged.
