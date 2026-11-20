---
id: "149"
title: a sequence of actions by recurrence, from the energy of card 148
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-19
---

# 149: a sequence of actions by recurrence

Roadmap step 2, rung 2, on the direction under trial (energy modules). The
user, 2026-11-19: "I also want to test out sequences", and on the form: "I
was thinking the natural design was a recurrence ... whatever output you
choose updates the internal state accordingly allowing for a sequence of
output states"; asked whether to draft it that way: "yes". Each run is
under ten minutes. Biological counterpart: an inverse model that proposes
an action and a forward model that predicts its effect, paired (Wolpert and
Kawato 1998), with prefrontal working memory holding the goal while one
action at a time is released.

## 1. Question

Can card 148's energy, used one step at a time with the world answering in
between, produce a sequence of different actions that reaches an effect no
single step can, and what does the agent have to carry from step to step?
P7.

## 2. What changes

Nothing in og4. Around card 148's couplings (unchanged, the same setting) a
loop is added, and one memory.

```
each step   observe: where the hand and objects are, their widths, which objects the hand touches
            carried: a belief of which object is held (not observed)
  propose   clamp the state and the wanted effect, settle the action              (the energy, inverse)
  verify    clamp the state and that action, settle every effect: is it the wanted one?   (the energy, forward)
  if not    run both again as if the object were touched; if the wanted effect then comes out, touching the
            object is wanted first.  The nearest stored place of the hand relative to an object it touched
            (memory, one shot, same width preferred) gives a wanted move of the hand; the action for it is settled
  act       the world answers
  update    clamp what was seen (touch, action, how hand and objects moved), settle "held after": the new belief
tasks       1: hold object X.   2: object X at place G and let go.
            The wanted effect of a step is read off the goal: held; or X moved toward G by at most 4; or let go.
```

Given, not learned: subtraction of places (a place relative to an object;
how far an object is from where it is wanted). A touching place that is
reached without touching is not used again in the episode.

**A first design failed and is not the one tested.** It judged whether a
wanted effect can happen from the probability the energy gives the wanted
token after settling with it clamped. On validation, task 1 succeeded in
between 0% and 98% of episodes depending on how many records the couplings
had seen, with no order to it. The couplings are trained only on
situations that occurred; clamped to an effect that cannot happen from
here, their probabilities mean little, while on real records of taking
they give "touched" and "held after" at 0.97 to 1.00. Propose-then-verify
only ever asks about a state and an action.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| As card 148: touch is sensed, effects are a few tokens, objects from the simulator | no | no | no | partly | declared, as card 148 |
| Places can be subtracted (an object's own frame) | yes (pose is given) | yes | yes | yes | given: og2's skills use the same |
| One unmet condition (touching) lies between the goal and now | no | no | no | no (drawers, locks) | declared: the chain is one link deep; the button and lock are not in this world |
| The hand can go straight to a place | no | no | mostly | no (obstacles) | declared |
| A goal place leaves room for the hand on every side | - | - | - | - | declared: places within 7 of the edge are not asked |

## 3. Dependencies

Card 148's couplings, with its setting (L1 0.001, 500 steps), refit on
each stream of records; its simulator. The energy is settled from four
starts (one uniform, three random) and the lowest kept, because more
tokens are free here than the three of card 148.

## 4. Data check

Experience as card 148 (scripted behaviour with 20% random actions). The
stored touching places number 8, 25, 69 and 153 after 100, 300, 1,000 and
3,000 steps (validation stream). Tasks: 200 random scenes each; task 2's
place is at least 8 from where the object starts.

## 5. Feasibility gate

- **Upper bound:** a scripted solver that goes to the nearest side, takes,
  carries and lets go.
- **Trivial baselines:** the same energy with no verification (the action
  settled for the goal is always carried out); random actions.

Result of the gate, on validation tasks, before the main run: passed. The
solver succeeds in 98.5% (task 1, median 6 steps) and 100% (task 2, 12
steps) within 60 steps; without verification 0.0% and 2.5%; random actions
12.0% and 0.0%.

## 6. Success criteria and prediction

On 200 test tasks of each kind, 60 steps allowed, mean of five streams of
records.

1. **With 3,000 steps of experience:** task 1 at least 93.0%, task 2 at
   least 90.0%.
2. **No wandering:** successful episodes take at most 1.5 times the
   scripted solver's steps on both tasks (AGENTS' default margin).
3. **With 300 steps of experience:** task 1 at least 90.0%.

Reported without a bar: task 2 with less experience; the agent without its
belief of what is held; how often the belief is right; how often no way is
found.

Settings: none new was tuned. A weaker penalty with longer training (L1
0.0001, 2,000 steps) was tried on validation and was worse at 3,000 steps
(task 1 30.5% against 97.5%), so card 148's setting stayed. On validation
the tested design scored, with 100, 300, 1,000 and 3,000 steps: task 1
83.0, 98.0, 98.0, 97.5%; task 2 12.5, 38.5, 91.5, 98.0%. The bars were
written after seeing these.

Prediction: all three pass; task 2 needs about 1,000 steps of experience.

## 7. Result

Percent of 200 test tasks reached within 60 steps, mean of five streams
(each stream's value in brackets where they disagree); numbers in
[results.json](results.json). The scripted solver: 97.5% and 100.0%.

| Steps of experience | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|
| Task 1, hold X | 75.3 | 89.2 (97.5, 97.0, 91.0, 63.0, 97.5) | 64.4 (97.5, 97.0, 30.5, 0.0, 97.0) | **58.8** (99.0, 0.0, 0.0, 98.0, 97.0) |
| Task 2, X at G and let go | 16.2 | 48.4 | 83.4 | **96.9** (94.0 to 99.0) |

| At 3,000 steps | Task 1 | Task 2 |
|---|---|---|
| No verification | 0.0 | 2.5 |
| Random actions | 20.5 | 0.0 |
| Without the belief of what is held | 58.8 | 23.9 |
| Steps against the solver's, successful episodes | 1.10 | 1.00 |
| Belief of what is held right, per step | 98.5 | 95.6 |

| Criterion | Verdict |
|---|---|
| 1. 3,000 steps: task 1 ≥ 93.0%, task 2 ≥ 90.0% | **fail** (58.8; 96.9) |
| 2. Steps at most 1.5 times the solver's | pass (1.10; 1.00) |
| 3. 300 steps: task 1 ≥ 90.0% | **fail** (89.2) |

The card fails, 1 of 3. The harder task passes and the easier one does not,
for one reason, which I checked. For "hold X" while touching X, the energy
proposes in every stream "close the grip and do not move". In the records
the hand took an object without moving 0 or 1 times in about 200 takes
(the scripted behaviour takes and moves off in one step). Asked forward
what that action does, the couplings of streams 100, 103 and 104 say
"held" and those of 101 and 102 say "not held"; task 1 succeeds in 97 to
99% on the first three and 0% on the other two. Task 2 wants the object
moved, so its proposals are takes with a move, which the records hold, and
it succeeds on all five streams. The single validation stream happened to
be of the first kind, so validation did not show this.

So the recurrence itself does what was asked where the energy can be
trusted: it produces approach, take, carry and let-go in order, in the
solver's number of steps, from one energy used in both directions, and
nothing works without the verification (0.0%, 2.5%). The carried belief
matters: without it task 2 falls from 96.9 to 23.9%. What fails is the
same thing as the first design's failure, one level down: the energy
answers reliably about combinations it has seen and arbitrarily about
ones it has not, and settling can propose an unseen one.

Other limits: touch, widths and displacements come from the simulator; the
chain of conditions is one link deep; the wanted effect of each step is
read off the goal by a rule I wrote; with little experience task 2 is poor
(16.2% at 100 steps) because verification fails more often than the action
is wrong.

## 8. Decision

**Revise.** Not by a setting: the bars stay as they are and task 1 failed.
The revision is for the agent to learn from its own proposals. When the
energy proposes an action in a combination its records do not hold, the
agent tries it and the outcome becomes a record, so what the energy is
asked about and what it has experienced stop being two different sets
(the user's "conditions from memory and experiment", 2026-10-17). That is
card 149.1, to be run only with the user's go-ahead. Kept from this card:
propose-then-verify with one energy, and the belief of what is held
settled forward each step.
