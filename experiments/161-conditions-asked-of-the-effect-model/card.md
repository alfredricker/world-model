---
id: "161"
title: conditions asked of the effect model in the present situation — one model for predicting a tick and for planning, in MiniGrid
rung: 5
serves: [P12, P6, P3]
status: done
verdict: fail
arch_version: og4
date: 2026-11-24
---

# 161: conditions asked of the effect model in the present situation

Back in MiniGrid after the energy cards were set aside
([the series' note](../energy-modules/README.md)). The agent is
[card 160](../energy-modules/160-minigrid-goals-against-ordinary-baselines/card.md)'s:
it learns what actions do from reward-free random play and reaches a goal
by chaining conditions backward from it. Its effect model here is card
160's perceptron, an ordinary network.

The user's rule for this card (2026-11-24): build from how parts are
similar, not from how they differ. Card 160 kept two things that do one
job. The effect model predicts what every token does at the next tick.
Beside it sat a second structure, the statement, compiled in advance with
its own rule of evidence, and the planner read conditions from that.

## 1. Question

If the planner asks the effect model itself whether an effect will happen
in the present situation, in place of reading a statement compiled in
advance, does the agent solve the tasks card 160 failed? Rung 5. P12
(conditions evaluated from the present state and memory), P6, P3 (a
condition learned on some objects applies to others).

## 2. What changes

One component: how the agent learns that an effect fails while some token
is out of view (in MiniGrid, "the hands must be empty").

```
160:  statement(code, effect) made in advance = action, places, needed tokens, tokens to avoid.
      "avoid X" is stated only if this code itself was tried at least 3 times with X out of view.
161:  when the plan wants an effect, ask the model: with what is out of view now, will it happen?
      If not: would it happen with one of those tokens back in view? Then bring that token back first.
```

The statement keeps the action, the places and the needed tokens. The
experience, the fits and the rest of the agent are card 160's.

**Two arms**, the same agent in both:
- **A:** card 160's perceptron, unchanged.
- **B:** the same network with one output, "nothing changes", shared by
  every token. In card 160 each code had its own output for staying as it
  is, so "this action does nothing here" was learned code by code.

Not in this card, in the order I would take them: needed tokens asked of
the model the same way; the agent's own move predicted by the same
network as every other token; an effect the agent pursues over many
ticks treated as an action whose "next tick" is the next event.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds (cells) | needs a front end | declared, as in card 160 |
| The whole grid is in view | no | no | holds | no | declared, as in card 160 |
| A token out of view is remembered with the action at which it went | holds | an inventory holds counts | rarely needed | holds (grasped, in a drawer) | general; counts are not handled |
| One token back in view is enough to make an effect possible | mostly | several items can spoil | mostly | mostly | declared; when no single token explains the failure the agent tries anyway |

## 3. Dependencies

- Card 160: the tokens, the memory, the experience (three random-play
  experiences of 700,000 steps), the perceptron, the chaining agent with
  the planner repair made after its test, the rules agent, the tasks.
- Contrast between a failure and a success to find a condition: card 111
  and cards 125–125.2 here; established in the literature as learning
  preconditions by contrast (LITERATURE, "from skills to symbols").

## 4. Data check

Where card 160's agent first runs out of plan, 40 layouts per task that
were never tested (seeds 2000–2039), experience 1, the perceptron
(`tools/card161/diag.py`):

| Task | Solved | First step without a plan, in the failures |
|---|---|---|
| UnlockPickup | 11 of 40 | 29 of 29: holding the key, facing the box |
| BlockedUnlockPickup | 13 of 40 | 27 of 27: holding the key and facing the box, or holding the ball and facing the key |
| KeyCorridorS3R3 | 36 of 40 | hands empty, facing a wall, a door or nothing; not explained |

In all 56 failures on the two pickup tasks the agent tried to pick up with
full hands, failed twice, acted at random and destroyed the box with a
random toggle. The statement lacked "hands must be empty" for 8 objects,
all six boxes among them: random play never picked up a box with
full hands.

**The model already knew.** Asked directly, the perceptron gives a pickup
with a key held a probability of 0.00 to 0.27, for boxes it never tried in
that case (`tools/card161/dev1.py`). The statement's rule of evidence
discarded this. Dropping that rule without changing the question makes
statements worse (39 of 60 right against 52), because the statement asks
about combinations no situation produces. Hence the change in section 2:
ask about the present situation.

## 5. Feasibility gate

- **Upper bound:** the agent told MiniGrid's rules, 50 of 50 on every task
  (card 160's gate).
- **Trivial baseline:** card 160's agent with the perceptron, the table
  above.

Result (2026-11-24, the same 40 layouts, experience 1):

| | UnlockPickup | BlockedUnlockPickup | KeyCorridorS3R3 |
|---|---|---|---|
| Card 160's agent, perceptron | 11 | 13 | 36 |
| Arm A | 30 | 33 | 36 |
| Arm B | 38 | 37 | 33 |

Passed: the test can be passed and the baseline is far below. Every
remaining failure on the pickup tasks still ends with the box destroyed by
a random action. Arm B's lower KeyCorridor score (33 against 36) is within
what 40 layouts can show; the test will say.

## 6. Success criteria and prediction

Test: card 160's five tasks, 100 fresh layouts each (seeds 5000–5099),
three experiences, code frozen before the run. Card 160's test layouts
(seeds 1000–1099) are run too and reported, but not judged: they have been
run several times. Card 160's agent with the perceptron is run on the
fresh layouts as the comparison. Each criterion is judged per arm.

1. **Success:** mean over the five tasks at least 85% in each experience
   (card 160's agent with the perceptron: 72.5%; told the rules: 100%),
   and no task more than 5 points under card 160's agent.
2. **The failed tasks:** UnlockPickup and BlockedUnlockPickup each at
   least 80% on the mean of three experiences (card 160: 41 and 46%).
3. **The condition carries to objects never tried:** for every object that
   can be picked up and every token that can be held, the model's answer
   to "will the pickup happen" is right in at least 95% of cases in each
   experience, scored against MiniGrid's rules, counting the objects never
   tried with full hands (card 160's statements: 87 to 92% right).

Prediction: arm A passes 1 and fails 2 and 3 (it gave one box 0.82 with
another box held). Arm B passes all three. KeyCorridor stays near 85%.

Budget: two fits of arm B's network (3.5 minutes each; experience 1 is
fitted), then 15 test runs, each split so that no run exceeds 10 minutes;
about 90 minutes in all.

## 7. Result

Run 2026-11-24, approved by the user. Code frozen before the test
(`runs/161/frozen.txt`, unchanged after it); numbers in
[results.json](results.json); `tools/card161/` (`ev.py`, `wave.sh`).

Percent of 100 fresh layouts solved, mean of three experiences (each
experience in brackets):

| | DoorKey-8x8 | DoorKey-16x16 | UnlockPickup | BlockedUnlockPickup | KeyCorridorS3R3 | Mean |
|---|---|---|---|---|---|---|
| Card 160's agent, perceptron | 100 | 100 | 41.0 (41, 41, 41) | 39.3 (46, 36, 36) | 71.3 (87, 61, 66) | 70.3 (74.8, 67.6, 68.6) |
| **Arm A** | 100 | 100 | 84.7 (80, 87, 87) | 82.7 (75, 85, 88) | 66.7 (83, 61, 56) | **86.8** (87.6, 86.6, 86.2) |
| Arm B | 100 | 100 | 63.7 (88, 58, 45) | 65.7 (89, 63, 45) | 58.3 (70, 51, 54) | 77.5 (89.4, 74.4, 68.8) |
| Told MiniGrid's rules (card 160) | 100 | 100 | 100 | 100 | 100 | 100 |

On card 160's own test layouts (not judged) arm A's mean is 88.2% and arm
B's 79.9%; card 160 reported 72.5% for the perceptron there.

| Criterion | Arm A | Arm B |
|---|---|---|
| 1. Mean at least 85% in each experience; no task more than 5 points under card 160's agent | **fail as worded.** The mean passes in every experience. KeyCorridor is 4.6 points under on the mean and 10 under in experience 3 (56 against 66) | fail (two experiences under 85%) |
| 2. UnlockPickup and BlockedUnlockPickup each at least 80% | **pass** (84.7 and 82.7) | fail (63.7 and 65.7) |
| 3. "Will the pickup happen" right in at least 95% of cases, in each experience | **pass** (97.2, 99.4, 97.2; of the cases never tried, 293 of 303, 307 of 309, 301 of 311) | fail (98.1, 87.3, 68.7) |

**Verdict: fail. Arm A meets two of three criteria, arm B none.** My
prediction was wrong on both arms: I expected A to fail criteria 2 and 3
and B to pass all three.

- **Asking the model about the present situation doubled success on the
  pickup tasks** (41.0 → 84.7 and 39.3 → 82.7), with the same network and
  the same experience as card 160. The knowledge was in the network; the
  statement's rule of evidence had hidden it.
- **The condition carries to objects never tried with full hands.** The
  network answers 97 to 99% of such cases right, having met only about 10%
  of them.
- **Arm B is worse, not better.** One shared output for "nothing changes"
  looked best in the gate (38 and 37 of 40) and is best in experience 1.
  In experiences 2 and 3 it answers the pickup question wrongly for 13% and
  31% of cases and loses the pickup tasks again. The gate used experience 1
  alone. I do not know why the shared output generalises less reliably.
- **KeyCorridor is the open problem, and this card does not touch it.**
  It fails in a third of layouts under either agent, with hands empty and
  no plan, as in the data check. Layout by layout, arm A fails 38 that card
  160's agent solves and solves 24 that it fails (in experience 3: 18 and
  8). That is not clearly more than chance, since most steps there are
  random under both agents (70% without a plan).
- **Every remaining failure on the pickup tasks ends the same way** (98 of
  98 for arm A): no plan, a random action, the box destroyed.

Limits: tokens, the full view and the chaining procedure are supplied, as
in card 160. The agent asks about one held token at a time. Card 160's
test layouts had been seen in aggregate before this card; the fresh ones
had not.

### After the test: KeyCorridor explained (2026-11-24; diagnostics on layouts outside every test)

The failures are in the learned effect of the agent's own move, not in
the conditions (`tools/card161/trace.py`, `dev2.py`, `dev3.py`).

- At the first step without a plan the agent wants the key and finds no
  way to it, though only closed, unlocked doors lie between.
- The move effect does not let it walk **west** through an open door of
  some colours: grey and purple in experience 1 (43 of 233 door approaches
  heading west), green, red, grey and purple in experience 3 (103 of 233).
  East, south and north pass for every colour. Random play never walked
  west through an open door of those colours (0 forward moves).
- With MiniGrid's rule in place of the learned move effect, and everything
  else learned, the agent solves 40 of 40 and 38 of 40 (learned: 36 and
  28, experiences 1 and 3).

The cause is how the move effect is built. Each heading has its own
weights, and each neighbour enters as one input per (position, code)
pair. So "an open purple door can be walked through" is learned once per
heading and per colour, and what was learned heading east says nothing
heading west. The role (the cell ahead) is not separated from what fills
it (which door), which is GOAL's P3. Still open: with the rule for the
move, 41% of steps have no plan though the episodes are solved.

## 8. Decision

**Keep arm A's change; drop arm B.** The planner asks the effect model
about the present situation, and the statement's list of tokens to avoid
is removed. The mean rises from 70.3% to 86.8% on fresh layouts. The one
clause it fails concerns a task whose failures have another, unexplained
cause. Next, with the user's go-ahead: the agent holds its expected next
event and notices when it does not come (the user, 2026-11-24), which is
also where "no plan, then a random action" must be replaced; and one
effect model in which a neighbour is what it is and where it lies
relative to the agent, so that a move learned in one heading carries to
the others (KeyCorridor, explained above).
