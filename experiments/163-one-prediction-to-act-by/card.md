---
id: "163"
title: one prediction to act by — will the expected event come, learned from play, in place of the planner's procedures
rung: 5
serves: [P12, P21, P6, P16]
status: approved
verdict:
arch_version: og4
date: 2026-11-24
---

# 163: one prediction to act by

MiniGrid, the agent of
[card 162](../162-a-neighbour-is-what-it-is-and-where-it-lies/card.md). It
learns what actions do from reward-free random play and reaches a goal by
a hand-written procedure: want an effect, then its needs, then clear what
blocks it, then walk by a shortest-path search over the learned move, and
take a random action when no plan is found. Almost every failure left
(93 of 94 on the pickup tasks) ends in that random action.

The user's rule (2026-11-24): the agent learns how to orient itself, the
tree of wants is not a fixed structure, and there is no algorithm with
modes for walking, standing at an object or failing. Walking toward a key,
picking it up and recovering from a failed try are one question:
**given this situation and this action, will the expected event come?**

Biological counterpart: the hippocampal successor representation, a
learned prediction of what is to come from each place (Stachenfeld et al.
2017); dopamine that ramps up as a goal nears (Howe et al. 2013); the
reward-prediction error that teaches both (Schultz et al. 1997). All are
learned predictions of an upcoming event, not counts of steps.

## 1. Question

Can one learned prediction, "will this event come, and how soon, if I take
this action", choose every action of the agent, in place of the
shortest-path search, the fixed order of steps and the random action, and
what experience does it need to learn that? Rung 5 (Roadmap step 3: a
learned closeness from reward-free play by hindsight). P12 (a learned
sense of closeness; subgoals from conditions), P21, P6, P16.

## 2. What changes

One component: how an action is chosen. Tokens, memory, the goal (events
at 20 example success steps) and the experience are card 162's.

```
162:  want an effect -> its needs -> what blocks it -> walk (shortest path over the learned move) -> action;
      no plan -> a random action
163:  q(situation, action; event) in [0, 1]: 1 if the action brings the event now, lower the later it comes.
      Act: the action with the highest q for the expected event.
      The expected event: the goal itself, or the known event E' for which
          q(E' comes) x q(the goal comes, in the situation E' leaves)   is highest.
```

- **One network, card 162's form.** Every token is what it is and where
  it lies; attention combines them. Asked about a token and a next code
  ("this key goes from view"), it answers for each action.
- **Learned by the rule that learns a tick.** At the step where the event
  happens the target is 1. At any other step it is a fixed fraction of
  the best prediction one step later. A next-tick effect and an effect
  many steps away are the same output at different heights. Every step
  of play teaches about every token on the map, not only the one touched.
- **"Getting closer" is the prediction rising.** No route is computed.
- **The tree is what the model answers.** A chain's prediction is the
  product of its links, so "pick up the key, then the door opens" is
  compared with "the door opens" on one scale. What an event leaves (the
  token's new code, and the old one gone from view) is imagined whole, as
  cards 160–162 already do. Nothing fixes the order of steps.
- **One rule stays, declared:** an action that changed nothing is not
  repeated in the same situation; the next best is taken. This is the
  agent's own failed try correcting the prediction.

Declared (C3): an event is a supplied token changing code, or the agent
arriving on a token (learned events are a later card); chains are compared
to three links; the fraction per step is fixed at 0.95.

**Experience, in the order the user set (2026-11-24).** The same learner
at each stage; a stage is tried only if the one before fails the gate.

| Stage | Experience | Counterpart |
|---|---|---|
| R | Card 160's 700,000 random steps, kept in order | free play |
| S | Structured play: an episode starts at a checkpoint, a situation part-way along a solved episode of a play world, and random play follows; same number of steps | a parent puts the key in the child's hand at the door (scaffolding, Wood, Bruner and Ross 1976) |
| D | Stage R or S with whole solved episodes added | learning by watching (P16) |

The checkpoints and solved episodes come from the agent that is told
MiniGrid's rules, in the play worlds, on layouts never tested. Two play
worlds (UnlockPickup, BlockedUnlockPickup) are also test tasks, so stages
S and D would show those two tasks solved on other layouts; DoorKey-8x8,
DoorKey-16x16 and KeyCorridor are never shown. This is my reading of
"structured play with checkpoints", to be confirmed with the user before
stage S is used. Approved by the user, 2026-11-24.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds | needs a front end | declared, as in card 160 |
| An event is a token changing code | needs learned events | the same | holds | needs learned events | declared; the next card on the user's account |
| Where a token lies is an offset vector | holds, in three dimensions | holds | holds | holds | general: this card replaces card 162's table of grid offsets by a function of the vector |
| The whole map is in view | no | no | holds | no | declared, as in card 160 |
| A known event can be imagined whole | holds for learned effects | holds | holds | approximately | general |
| Checkpoints can be set (stage S) | a saved world | a saved state | a reset to a level | a person arranges the scene | declared for stage S |

## 3. Dependencies

- Cards 160–162: tokens, memory, the three experiences, the goal
  examples, the network form, the rules agent (evaluator).
- Learning a prediction by bootstrapping from its own next value, with
  every later event as a target in hindsight: fitted Q iteration (Ernst
  et al. 2005), hindsight relabelling (Andrychowicz et al. 2017),
  predictions of many events at once (Sutton et al. 2011). Card 097.1
  did this for walking alone and agreed with the exact field on 99.0 to
  99.5% of decisions.
- Two lessons to respect: a learned closeness that crept up on
  unreachable states (card 012), and "trust a learned distance only
  within the reach its input shows" (card 097.1).

## 4. Data check

Experience 1, replayed in order (`tools/card163/seq.py`; its counts equal
card 160's exactly): 700,000 steps, 7,886 episodes, 14,778 events.

| What the prediction must learn from | In random play |
|---|---|
| Steps that teach walking toward a token | every step, for every token on the map |
| Events reached from one cell away, 2–3, 4–6, 7 or more (where the stretch began) | 10,871, 3,512, 394, 1 |
| Key picked up, then its locked door opened | 298 |
| Locked door opened, then a box or ball picked up beyond it | 0 (a box toggled away: 4) |
| Episodes with three or more events | 1,302 of 7,886 |

Random play walks to an event from across a room almost never, and the
last link of UnlockPickup, BlockedUnlockPickup and KeyCorridor is absent.
Bootstrapping can join short pieces into a long walk; it cannot supply a
link never played. I expect stage R to learn the walk and the first
links, and stage S or D to be needed for the rest.

## 5. Feasibility gate

On experience 1, layouts outside every test (seeds 2000–2039), the play
worlds' own tasks first (DoorKey-5x5, 6x6, Unlock), then the five tests.

- **Upper bound:** the same network and acting rule, fitted on stage D's
  experience. If it cannot solve the play worlds' tasks with solved
  episodes in its experience, the form is at fault, not the data.
- **Trivial baseline:** uniform random actions (card 160: DoorKey-5x5
  9%, the tests 0 to 1%).
- **Checked at each stage:** does the prediction rise along a walk to a
  key from across the room, in rooms of play's sizes and in DoorKey-16x16
  (larger than any played); does it stay low for a token behind a locked
  door; is the first expected event the right one.

Result of the gate, before the main run:

**Stage R (2026-11-24, experience 1, 40 layouts per task, seeds
2000–2039).** One fit, 40,000 steps, 9 minutes (`tools/card163/q.py`).

The prediction along episodes of the agent told MiniGrid's rules, for the
event that comes next (DoorKey-16x16, rooms larger than any played):

| Steps to the event | 0 | 1–2 | 3–5 | 6–10 | 11–20 | over 20 |
|---|---|---|---|---|---|---|
| Mean prediction | 1.00 | 0.91 | 0.74 | 0.56 | 0.36 | 0.23 |
| Best action is the told agent's | 100% | 100% | 100% | 76% | 70% | 71% |

It rises as the event nears. The goal's own prediction also rises after
each event on the way (DoorKey-8x8: 0.35 at the start, 0.42 with the key
held, 0.65 with the door open): bootstrapping joins links that random
play holds only apart.

The agent, solved of 40 (mean steps when solved):

| | DoorKey-5x5 | DoorKey-6x6 | Unlock | DoorKey-8x8 | DoorKey-16x16 | UnlockPickup | BlockedUnlockPickup | KeyCorridor |
|---|---|---|---|---|---|---|---|---|
| Asking about the goal only | 40 (10) | 40 (13) | 40 (10) | 40 (23) | 36 (1190) | 0 | 0 | 2 |
| Comparing chains of three links | 40 (12) | 40 (19) | 40 (11) | 40 (39) | 36 (974) | 4 | 0 | 8 |

- **Asking about the goal alone is enough where random play holds the
  links,** and takes fewer steps than comparing chains. The prediction
  already carries the chain. Comparing chains by imagining an event
  whole is dropped unless a later stage needs it.
- **Stage R fails the three tasks whose last link random play never
  holds,** as the data check said, and wanders in DoorKey-16x16 (solved,
  in about 1,200 steps): beyond six steps the best action is right 70% of
  the time.
- Stage R does not pass the gate for the five tasks.

**Stages S and D (2026-11-25, experience 1, the same 40 layouts; the user
confirmed the reading of "checkpoints").** Stage S: 951 checkpoint
episodes, 249,364 of the 700,000 steps; random play now picks up a box
292 times (0 in stage R). Stage D adds 2,500 recorded solutions (40,198
steps, all solved). The agent asks about the goal only. Solved of 40
(mean steps):

| | DoorKey-5x5 | DoorKey-6x6 | Unlock | DoorKey-8x8 | UnlockPickup | BlockedUnlockPickup |
|---|---|---|---|---|---|---|
| Stage S | 40 (10) | 40 (16) | 38 (24) | 40 (27) | 0 | 0 |
| Stage S, chains of three links | | | | | 0 | |
| Stage D, the upper bound (fit stopped at 36,555 of 40,000 steps) | | | | 40 (16) | 0 | |
| Stage S, double Q-learning | | | 40 (10) | 40 (46) | 0 | |

- **The upper bound fails.** With whole solutions of UnlockPickup in its
  experience the agent still solves 0 of 40. By this gate's own rule the
  learner is at fault, not the experience.
- **Why.** In stage S the goal's prediction is about right on average
  along a solution (0.37 at 16–19 steps from the goal, against 0.41 for
  0.95 per step), but the best action leads the second by only 0.011
  there, less than the fit's error, and it is the rules agent's action
  in 35 to 50% of steps. The agent picks the key up and puts it down
  without heading for the door. Within five steps of an event the gap
  is 0.03 and the action is right.
- **One explanation tested and rejected:** that taking the best next
  prediction inflates far ones (Stage D's are inflated: 0.61 at 16–19
  steps). Double Q-learning (van Hasselt 2010) removes that bias and
  solves no more (`q.py`, stage name `S+dq`).
- **A bound that passes: the next event handed over.** The agent is told
  the next event of the rules agent's solution and acts on the same
  stage S prediction for it (`dev4.py`):

| | DoorKey-8x8 | DoorKey-16x16 | KeyCorridor | UnlockPickup | BlockedUnlockPickup |
|---|---|---|---|---|---|
| Solved of 40 | 40 | 40 | 39 | 24 | 23 |
| Mean steps (the rules agent) | 15.8 (15.8) | 68 (33) | 31 (22) | 21.1 (21.1) | 76 (30) |

  Walking to a near event and bringing it about are learned, in rooms
  larger than any played and in a task never shown. What is not learned
  is which event to expect next when the goal is far.
- **A fault in the event's definition.** Every UnlockPickup failure
  inspected under this bound (12) reached all four events: the agent
  toggled the box, which also turns it into empty floor, instead of
  picking it up. "A token changing code" does not say by which action;
  cards 160–162's goal did.
- **The gate does not pass at any stage.** The main run is not made.

## 6. Success criteria and prediction

Test: card 160's five tasks, 100 fresh layouts each (seeds 7000–7099,
never run), three experiences, code and networks frozen before the run,
at the first stage that passes the gate. Comparison: card 162's agent on
the same layouts.

1. **The walk is learned.** While the agent pursues an event within
   reach, the action it takes is one MiniGrid's rules rate as closest in
   at least 95% of steps, per task, on the mean of three experiences.
2. **The tree comes from the model.** In at least 90% of layouts per
   task, the first event the agent expects is the first event of a
   solution by MiniGrid's rules.
3. **Tasks, with a declared loss.** This replaces hand-written
   procedures by one learned form, so a loss is allowed (AGENTS, general
   forms): mean success over the five tasks at most 10 points under card
   162's agent (92.5%), and DoorKey-8x8 at least 95%.

Prediction: stage R passes criterion 1 at play's room sizes and is
doubtful in DoorKey-16x16; it fails criteria 2 and 3 on the three tasks
whose last link random play never holds. With stage S or D, 1 and 2
pass; 3 is uncertain, and BlockedUnlockPickup (five events in order) is
the likeliest to fall short.

Budget: one fit per experience and stage on the GPU, in resumable parts
of under 10 minutes, at most 30 minutes each; gate on experience 1 first,
about one hour; test about 30 minutes. Stages S and D need new play,
about 90 seconds per experience.

## 7. Result

## 8. Decision
