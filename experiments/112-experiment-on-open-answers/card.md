---
id: "112"
title: experiment on open answers — try the action where memory's answer is open
rung: 1
serves: [P15, P12, P10, P5]
status: done
verdict: fail
arch_version: 27
date: 2026-10-18
---

# 112: experiment on open answers

Roadmap step 2, its experiment part. Drafted at the user's request,
2026-10-18. Memory's part is built (cards 111–111.2): memory finds the
conditions it holds the evidence for, and marks the rest open. This card
asks whether the agent's own tries, aimed at those open answers, close
them faster than random play does (GOAL P15). Development world: breaking
world 1, whose rooms the simulator can play live. The Crafter follow-up
(112.C) needs skills to set up a try (a table, materials), so it waits
for rung 3.

## 1. Question

At an equal number of steps in new rooms, does trying actions where
memory's answer is open find more of the conditions of lamps lighting and
going out than random play? Rung 1. Serves P15, P12, P10 and P5.

## 2. What changes

One component: the exploration policy. Memory's contrast is 111.2's,
unchanged ([tools/card111.2/contrast.py](../../tools/card111.2/contrast.py)).

```
random:    each step, an action at random (card 102's policy)
directed:  questions: whenever a token changes code during play (a, c0 -> c1), memory is asked for the
           conditions (111.2). An open answer becomes a question: its kept facts, anchored on the token
           each step: if some token with code c0 satisfies a question's kept facts and the question's action
           is a, take a (the question with the fewest remembered cases first); otherwise act at random
           every 500 steps: memory is rebuilt with the new steps; every question is asked again and
           dropped once closed
```

- **Changed after the smoke test (2026-10-18), before the main run.** A
  question isn't tried again in a state it was already tried in. A
  repeated try adds no new case. Without this rule, the agent spent 1,946
  of 3,000 steps repeating tries, for example walking into the same wall,
  and saw 8 lamp events (21 with the rule).
- **The try is opportunistic.** The agent tries the action when the
  present already matches an open answer. It does not act to bring such a
  state about. That needs acting toward a condition (rung 4, not built).
  This is the weakest form of experiment, and the card measures whether
  even it beats random play.
- **Cost rule.** A kind of change whose last 20 answers were all closed
  is asked only 1 time in 20. This is general, and it keeps moves from
  dominating.
- **No prior** about lamps, boxes or walls. Questions and matches are
  facts over codes at offsets, as in 111.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A question's state recurs during play often enough to be tried | yes (open world, long play) | yes | levels repeat | yes | declared; deliberately setting up a state is rung 4 |
| Discrete actions, one step | yes | yes | yes | needs a kernel | as 111 |
| Memory can be rebuilt with new steps | yes | yes | yes | yes | general |

## 3. Dependencies

- **Card 111.2:** the contrast, closing on 29 cases, the scoring and the
  fresh test rooms. Its decision was revise (scoring only). This card
  carries that fix: the truth also checks the room's border.
- **Card 102:** world 1's room generator and random policy.
- **Literature:**
  - Settles 2009, active learning (uncertainty sampling);
  - Schmidhuber 1991 and Pathak et al. 2017: curiosity aimed at what the
    model can't yet predict.
  Both are added to LITERATURE.

## 4. Data check

- **Exploration rooms:** card 102's test-room generator (lamps in
  columns 5–7, never seen in training), seed 1113. These are new rooms,
  different from 111.2's evaluation rooms (seed 1112).
  - Random play there lights a lamp in about 1 step in 560 (111.2's
    fresh rooms: 355 in 200,000).
- **Evaluation:** 111.2's events (300 lamps lighting, 264 going out, 300
  each of push, move and blocked), against memory after exploration.
- **Memory before exploration:** card 102's training set (300,000 steps).
- **Truth:** 111.2's, with the room's border added to the check for
  impossible states.

## 5. Feasibility gate

- **Upper bound:**
  - memory after a random exploration 5 times longer (50,000 steps);
  - memory over all of 111.2's evaluation rooms themselves (200,000
    steps; the events' own rooms, scoring only).
  If neither lifts lamp recall, more experience in new rooms doesn't help,
  and no exploration policy can pass.
- **Baseline:** no exploration (111.2's memory, rescored with the border).

## 6. Success criteria and prediction

Exploration budget: 10,000 steps (100 episodes of 100 steps). Three seeds
per arm; the directed and random arms share each seed's rooms.

1. **P15: directed beats random.** Recall on world 1's lamps (lighting and
   going out together) is higher with the directed arm than the random
   arm, in all three seeds, and by at least 0.03 on the mean. It's also
   reported at 2,500 and 5,000 steps.
2. **No loss.** Precision ≥ 0.9 on every kind, and pushes, moves and
   blocked moves keep recall ≥ 0.98, in both arms.
3. **Closed still means complete:** ≥ 0.9 of closed answers, every kind
   with at least 20 closed.

**Also reported:**
- the questions raised and closed;
- the tries the directed arm made, and how many matched a question;
- lamp events seen during exploration, per arm;
- the share of answers closed.

**Prediction:**
- **Random play in the new rooms already raises lamp recall,** since
  layouts with lamps in columns 5–7 enter memory. Expect 0.65 → about
  0.75 at 10,000 steps.
- **The directed arm tries the matching action every time.** Random play
  tries it 1 time in 5, so the directed arm adds up to 5 times as many
  cases to open questions. Expect +0.03 to +0.08. That's uncertain: it
  depends on how often the present matches a question.
- **Criteria 2 and 3 hold**, since the method is unchanged.

**Decision rules:**
- **Keep** (experiment becomes step 2's second source, and the
  directed-try policy stays) if 1–3 hold.
- **Revise** if 1 fails but the directed arm's tries matched questions
  rarely (fewer than 1 step in 100). The trying is then fine, and setting
  up states is what's missing; that is rung 4's skill, named as the next
  card.
- **Stop** if 1 fails while matches were frequent. Trying where memory is
  open then doesn't add information.

**Budget:** no training.
- Each run is 10,000 steps, about 20 rebuilds of memory, and questions
  asked under the cost rule.
- Expect 2–4 minutes per run. That's 6 runs plus the 2 gate runs: under
  30 minutes, run as a capped chain after a 1,000-step smoke test.
- If the smoke test projects more than 30 minutes, the budget falls to
  5,000 steps, and the card says so.

## 7. Result

Run 2026-10-18 ([results.json](results.json); `runs/112/`). Every run was
capped, as two chains in parallel.
- **Time:** directed runs took 391–436 s, random 59–63 s.
- **Truth:** 111.2's, with the room's border added.
- **Recall on lamps** combines lighting (300 events) and going out (264).

| Arm (seed) | Lamps recall at 2,500 / 5,000 / 10,000 steps | Lamp events seen | Tries matching a question | Precision, lamps lighting / going out | Closed and complete, lamps lighting |
|---|---|---|---|---|---|
| none | 0.614 | | | 0.94 / 0.99 | 1.0 (57 closed) |
| random (0) | 0.629 / 0.627 / 0.647 | 30 | | **0.81** / 0.98 | 1.0 (57) |
| random (1) | 0.614 / 0.628 / 0.634 | 32 | | 0.96 / 0.98 | 1.0 (57) |
| random (2) | 0.609 / 0.607 / 0.640 | 23 | | **0.89** / 0.98 | 1.0 (57) |
| directed (0) | 0.642 / 0.640 / **0.662** | 66 | 941 | 0.90 / 0.98 | 1.0 (57) |
| directed (1) | 0.614 / 0.612 / **0.629** | 38 | 1,117 | 0.94 / 0.99 | 1.0 (57) |
| directed (2) | 0.634 / 0.643 / **0.662** | 34 | 203 | 0.97 / 0.98 | 1.0 (69) |
| gate: random, 50,000 steps | 0.684 | 140 | | 0.99 / 0.98 | **0.51** (111) |
| gate: the evaluation rooms themselves | 0.841 | | | 0.93 / 0.97 | **0.54** (244) |

Pushes, moves and blocked moves: 1.0 / 1.0 in every arm.

- **Gate: pass.** More experience in new rooms lifts lamp recall: 0.68
  at 50,000 steps, and 0.84 with the events' own rooms.
- **Criterion 1, directed beats random: fail.**
  - The gap at 10,000 steps is +0.011 on the mean, against 0.03 required.
  - Directed won in 2 of 3 seeds: +0.014, −0.005, +0.022.
  - It saw more lamp events (46 against 28 on average), but they added
    little.
- **Criterion 2, no loss: fail.** Random play's precision on lamps
  lighting fell to 0.81 and 0.89. The directed arm kept 0.90–0.97. More
  cases make weak facts about the lamp's place significant.
- **Criterion 3, closed means complete: pass** at 10,000 steps (1.0). But
  the gate runs show the closing rule breaks as memory grows:
  - lamps lighting close more often (111 and 244 events);
  - only 0.51 and 0.54 of those are complete.
  - A closed answer is complete only with respect to the facts its cases
    varied. A condition that never varied among 29 successes, such as the
    box's destination being floor, stays invisible however many such
    cases memory holds.
- **Questions:** 12–18 were raised per run, and 1 was closed.

## 8. Decision

**Stop** (2026-10-18), by the rule: criterion 1 fails while tries matched
questions often (203–1,117 per 10,000 steps, 2–11% of steps). Trying where
memory is open, opportunistically, adds too little.
- **Why, from the result.** A try that matches a question's kept facts
  repeats the situation memory already has, with the missing fact at its
  usual value. The open conditions need the opposite: a case that differs
  in the missing fact (the box's way blocked, a second box beside the
  lamp). A random walk rarely makes one, whoever chooses the action.
- **What stays:**
  - experiment as a source (CHARTER);
  - questions from open answers;
  - not repeating a try in the same state.
- **Next, for the user to choose:**
  1. **Experiments that set up the contrasting state.** The agent acts to
     change the near-miss fact before trying. That needs acting toward a
     condition (rung 4), so the order of the ladder would change.
  2. **Closing that knows what never varied.** An answer stays open while
     a fact of e never varied among its cases. That's P5, inside memory's
     share, and the gate runs show it's needed (0.51–0.54). The hard part:
     in a small set of cases almost every far cell never varies, so the
     unvaried facts must be ranked, not all doubted.
  3. **Demonstration (P16).** Another agent's successes and failures
     supply the contrasting cases.
