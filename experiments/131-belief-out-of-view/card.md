---
id: "131"
title: belief out of view — a reading can be absent; absence is a value the goal can ask for, and the last value is kept meanwhile
rung: 3
serves: [P4, P12, P21]
status: abandoned
verdict:
arch_version: og4
date: 2026-11-11
---

# 131: belief out of view

Roadmap step 3 (skills and plans), rung 3. Version og4 reads every
variable afresh from each frame and has no state across time; in
tasks 4 and 5 the goal has the cube inside the closed drawer and og4
scores 0%. The user (2026-11-11): "state across time is definitely a
gap for both architectures and perhaps could be resolved by some sort
of short term belief memory". This is step one of that, agreed as a
small question: can the agent tell "not in view" from "here", and
does a kept value survive a closed drawer?

Biological counterpart: object permanence and working memory. An
infant comes to keep a hidden toy's place in mind (Piaget's A-not-B;
Baillargeon 1987); prefrontal cells hold a seen location through a
delay (Funahashi et al. 1989).

## 1. Question

Can the agent tell from its own read-out when a target is not in
view, keep its last value meanwhile, and treat "not in view" as a
value the goal can ask for, so that it puts the cube away when the
goal frame shows no cube? P4 (memory), P12 (chains), P21.

## 2. What changes

One component: the value of an aimed variable when its reading is
absent.

```
og4:  aim = the read-out's answer, always taken as a position (a hidden cube reads as some
      point 45 cm off, and the agent "moves" the cube toward it)
131:  a reading is valid if the read-out's largest patch score is at least the 5th percentile
      of that score over memory's frames that preceded a touch of the group; else it is absent
      belief  = the last valid reading; kept while the reading is absent; after the agent's own
                move of the believed target, the point where the hand released it
      differs = both valid: as og4 (distance). One valid, one absent: differs. Both absent: same
      recall, goal absent = a remembered move after which the group's reading became absent,
                start nearest the aim now; the path is shifted at the grasp only
      worked  = after the move the reading is absent
```

Nothing names the cube or the drawer. The rule is version 27's ("a
place keeps the last token seen there"), with absence as a value.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A reading carries its own confidence, calibrated on memory | needs a read-out per kind of target | the same | cells are always in view | holds for learned detectors | measured, section 4 |
| An unseen target stays where it was last seen or released | false for animals and falling blocks | false for creatures | holds | false for carried or pushed things | false once: the drawer carries the cube 16 cm (section 4) |
| The goal is a frame, so "not in view" can be asked for | no goal frame | no | no | sometimes | OGBench's goal image |
| Memory holds moves that ended out of view | yes (chests) | no containers | no | yes | 29 in 100 episodes |

## 3. Dependencies

- Cards 128–130 (read-out, one skill, own tries), 127 (planner).
- Version 27's belief rule (MiniGrid line, ARCHITECTURE).
- Confidence from a detector's own score: standard practice; here
  the threshold comes from memory, with no label.

## 4. Data check

Run 2026-11-11 (`tools/card131/check.py`, `stream.py`; each under 3
minutes, capped). The truth only scores.

**Live scenes, arm at rest, 40 each.** Threshold from memory: 5.69.

| Scene | Largest score, median (10th–90th) | Read as valid | Error of the answer |
|---|---|---|---|
| Cube on the table | 6.5 (5.8–7.4) | 93% | 0.7 cm |
| Cube in the closed drawer | 1.0 (0.6–3.4) | 0% | 45 cm |
| Cube in the open drawer | 3.5 (2.7–4.9) | 0% | 25 cm |

**Recorded episodes the read-out never trained on** (set 500–599,
2,032 frames, one before each touch), with the belief rule above
carried along each episode:

| Where the cube truly is | Frames | Read as valid | Belief against the true cube, median | Within 4 cm |
|---|---|---|---|---|
| On the table | 1,667 | 93% (error 0.7 cm) | 0.7 cm | 97.5% |
| In the open drawer | 266 | 0.4% | 4.5 cm | 45% |
| In the closed drawer | 99 | 0% | 11.2 cm | 0% |

- **"Not in view" is told from "here"** with no label: 93% against 0%.
- **Absent means "nowhere I could grasp it", not "hidden".** The cube
  lying in the open drawer is also absent: memory has no grasp from
  the drawer (0 returns to the table in these 100 episodes), so the
  read-out never learned one.
- **The kept value does not survive the drawer closing.** Closing
  carries the cube 16 cm and the belief stays behind (11 cm off).
  One variable's move changing another's value is outside this rule;
  it is what an effect model would learn (card 126's line).
- **Material for recall:** 29 moves in memory ended with the cube in
  the drawer, all with the drawer open (at least 13 cm). They land
  within 4 cm of the benchmark's spot in 48% (30% in the other set);
  the goal frame gives no position to aim the release by.

## 5. Feasibility gate

- **Reading (passed, section 4):** valid in at least 90% of table
  frames and at most 5% of frames with the cube in the drawer: 93%
  and 0.4%.
- **Upper bound for putting away:** each of memory's 29 put-in moves
  replayed live with the true cube position supplying the shift at
  the grasp, drawer open. Not yet run.
- **Trivial baseline:** og4 on the same problems (it takes the goal's
  garbage reading as a table position).

## 6. Success criteria and prediction

Proposed, to be fixed with the user. Self-made problems only; no
benchmark task is run in this card.
1. Put-away problems (cube at random on the table, drawer open at
   the start and in the goal, goal frame with the cube in the
   drawer), 50: the cube ends inside the drawer in at least 70%
   (og4 expected near 0%).
2. Random table problems, 50: at least 90% (og4's family: 96%), so
   that absent readings (7% of table frames) and the belief do no
   harm.
3. The same put-away problems: the agent stops by itself, reading the
   cube as absent, in at least 90% of those it solved (the belief of
   "done" is its own).

**Prediction.** 1 passes if the upper bound does; the risk is a cube
dropped on the drawer's rim. Landing within 4 cm of the benchmark's
spot will be near 40%, which caps task 4 later.

**Not in this card:** task 4 itself. Its drawer starts closed and
locked, so the put-in needs "the drawer is open" as a condition. No
remembered put-in failed, so contrast over memory cannot find it; it
has to come from the agent's own failed try ("not under these
conditions": compare now with the remembered success). That is the
next card. Also not here: the cube riding with the drawer, and
grasping from the drawer.

**Budget.** No training. Gate and three runs of 50 problems, each
under 5 minutes, capped.

## 7. Result

Not run beyond the data check of section 4.

## 8. Decision

**Stop.** The user (2026-11-12) chose to isolate memory as a component
and study it on true visible state over a series of cards, starting
with [card 132](../132-memory-test-bed/card.md). This card's rule (keep the last
value) is one arm there, and its data check stands as evidence.
