---
id: "129.1"
title: two readings chosen by memory — every group of places has an appearance and an aim, each used where memory shows it informative; one skill
rung: 3
serves: [P12, P4, P21, P1]
status: done
verdict: fail
arch_version: og2
date: 2026-11-11
---

# 129.1: two readings chosen by memory

Roadmap step 3 (skills), rung 3.
[Card 129](../129-one-skill-one-kind-of-variable/card.md) showed that a
position cannot be every variable: it fits the cube and fails for a
slide. The user (2026-11-11) agreed the revision: keep both readings,
and let memory decide which is used where, so that no code path names
an object.

Biological counterpart: the dorsal stream (where to reach) and the
ventral stream (what state a thing is in) read the same scene
(Goodale and Milner 1992); which one guides an act depends on the act.

## 1. Question

With groups of places found from memory, and each group's appearance
and aim switched on by a test on memory, does one planner with one
skill match og2 on tasks 1–3? P12, P4, P21, P1.

## 2. What changes

One component: og2's two code paths become one, with two data-driven
switches.

```
og2:    slide places: appearance variable, exact replay      cube: named read-out, shifted replay, moved first
129.1:  groups of places = places joined by remembered moves both ways (card 129)
        per group: appearance, if its touches change its 10 patches at least 5 times the general level
                   aim (the group-asked read-out), if its held-out error is under half that of the
                   touch's place's mean contact point
        differs from the goal = either reading differs; ready groups are taken in random order
        one skill: recorded path + offset (aim now − contact, aim in goal − end); zero without an aim
```

- **A move's end** is where the hand next found the target (the contact
  point of the group's next touch), not where the gripper opened.
- **The aim differs** when now and goal are farther apart than twice
  the read-out's 90th-percentile error on held-out touches (2.4 cm).
- **Conditions** count only for groups that have a reading.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Things that are one variable are linked by moves both ways between their places | holds for carried items | holds | no moves between places | holds | gives cube, drawer, window; buttons alone |
| A reading's worth can be tested on held-out touches | holds | holds | few touches | holds | 800 held-out touches |
| A fixed camera, so a patch maps to a position | no | no | yes | fixed cameras only | declared since card 119 |
| The goal is given as a frame | no | no | no | sometimes | OGBench's goal image |

## 3. Dependencies

Cards 127, 128, 129 (planner, shifted skill, groups and the
group-asked read-out), 125–125.2 (conditions).

## 4. Data check

As card 129: 4,067 touches with a seen change from three sets of 100
episodes; groups and knowledge from set m400.

## 5. Feasibility gate

Run 2026-11-11, with no truth (`know.py --groups`, `aim.py`).

| Group (named afterwards) | Appearance: change against general level (bar 5) | Aim error / place-mean error, median cm | Condition's cause |
|---|---|---|---|
| Window's two places | 7.9: used | 2.0 / 1.3: not used | window's button |
| Drawer's two places | 6.9: used | 2.4 / 1.7: not used | drawer's button |
| Table's three places (cube) | 2.9: not used | 0.7 / 4.8: used | none |
| Each button's place | 2.8, 2.1: not used | 0.5 / 0.4, 0.6 / 0.55: not used | (not planned for) |

The switches come out as og2 had them by hand. Passed.

## 6. Success criteria and prediction

Card 129's, agreed with the user before either card's run:
1. Benchmark tasks 1, 2 and 3 each within 10 points of og2 (84%, 96%,
   86%), 50 episodes each, run once.
2. Random cube problems at least 88% (og2: 98%).
3. One skill path and one comparison with the goal; no line names the
   cube; none fixes the order of variables.

## 7. Result

Run 2026-11-11 (`tools/card129.1/plan.py`). Development on self-made
problems found three faults, fixed before the benchmark: the aim
threshold (Otsu on move lengths gave 11 cm); a move's end taken at the
gripper's release (cube left 3 cm off); and the rest pose, which had
dropped low over the table and knocked the cube 5 cm.

| | Result | Bar | og2 |
|---|---|---|---|
| 2. Random cube problems (50 fresh) | 96% | at least 88% | 98% |
| 1. Task 1 | 86% | at least 74% | 84% |
| 1. Task 2 | 54% | at least 86% | 96% |
| 1. Task 3 | 32% | at least 76% | 86% |
| 3. Code | holds | | |

**Criterion 1 fails** on tasks 2 and 3; 2 and 3 pass. Tasks 4 and 5:
0%, as og2.

**Why** (truth for scoring only): in 41 of the 48 stopped episodes on
tasks 2 and 3 a window was left 6–15 cm from closed, in 6 a drawer
4–5 cm short, and the planner read them as at the goal. With a
slide's two places merged into one group, the group has one set of 10
appearance patches where og1 had one set per place, and a half-moved
slide falls between them.

**Diagnostic after the run, self-made problems only, not judged**
(`--places 1`: appearance and conditions per place as in og1, groups
only for the aim; 50 problems each):

| Problems | Appearance per group | Appearance per place |
|---|---|---|
| Locks and slides | 72% | 94% |
| Locks, slides and cube | 84% | 84% |
| Cube only (30) | 100% (20 earlier) | 97% |

## 8. Decision

**Revise.** Kept: groups from two-way moves, the two switches, one
skill, a move's end as where the hand next found the target; with
them the cube needs no name (96%). Failed: reading a slide's
appearance once per group. Revision (129.2, for the user): appearance
and conditions stay per place, groups decide only which touches share
an aim; the diagnostic supports it on self-made problems. The
benchmark has now been seen once by this planner, so 129.2's run of it
is a second look and must say so. og2 stays the current version.
