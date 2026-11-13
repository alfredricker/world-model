---
id: "129.2"
title: appearance per place, aim per group — card 129.1's planner with a slide read at each of its places
rung: 3
serves: [P12, P4, P21, P1]
status: done
verdict: fail
arch_version: og3
date: 2026-11-11
---

# 129.2: appearance per place, aim per group

Roadmap step 3 (skills), rung 3.
[Card 129.1](../129.1-two-readings-chosen-by-memory/card.md) unified
og2's two code paths and lost the slides, because a slide's two places
shared one set of appearance patches. The user (2026-11-11, "yes")
approved this fix with the same pass marks.

Biological counterpart: as card 129.1 (where to reach and what state a
thing is in are read separately).

## 1. Question

With appearance and conditions read per place, as in og1, and groups
used only to say which touches share an aim, does the one planner
with one skill match og2 on tasks 1–3? P12, P4, P21, P1.

## 2. What changes

One component, against card 129.1: the unit that carries appearance.

```
129.1:  a group of places has one appearance variable (10 patches) and one condition
129.2:  each place keeps its own appearance variable and condition (card 127's knowledge);
        a group only says which touches are moves of the same target, for the aim and for recall
```

Everything else is card 129.1's: the switches, the one skill, a move's
end as where the hand next found the target, ready variables taken in
random order. Run as `tools/card129.1/plan.py --places 1`.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

As card 129.1; nothing new is assumed.

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| A thing's appearance is read where the hand meets it, place by place | needs patches placed in the world | the same | holds | fixed cameras only | two places per slide |

## 3. Dependencies

Cards 127, 128, 129, 129.1.

## 4. Data check

As card 129.1.

## 5. Feasibility gate

Card 129.1's diagnostic on self-made problems (50 each): locks and
slides 94% (per group: 72%), with the cube as well 84%, cube only 97%.
Passed.

## 6. Success criteria and prediction

Card 129's, unchanged:
1. Benchmark tasks 1, 2 and 3 each within 10 points of og2 (84%, 96%,
   86%), 50 episodes each.
2. Random cube problems at least 88%.
3. One skill path and one comparison with the goal; no line names the
   cube; none fixes the order of variables.

**Second look.** Card 129.1's planner has already been run on the
benchmark tasks, and this fix was chosen after seeing why it failed
there. The fix itself was tried only on self-made problems.

## 7. Result

Run 2026-11-11, once, no changes after the diagnostic.

| | Result | Bar | og2 | 129.1 |
|---|---|---|---|---|
| 2. Random cube problems (50 fresh) | 96% | at least 88% | 98% | 96% |
| 1. Task 1 | 84% | at least 74% | 84% | 86% |
| 1. Task 2 | 90% | at least 86% | 96% | 54% |
| 1. Task 3 | 72% | at least 76% | 86% | 32% |
| 3. Code | holds | | | |
| All five tasks (4 and 5: 0%) | 49.2% | | 53.2% | 34.4% |

**Criterion 1 fails on task 3 by 4 points** (two episodes of 50);
tasks 1 and 2 are inside the margin; criteria 2 and 3 pass. With 50
episodes a rate is uncertain by about 6 points, so 72% against 86% is
not clearly a real loss, and not clearly chance.

Task 3's 14 failures (truth for scoring only): 8 ran out of steps
retrying the drawer, 4 stopped with the drawer 4–5 cm short of open,
2 with the window 9 cm from closed. The cube was within 4 cm in 12 of
the 14. These are og1's slide failures, not the unification's. One
difference from og2 that may cost steps: a ready variable is taken at
random, where og2 took the best-matching pull; solved episodes took
398 steps on average against og2's 276.

## 8. Decision

**Keep** (the user, 2026-11-11: "let's adopt the unified planner").
By the rule set before the run the planner missed criterion 1 on task
3 by 4 points (verdict: fail), and the user adopted it as version og3
with that loss stated, under the exception for general forms: task 3
is 14 points under og2 and task 2 6 under, to be won back. What it
shows: no code needs to name the cube; groups, switches and one skill
come from memory. What costs the points is the slide skill it
inherits: opening the drawer works in about a third of tries (card
127) and uses up task 3's steps. Not tested: more than one seed;
whether random order or chance accounts for the 14 points.
