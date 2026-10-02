---
id: "046"
title: what can be made walkable, read at recall's two levels
rung: 0
serves: [P17, P6, P21, C1]
status: done
verdict: pass
arch_version: 7
date: 2026-10-01
---

# 046: what can be made walkable, read at recall's two levels

## 1. Question

[Card 045](../045-movement-through-conditions/card.md) made walking a
learned approach plus its conditions, and passed. But when no clear way
exists, it tries every token recall calls "openable" as a condition ("make
this tile walkable"). Recall judges openable by similarity alone: a thing
counts if memory has made walkable any thing whose vector is close to it.
In seeds 401 and 404 the wall is close enough to a door. So each wall
tile became a condition to try. The key world then planned at 27 s per
layout (seed 400: 0.37), and card 045 needed a filter to bring that down
to 2.9 s (card 044: 1.1). The user called the filter a patch (2026-10-01).

Recall's other judgements already read two levels (card 042): tries on
the same thing (equal codes) first, things like it only when there are
none. Does reading openable the same way remove the slowdown, with the
filter gone and nothing else lost? Before rung 1. Serves:
- P17, planning cost;
- P6, learned effects read the same way everywhere;
- P21, obstacles as conditions;
- C1, what can be opened learned, not supplied.

## 2. What changes

One component: recall's judgement of what can be made walkable (card
038's `World.openable`). Card 045's filter on door candidates is removed.

```
card 045:  openable(u) = memory holds a pick up or toggle that made a thing like u walkable
                         (similarity k >= KMIN on the similar-thing level)
           filter: only tokens a route alone-blocked by them could use
card 046:  per pick up and toggle:
             memory holds tries on the same thing as u (equal codes)
                 -> openable if one of those tries made it walkable
             it holds none -> card 045's rule (things like u)
           no filter: every token judged openable is tried
```

- **As recall's templates do** (card 042): what a thing does is read from
  the thing itself when memory has tried it. Similarity speaks only for a
  thing never tried, such as a door in a new colour.
- **Declared exceptions** (CHARTER rule 8), card 045's: the route
  procedure; placing the view and recall's appearance set (card 044).
- **Unchanged:** card 045's walking (System 1 approach, route conditions,
  doors as conditions, waypoints), card 044's tokens, card 043's planner,
  card 042's recall otherwise, arm A's encoders (seeds 400–404), the data.

## 3. Dependencies

- Card 045 (pass, revise): walking through conditions; its criteria and
  results are the comparison.
- Card 042's same-thing level: one code tuple per stored try
  (`code_id`); a tuple with a "new" piece matches only its own vector.
- Card 038's similar-thing rule, kept as the fallback.

## 4. Data check

Done (`tools/card046/openable.py --check`, `runs/046_openable_check.json`,
key world, seeds 399–404, 3 minutes). Memory's distinct stored tries on
the same thing, pick up / toggle: wall 153 / 150, goal 16 / 14, switch
off 44 / 60, switch on 53 / 64, green door 23 / 24, green key 58 / 53,
vase 39 / 55. None made the wall, the goal or the switch walkable.

| Thing | Card 045's rule (similarity) | Card 046's rule |
|---|---|---|
| closed doors, keys (three colours), vase | openable, all six seeds | openable, all six seeds |
| switch, off or on | openable, all six seeds | not openable |
| wall | openable in seeds 401 and 404 | not openable |
| goal | openable in seed 404 | not openable |

New colours (the yellow and purple tests) were never tried, so they are
judged by similarity, as in card 045.

## 5. Feasibility gate

- **Upper bound:** card 045 with its filter (all criteria passed, 5 of 5
  seeds; key world 2.9 s per layout in seeds 401 and 404, 0.37 in the
  others).
- **Trivial baseline:** card 045 without the filter (seed 401's key world:
  27 s per layout).
- **Mechanism:** section 4; the switch, wall and goal stop being
  candidates, and nothing that can be opened is lost.
- **Shakedown:** spare seed 399, 30 layouts per world and per unseen room,
  against card 045's shakedown on the same layouts; and seed 401's key
  world, 30 layouts, for time.

Result of the gate, after building (`tools/card046/openable.py`):
- **Mechanism:** section 4's table.
- **Shakedown** (`runs/046_shakedown_399.json`, 3 minutes): identical to
  card 045's shakedown in all 20 tests (four familiar worlds, four
  new-colour tests, twelve unseen rooms). Goal, moves, kinds of steps,
  refusals and wrong predictions all match. Time per layout in the
  familiar worlds: 0.41–0.65 s, against 0.54–0.71 for card 045's
  shakedown (before its speed changes).
- **Seed 401's key world, without the filter** (`runs/046_dev_401_key.json`,
  30 layouts): 0.615 s per layout, against 0.603 for seed 400 on the same
  layouts (`runs/046_dev_400_key.json`). Card 045 without the filter took
  27 s per layout there in its first main run, and 2.9 with it.

## 6. Success criteria and prediction

Arm A, seeds 400–404; each criterion in at least 4 of 5 seeds.

1. **Nothing lost.** Card 045's three criteria: familiar worlds ≥ 99%
   with mean steps within 5% of card 029's, and the key world with yellow
   and purple keys ≥ 99%; unseen rooms ≥ 99% at most 1.15 times the
   shortest route; how-soon rank correlation ≥ 0.95.
2. **No seed-specific slowdown.** In each familiar world, every seed's
   time per layout within 10% of the five seeds' median. Card 045 failed
   this in seeds 401 and 404's key world (7.8–7.9 times the median),
   and card 044 did too (2.9 times).

Reported: decisions against card 045's (goal, moves, kinds of steps,
refusals); time per layout against cards 045 and 044; the new-colour
tests; walking's share of planning time.

**Prediction.** Both pass. Decisions match card 045's, since a token that
can never be made walkable was tried and failed there. Seeds 401 and 404's
key world falls to about 0.37 s per layout. The switch world's new-colour
tests in seed 403 come back to card 044's time (12–13 s), since the switch
no longer counts as a door.

**Budget.** Shakedown about 5 minutes. Main run: card 045's took 28
minutes, about 6 of them in the two slow key worlds, so about 22 minutes.
If over 30 minutes, the main run goes to the user.

## 7. Result

Main run, arm A, seeds 400–404 (`runs/046_armA.json`, 20 minutes; card
045's took 28). Compact numbers are in [results.json](results.json).

| Seed | Familiar, four worlds | Key world, yellow / purple | Unseen rooms, 12 | How soon | Slowest world, time over the median | Criteria 1 / 2 |
|---|---|---|---|---|---|---|
| 400 | 100% | 100% / 100% | 100% | 1.0 | 1.000 | pass / pass |
| 401 | 100% | 100% / 100% | 100% | 1.0 | 1.000 | pass / pass |
| 402 | 100% | 100% / 100% | 100% | 1.0 | 0.997 | pass / pass |
| 403 | 100% | 100% / 100% | 100% | 1.0 | 1.034 | pass / pass |
| 404 | 100% | 100% / 100% | 100% | 1.0 | 1.025 | pass / pass |

- **Criterion 1: pass, 5 of 5.** Card 045's three criteria, with the same
  numbers: mean steps 0.992–0.997 of card 029's, unseen rooms 1.01–1.11
  times the shortest route, rank correlation 1.0.
- **Criterion 2: pass, 5 of 5.** Every seed within 3.4% of the median in
  every familiar world. Seeds 401 and 404's key world: 0.355 and 0.363 s
  per layout, against card 045's 2.92 and 2.95 and card 044's 1.07 and
  1.09. Every other seed and world is 1–7% faster than card 045's.
- **Reported:**
  - decisions: 95 of 100 tests (4 familiar worlds, 4 colour tests, 12
    unseen rooms, 5 seeds) match card 045's exactly in goal, moves, kinds
    of steps, refusals and wrong predictions;
  - the 5 that differ are all the switch world with a new-colour door
    (reported only), and all are worse. Goal reached, card 045 → 046:
    seed 400 yellow 54% → 13% and purple 51% → 13%; seed 402 yellow
    16% → 10% and purple 59% → 13%; seed 404 purple 74% → 68%;
  - the cause, traced on seed 400's yellow test, layouts 0–11: the agent's
    first plan toggles the new door and fails, since the switch is off.
    Memory now holds a try on that very door, so this card's rule reads
    only that try and no longer counts the door openable. After the
    switch comes on (by a random toggle), the agent never tries the door
    again. Card 045's rule kept the door openable as like other doors, and
    recall then predicted the toggle would work with the switch on in
    view (layouts 6 and 10, goal in 134 and 137 steps);
  - the new-colour tests' time per layout: 0.83–1.03 times card 045's,
    except seed 403 with purple (4.2 s against 20.6; card 044: 11.9).
- **Prediction:** met for both criteria and for time; not met for
  decisions, which differ in the new-colour tests.

## 8. Decision

**Revise** (the user, 2026-10-01). Reading the same thing first removed
the wall's slowdown without the filter, but it used recall's two levels as
a switch. After one failed toggle of a new-colour door (switch off), the
door counted as never openable, where the try only shows it does not open
in that situation. As the user put it: the agent can change its
situation, and a change in conditions may change the outcome. The same
switch sits in the search for conditions (which remembered situations the
planner considers for an action on a thing), so the agent could not find
"switch on in view" as the condition either. [Card
047](../047-situations-from-both-levels/card.md) takes situations from
both levels wherever the planner asks in what situations an action could
work, and lets recall's own prediction judge each. A quick check on seed
400's yellow test (layouts 0–11): goal in 10 of 12, against 4 for card
045 and 2 for this card, with the switch turned on by plan in 4.
