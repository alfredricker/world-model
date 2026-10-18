---
id: "111.2"
title: memory's share of conditions, rescored — the pushed box counted, separable by significance, closed on 29 cases
rung: 1
serves: [P12, P10, P5, P19]
status: done
verdict: fail
arch_version: 27
date: 2026-10-18
---

# 111.2: memory's share of conditions, rescored

Roadmap step 2, its memory part. This revises 111.1 as that card's decision
set out; the user confirmed the revision and said "proceed with 111.2",
2026-10-18. The method is 111.1's (a)–(d). What changes is two scoring
flaws, and the rule for when an answer is closed. All three are fixed
here, before the run. The test uses events 111.1 did not see.

## 1. Question

With the scoring fixed, does contrast over memory state the conditions
memory holds the evidence for (precision and recall ≥ 0.9)? And is an
answer it marks closed complete? Rung 1. Serves P12, P10, P5 and P19.

## 2. What changes

```
111.1:  closed when the kept facts leave only successes, on >= 5 cases
111.2:  closed when they leave only successes, on >= 29 cases: a failure rate of 10% or more would most likely
        have shown ((1 - 0.1)^29 < 0.05)
```

This is the one change to the method
([tools/card111.2/contrast.py](../../tools/card111.2/contrast.py)). Scoring
changes, both from 111.1's checks:
- **Truth, world 1:**
  - A cell whose every replacement gives a state the world can't produce
    counts as a condition when the event changes it (the box pushed away
    from a lamp).
  - If the event doesn't change it, the cell is still not judged.
- **"Contrastable"** means memory holds the evidence. Among the cases
  that match e on every other true condition, e's value raises success,
  with a one-sided Fisher p < 0.01 / 50. (111.1 asked for 5 cases a side.)

### Portability check (CHARTER rule 9)

As 111.1. The closing rule is a statistic over stored cases and assumes
nothing about the world.

## 3. Dependencies

Cards 111 (kept) and 111.1 (revise), and their literature.

## 4. Data check

- **World 1, fresh events.** 111.1 used every lamp event in card 102's
  test rooms (213 and 141). This card plays 2,000 new test-room episodes,
  100 steps each, with card 102's random policy and room generator, seed
  1112 (card 102 used 1022).
  - Events are scored on the evaluator's kinds: world 1's codes map one to
    one onto them (card 111).
  - Memory is card 102's training set, as before.
- **Crafter.** The closing rule was set on world 1's lamps lighting, so
  none of Crafter's events tuned it.
  - Blocked moves and moves that succeeded are a new draw of 300 (seed
    1112).
  - Makes are the same 236: the test set holds no others.
- Counts are reported in section 7.

## 5. Feasibility gate

- **Upper bound:** Crafter on the evaluator's labels.
- **Baselines:**
  - 111.1's closing rule (5 cases), run beside it on the same events;
  - "every token near the change".

## 6. Success criteria and prediction

Judged in world 1 (fresh events) and in Crafter on perception's codes.

1. **Precision ≥ 0.9** of stated conditions, for every kind with ≥ 100
   events.
2. **Recall ≥ 0.9 on contrastable conditions**, for every kind.
3. **Closed means complete:** at least 0.9 of the events memory marks
   closed have no missed true condition, for every kind with at least 20
   closed events. The share closed is reported.

**Prediction:**
- **Criterion 2** passes everywhere on labels. 111.1's checks found 3
  misses in about 238 for a lamp going out.
- **Criterion 1:** world 1's lamps lighting are borderline (0.896 in
  111.1). Crafter's codes fail as before (code splits).
- **Criterion 3** passes. Lamps lighting close less often: about 35%
  where 111.1 had 45%.

**Decision rules:**
- **Keep** if 1–3 hold on labels, and on codes except where the miss is a
  code split named in the result. Memory's part of step 2 is then built,
  and card 112 (experiment on the open answers) is next.
- **Revise** if a criterion fails on labels with a named fix.
- **Stop** if criterion 2 fails on labels.

**Budget:** episode generation in under a minute. Then three runs (world 1;
Crafter on codes and on labels), about 2.5 minutes in all, capped.

## 7. Result

Run 2026-10-18 ([results.json](results.json); `runs/111.2/`). Every run was
capped: world 1 took 119 s, Crafter 26 s per form.
- **Fresh world 1 test rooms:** 200,000 steps. 355 steps light a lamp and
  263 put one out. The draw gives 300 and 264 events.
- **Crafter:** a new draw for its moves; the same 236 makes.
- **Baseline** "near the change": precision 0.04–0.44.

| Kind (events) | Precision (1) | Recall | Contrastable share | Recall on contrastable (2) | Closed (events) | Closed and complete (3) | 111.1's closing on 5: closed / complete |
|---|---|---|---|---|---|---|---|
| World 1, lamp lights (300) | **0.894** | 0.65 | 0.62 | 0.95 | 19% (57) | 1.0 | 54% / 0.35 |
| World 1, lamp goes out (264) | 0.99 | 0.57 | 0.31 | 0.94 | 8% (22) | 1.0 | 8% / 1.0 |
| World 1, push (300) | 1.0 | 1.0 | 1.0 | 1.0 | 100% | 1.0 | same |
| World 1, move (300) | 1.0 | 1.0 | 1.0 | 1.0 | 100% | 1.0 | same |
| World 1, blocked (300) | 0.96 | 1.0 | 0.99 | 1.0 | 100% | 1.0 | same |
| Crafter labels, make (236) | 0.96 | 0.93 | 0.44 | 1.0 | 46% (108) | 1.0 | 89% / 1.0 |
| Crafter labels, blocked (300) | 1.0 | 1.0 | 0.99 | 1.0 | 100% | 1.0 | same |
| Crafter labels, moved (300) | 0.97 | 1.0 | 1.0 | 1.0 | 100% | 1.0 | same |
| Crafter codes, make (236) | 0.96 | 0.93 | 0.44 | 1.0 | 46% (108) | 1.0 | 89% / 1.0 |
| Crafter codes, blocked (300) | **0.85** | 0.89 | 0.88 | 1.0 | 89% (266) | 1.0 | same |
| Crafter codes, moved (300) | **0.45** | 1.0 | 1.0 | 1.0 | 0% | (none closed) | same |

- **Gate: pass** (Crafter on labels meets 1–3).
- **Criterion 1, precision: fail.**
  - **World 1's lamps lighting: 0.894, on labels.**
  - Crafter's codes fail as predicted. Blocked moves' false conditions
    are code splits, and moves' are "health full" (214); both pass on
    labels.
- **Criterion 2, recall on contrastable conditions: pass** (0.94–1.0 on
  every kind).
- **Criterion 3, closed means complete: pass** (1.0 on every kind with
  closed events).
  - With 111.1's 5 cases, lamps lighting's closed answers would have been
    complete in 0.35.
  - Lamps lighting close less often: 19%, against 35% predicted.
- **P19 at 100 occurrences:**
  - World 1, and Crafter's makes and blocked moves: precision 0.74–1.0,
    recall on contrastable conditions 0.86–1.0.
  - Crafter's moves that succeeded state only wrong conditions at 100
    (precision 0). Their failures are too rare; the curve reaches 0.98 at
    300.

**Check after the run: the room's border**
([world1.py](../../tools/card111.2/world1.py) `--border`, not the card's
scoring).
- 48 of lamps lighting's false conditions are border walls beside the
  lamp: "a wall at (1, −2)" 32 times, "(1, −1)" 16 times.
- A room's border never changes, so replacing a border wall gives a state
  the world can't produce. CHARTER's rung 1 already says the truth is not
  read there. This card's producibility check covered only the lamp rule.
- **With border cells unjudged,** lamps lighting reach 0.94 and blocked
  moves 1.0. Every other number stays within 0.02. On labels, criteria 1–3
  then all pass.

## 8. Decision

**Revise** (2026-10-18), by the rule: criterion 1 fails on labels
(0.894), and the failure names a fix.
- **The fix is to scoring, not to memory.** The truth's producibility
  check must cover every invariant of the world (here the border), as
  CHARTER's rung 1 says.
  - It doesn't need another memory card. Card 112 scores with the full
    check, fixed before its run, and its memory-only arm reports these
    numbers again.
- **What 111.2 shows about memory's share:**
  - Memory finds what it holds the evidence for (criterion 2, 0.94–1.0).
  - It says closed only when that's true (criterion 3, 1.0).
  - The rest is open, and that is experiment's list:
    - lamps lighting, 81% open;
    - a lamp going out, 92%;
    - Crafter's makes, 54%.
- **Not memory's:** Crafter's code splits, for a later card on similarity.
