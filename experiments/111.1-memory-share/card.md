---
id: "111.1"
title: memory's share of conditions — state what memory can separate, mark what it cannot
rung: 1
serves: [P12, P10, P5, P19]
status: done
verdict: fail
arch_version: 27
date: 2026-10-18
---

# 111.1: memory's share of conditions

Roadmap step 2, its memory part. A revision of card 111, drafted at the user's
request ("continue working on this condition discovery through recall and
exploration", 2026-10-18). Conditions have three sources (CHARTER, "Where
statements come from"). This card judges memory only on what memory can
know, and asks it to mark what it cannot, which is the experiment card's
(112) work list.

## 1. Question

Does contrast over memory state every condition that memory holds the
cases to separate, with precision ≥ 0.9? And does it mark, as open, the
events where conditions may be missing? Rung 1. Serves P12 (conditions),
P10 (memory), P5 (memory knows where its answer is incomplete) and P19.

## 2. What changes

One component: card 111's contrast over memory
([tools/card111.1/contrast.py](../../tools/card111.1/contrast.py)).

```
111:    greedy: add the fact with the smallest Fisher p (p clamped at 1e-300); threshold 0.01 / all facts;
        conditions = the cells of the facts kept
111.1:  (a) p kept in log space                                   (a fault fix; already measured, the data check)
        (b) tie set: facts holding on exactly the same remembered cases as the chosen one are returned with it,
            as one condition with alternatives ("memory cannot tell these apart")
        (c) testable facts only (Tarone 1990): the divisor counts only facts whose smallest possible p,
            given their counts in the current cases, could pass. Bonferroni over facts that cannot pass makes
            small sets of cases look empty
        (d) open or closed: the answer is closed when the kept facts leave only successes (at least 5);
            otherwise open, with the remaining failures and the facts of e they differ on (near misses,
            Winston 1970). Open answers are card 112's experiments
```

- **Scoring change, world 1 only:** a replacement that breaks the world's
  own invariant (every lamp lit exactly when a box is beside it) is a state
  the world cannot produce, so the truth is not read there (CHARTER rung
  1). A cell whose every replacement breaks it is not judged. This is the
  box pushed away from a lamp: 141 of 111's false conditions. 105.4's truth
  is reported beside it.
- **A tie set is one stated condition.** It is correct if a member is a
  true condition. Its true members count as found.
- **"Contrastable"** (scoring only): among remembered cases that match e
  on every other true condition, the cell holds e's value in at least 5
  cases and another value in at least 5. A condition that isn't
  contrastable cannot come from memory.

### Portability check (CHARTER rule 9)

As card 111: facts are tokens at offsets from an anchor (in Craftium,
Crafter and ARC-AGI-3; binned in robotics); codes are crisp (declared,
similarity is a later card); failures are tries with the same action
(continuous actions need a kernel). (a)–(d) are statistics over stored
cases and assume nothing about the world.

## 3. Dependencies

- **Card 111** (kept): events, truth, memory, the contrast. Its tools are
  copied to `tools/card111.1/`.
- **Literature** (added when approved):
  - Tarone 1990, a modified Bonferroni method for discrete data;
  - Terada et al. 2013 (LAMP), which uses Tarone's method to find
    significant combinations;
  - Winston 1970, near misses;
  - Cheng and Novick 1990, and Cheng 1997, from card 111.

## 4. Data check

Run 2026-10-18 on 111's events, with (a) only
([runs/111.1/](../../runs/111.1/), `diag_*.json`). Precision / recall,
against 111's numbers:

| Kind | 111 | (a) log space | Contrastable share | Recall on those |
|---|---|---|---|---|
| World 1, lamp lights | 0.85 / 0.55 | 0.85 / 0.55 | 0.55 | 1.0 |
| World 1, lamp goes out | 0.71 / 0.49 | 0.71 / 0.49 | 0.73 | **0.50** |
| World 1, push; move | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 | 1.0 |
| World 1, blocked | 0.35 / 0.27 | **0.96 / 1.0** | 1.0 | 1.0 |
| Crafter codes, make | 0.97 / 0.92 | same | 0.49 | 0.99 |
| Crafter codes, blocked | 0.86 / 0.89 | same | 1.0 | 0.89 |
| Crafter codes, moved | 0.54 / 1.0 | same | 1.0 | 1.0 |

- **Blocked moves were a numerical fault, not stand-ins** (111's
  correction).
- **Lamp lights:** every miss is a condition memory never varied. Test
  rooms put lamps in new columns: training has no case of pushing a box
  rightward onto a lamp. That is experiment's.
- **Lamp goes out:** half of the contrastable conditions are missed. The
  greedy search ends with about 50 cases and 7 failures, and with about
  1,500 facts in the divisor nothing more can pass. (c) is for this.
- **Crafter's makes:** the share is low because counts are exact codes
  (wood 3 ≠ wood 4). This is a lower bound.
- **Crafter's codes:** blocked misses are code splits (e's code of the
  blocking cell does not raise blocking). Moves' false conditions are
  "health full" (185), which goes with no zombie nearby. On labels both
  pass (1.0 / 1.0, 0.97 / 1.0). Both are perception's identity, a later
  card on similarity.

## 5. Feasibility gate

- **Upper bound:** the evaluator's labels in Crafter (world 1's codes
  equal its labels). Pass: criteria 1 and 2 hold on labels.
- **Baselines:** (a) alone (the table above), and "every token near the
  change" (precision 0.04–0.38).

## 6. Success criteria and prediction

Judged on perception's codes, in world 1 and Crafter, on 111's events.

1. **Precision ≥ 0.9** of stated conditions, for every kind with ≥ 100
   events.
2. **Recall ≥ 0.9 on contrastable conditions**, for every kind. Overall
   recall and the contrastable share are reported.
3. **Closed means complete (P5).** Among events memory marks closed, at
   least 0.9 have no missed true condition, for every kind. The share
   marked open is reported. It's the size of card 112's work list.

**Also reported:**
- tie sets (how often, how large);
- the open answers' near-miss facts, and whether they hold the missed
  conditions;
- the P19 curve (10, 30, 100, 300 occurrences);
- (a)–(d) against (a) alone.

**Prediction:**
- **(c)** raises a lamp going out's contrastable recall from 0.50 to about
  0.9.
- **Criterion 1** holds everywhere except Crafter's moves on codes (health
  full, about 0.55). A lamp going out reaches about 1.0 under the new
  truth.
- **Criterion 2** fails only for Crafter's blocked moves on codes (0.89).
- **Criterion 3** is the uncertain one:
  - A second box beside a lamp is never in memory, so memory may stop
    with only successes left and be wrongly sure.
  - Expect 0.7–0.9 for lamps; push, move and blocked near 1.0.

**Decision rules:**
- **Keep** (memory's part of step 2 is built; card 112 is next) if 1–3
  hold on labels, and on codes except where the miss is a code split
  named in the result.
- **Revise** if a criterion fails on labels and the failure names a fix.
- **Stop** if recall on contrastable conditions stays below 0.9 on labels.
  Greedy contrast then cannot do memory's share, and a wider search is
  needed (several branches, or all small conjunctions).

**Budget:** no training. Each run took 13–70 s in 111. Three runs: world 1,
and Crafter on codes and on labels, each with P19. Under 10 minutes in all,
capped, after a 30-event smoke test.

## 7. Result

Run 2026-10-18 ([results.json](results.json); `runs/111.1/`). Every run was
capped: world 1 took 134 s, Crafter 33 s per form. A greedy search costs
7–19 ms per event. Precision and recall use this card's truth (states the
world can produce).

| Kind (events) | Precision (1) | Recall | Contrastable share | Recall on contrastable (2) | Closed share | Closed and complete (3) | (a) only: precision / recall on contrastable | P19 at 100: precision / recall on contrastable |
|---|---|---|---|---|---|---|---|---|
| World 1, lamp lights (213) | **0.896** | 0.63 | 0.56 | 0.97 | 0.45 | **0.34** | 0.85 / 0.97 | 0.91 / 0.90 |
| World 1, lamp goes out (141) | 0.99 | 0.52 | 0.73 | **0.53** | 0.13 | 1.0 | 0.99 / 0.53 | 0.99 / 0.35 |
| World 1, push (300) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| World 1, move (300) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 | none stated (too few failures) |
| World 1, blocked (300) | 0.98 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.98 / 1.0 | 0.74 / 0.97 |
| Crafter labels, make (236) | 0.96 | 0.93 | 0.49 | 0.99 | 0.89 | 1.0 | 0.97 / 0.99 | 0.98 / 0.86 |
| Crafter labels, blocked (300) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Crafter labels, moved (300) | 0.97 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.97 / 1.0 | none stated |
| Crafter codes, make (236) | 0.96 | 0.93 | 0.49 | 0.99 | 0.89 | 1.0 | 0.97 / 0.99 | 0.99 / 0.86 |
| Crafter codes, blocked (300) | **0.84** | 0.89 | 1.0 | **0.89** | 0.89 | 1.0 | 0.86 / 0.89 | 1.0 / 0.78 |
| Crafter codes, moved (300) | **0.46** | 1.0 | 1.0 | 1.0 | 0.0 | (none closed) | 0.54 / 1.0 | none stated |

- **Gate: pass in Crafter** (labels meet criteria 1–3). World 1's codes
  equal its labels, so its rows are the gate too, and they fail.
- **Criterion 1, precision: fail.**
  - World 1's lamps lighting: 0.896. The false conditions are walls
    beside the lamp, which stand for its place.
  - Crafter's codes: code splits and "health full", as predicted. Labels
    pass.
  - **A lamp going out: 0.99** on this card's truth (0.71 on 105.4's).
- **Criterion 2, recall on contrastable conditions: fail**, a lamp going
  out at 0.53, on labels.
- **Criterion 3, closed means complete: fail**, lamps lighting at 0.34.
  The missed conditions are ones memory never varied (contrastable recall
  0.97). In layouts training never had, the facts kept were walls beside
  the lamp, and they left only successes.

**Changes (b)–(d):**
- **(c) Tarone's threshold changed nothing.** A lamp going out stayed at
  0.53. My prediction of about 0.9 was wrong.
- **(b) Tie sets** appear only for lamps lighting: 14% of stated
  conditions, 3.5 cells on average. Blocked moves' ties disappear in log
  space.
- **(d) Near misses** hold 38–44% of the conditions missed in open
  answers in world 1, and 24–34% for Crafter's makes.

**Checks after the run** ([oracle_check.py](../../tools/card111.1/oracle_check.py),
[closed_check.py](../../tools/card111.1/closed_check.py)).
- **Criterion 2's measure was mis-specified twice.**
  - **The pushed box.** This card's truth leaves it unjudged: removing
    it gives a lit lamp with no box beside it. "Contrastable" therefore
    did not hold it fixed, and drew on cases where the box being pushed
    sat elsewhere. In plain terms the pushed box is a condition.
  - **Five cases a side** is not enough for any corrected test.
  - **With the box held:** 322 of a lamp going out's conditions are
    contrastable, and the search finds 235. Of the 87 it misses, only 3
    are significant at 0.01 / 50 with every other condition held, even
    in that best-case subset. The rest are too thin for memory, so
    they're experiment's.
  - Without the box held, 217 looked significant. That is where the 0.53
    came from.
- **Criterion 3: closure on too few cases.** Lamps lighting's closed
  answers resting on 30 or more cases are all complete (33 of 33). Those
  resting on fewer are all incomplete (0 of 63).
  - Most are "the agent two cells left of the lamp's lower neighbour, a
    box between": 15 cases.
  - The cell the box moves into was floor in every one, so memory never
    saw it vary.
  - A pure set of 5 cases is not evidence that nothing else matters.

## 8. Decision

**Revise** (2026-10-18). This departs from the pre-set rule. The stop
rule fired: recall on contrastable conditions, on labels, was 0.53 for a
lamp going out. But the checks above show the measure it read was
mis-specified. With the pushed box held and significance required, the
search misses 3 of about 238 conditions, so the rule's premise ("greedy
contrast cannot do memory's share") does not hold. The user confirmed
the departure on 2026-10-18 ("proceed with 111.2").
- **Kept:**
  - (a) log space;
  - (b) tie sets;
  - (d) open and closed answers;
  - (c) Tarone, which changed nothing here and is never worse than
    Bonferroni.
- **Revision 111.2, fixed before its run:**
  - **Truth:** a cell whose every replacement gives a state the world
    cannot produce counts as a condition when the event changes it (the
    box pushed away).
  - **"Contrastable"** means significant at 0.01 / 50 with every other
    true condition held, not 5 cases a side.
  - **Closed** needs enough cases that a failure rate of 10% or more
    would likely have shown: (1 − 0.1)^n < 0.05, so n ≥ 29. This was
    chosen after seeing 30 work here, so it must hold on events this card
    did not tune it on. 111.2 uses a fresh draw of world 1's events and
    Crafter's.
- **Not memory's:**
  - lamps lighting's conditions that never varied;
  - Crafter's code splits, which are for similarity.
  The first is card 112's experiments, aimed at the open answers.
