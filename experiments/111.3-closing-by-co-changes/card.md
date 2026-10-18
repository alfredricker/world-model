---
id: "111.3"
title: closing that knows what never varied — doubt the facts a co-occurring change depends on
rung: 1
serves: [P5, P12, P10]
status: done
verdict: pass
arch_version: 27
date: 2026-10-18
---

# 111.3: closing by co-occurring changes

Roadmap step 2, memory's part. Drafted at the user's request (option 2 after
card 112, 2026-10-18). Card 112's gate showed that when memory grows,
answers close on many successes but are complete only 0.51–0.54 of the
time. A closed answer is complete only with respect to the facts its cases
varied. This card makes memory doubt, by a general rule, the unvaried
facts that matter, and keep those answers open. The doubted facts are
experiment's targets.

## 1. Question

Does keeping an answer open while a co-occurring change's conditions never
varied make "closed" mean complete, even with a large memory? And does it
leave complete answers closed? Rung 1. Serves P5, P12 and P10.

## 2. What changes

One component: the closing rule ([tools/card111.3/](../../tools/card111.3/)).

```
111.2:  closed when the kept facts leave only successes, on >= 29 cases
111.3:  the co-changes of e: the other tokens that change in the same step (in Crafter, compared in the world's
        frame, by the agent's own pose: a view that shifts is not a change)
        each co-change is asked of memory as its own event: its kept facts, and its own code (the anchor's value)
        doubted facts: facts of e that a co-change needs and that never varied among e's final cases
        closed when 111.2 closes AND no fact is doubted; otherwise open, the doubted facts listed first
```

- **Changed after the smoke test (2026-10-18), before the main run.** A
  needed fact is doubted only if it varies somewhere in e's memory set
  (the same action and code) but never among e's final cases.
  - Without this, a lamp going out reopened every answer on facts that
    hold in every remembered case, such as "outside the room 8 columns to
    the right".
  - Facts that never vary anywhere in memory are background: no case and
    no try in memory's experience separates them.
- **Worked case** (card 112's gate, a lamp lighting). The agent is three
  left of the lamp, with a box between them, and pushes right. The box
  moves into the cell beside the lamp, which changes from floor to box.
  - In all 76 of the lamp answer's cases that cell was floor, so memory
    never saw it matter for the lamp.
  - For the co-change (that cell becoming a box), "it was floor" is the
    anchor's own code: a condition.
  - So it's doubted, and the answer stays open.
- **The rule is general.** It reads no kind of token. It holds wherever an
  effect happens through other changes: a tool made uses up materials in
  Crafter, a pushed object moves in robotics.
- **Stated conditions don't change.** Precision and recall are 111.2's.
  Adding the co-changes' conditions as stated conditions is reported as a
  diagnostic: it would cover the misses at a cost in precision.
- **Truth:** card 112's (the lamp rule and the room's border). A known gap
  is declared: world 1 never produces interior walls, and the check does
  not exclude them.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Changes in one step can be seen token by token | yes (pose-placed tokens) | yes (pose) | yes | yes, with tracking | general; the pose is given (C1) |
| A co-change is asked of memory as its own event | yes | yes | yes | yes | general |

## 3. Dependencies

- Cards 111.2 (the contrast and the closing on 29 cases) and 112 (its
  memories, its truth, and the gate's large memory).
- Literature: Cheng and Novick 1990, already listed. A condition is
  judged only where it varied. P21's chains of conditions: an effect
  enabled through another change inherits that change's conditions.

## 4. Data check

Run 2026-10-18 on card 112's gate memory: training plus the evaluation
rooms themselves.
- **Lamps lighting:** 244 closed, 131 complete.
- **All 113 misses** in closed answers are the box's destination: the
  cell beside the lamp, floor, which never varied among the answer's
  cases. 63 are left of the lamp and 50 below it.
- **All 113 are named by a co-change.** That cell itself changes from
  floor to box.
- **Adding** the co-changes' conditions as stated conditions would add
  146 false ones (judged) across those events.
- **A lamp going out:** 69 closed, all complete.

## 5. Feasibility gate

- **Upper bound:** the evaluator's labels in Crafter. World 1's codes are
  its labels.
- **Baselines:**
  - 111.2's closing (on 29 cases);
  - "doubt every unvaried fact", which closes almost nothing (reported).

## 6. Success criteria and prediction

Judged in world 1 with two memories, training only and training plus the
evaluation rooms (card 112's gate), and in Crafter on codes, with labels
as the gate.

1. **Closed means complete:** at least 0.9 of closed answers have no
   missed true condition, for every kind with at least 20 closed.
2. **Doubts point at what's missing.** In answers this rule keeps open
   that 111.2 would close, at least 0.8 of the missed true conditions are
   among the doubted facts, with at most 3 doubted facts per answer on
   average.
3. **Complete answers stay closed:** of the answers 111.2 closes and that
   are complete, at least 0.9 stay closed, for every kind with at least 20.

**Prediction:**
- **World 1:** criterion 1 reaches about 1.0 in both memories, and
  criterion 2 about 1.0, with 1–2 doubted facts.
- **Criterion 3 is the risk:**
  - In world 1, complete lamp answers keep the box's destination among
    their facts, or saw it vary.
  - In Crafter, perception's codes flicker, which makes false co-changes
    and extra doubts. Expect 0.9–0.95 on codes, and about 1.0 on labels.

**Decision rules:**
- **Keep** (memory's closing for step 2) if 1–3 hold on labels, and on
  codes except where the miss is a code split named in the result.
- **Revise** if 2 or 3 fails with a named cause.
- **Stop** if 1 fails on labels. Co-changes then don't account for what
  never varied.

**Budget:** no training. A few co-change questions per event (3–4 in
world 1). Two world 1 runs and two Crafter runs, about 5 minutes in all,
capped, after a 30-event smoke test.

## 7. Result

Run 2026-10-18 ([results.json](results.json); `runs/111.3/`). Every run was
capped: world 1 took 107–133 s, Crafter 19–21 s.
- **Co-changes asked per event:** 0–3.1.
- **Stated conditions are 111.2's.** The rule changes only which answers
  are closed.

| World, memory: kind | 111.2: closed, complete of closed | 111.3: closed, complete of closed (1) | Reopened; misses there doubted (2); doubts per answer | 111.2's complete answers still closed (3) | "Doubt every unvaried fact": closed |
|---|---|---|---|---|---|
| World 1, training + evaluation rooms: lamp lights | 81%, **0.54** | 42% (126), **1.0** | 118; **1.0** (113 of 113); 1.01 | 0.96 (126 of 131) | 44% |
| … lamp goes out | 26%, 1.0 | 26%, 1.0 | 0 | 1.0 | 26% |
| … push, move, blocked | 100%, 1.0 | 100%, 1.0 | 0 | 1.0 | 100% |
| World 1, training: lamp lights | 19%, 1.0 | 19%, 1.0 | 0 | 1.0 | 8% |
| … lamp goes out | 8%, 1.0 | 8%, 1.0 | 0 | 1.0 | 4% |
| … push, move, blocked | 100%, 1.0 | 100%, 1.0 | 0 | 1.0 | 100% |
| Crafter codes: make | 46%, 1.0 | 45%, 1.0 | 1; none missed; 1 | 0.99 (107 of 108) | 0% |
| Crafter codes: blocked | 89%, 1.0 | 89%, 1.0 | 0 | 1.0 | 0% |
| Crafter codes: moved | none closed | none closed | 0 | — | 0% |
| Crafter labels: make, blocked, moved | 46%, 89–100%, 100%; all 1.0 | the same; all 1.0 | 1 (make) | 0.99, 1.0, 1.0 | 0% |

- **Gate: pass** (Crafter on labels).
- **Criterion 1, closed means complete: pass**, 1.0 everywhere.
  - Lamps lighting with the large memory rise from 0.54 to 1.0.
- **Criterion 2, doubts point at what's missing: pass.** All 113 misses
  were doubted, with 1.01 doubts per reopened answer.
  - Every one is the box's destination: the cell beside the lamp, floor,
    which changes to box in the same step.
  - The criterion rests on that one kind, in one memory. It's the only
    place where 111.2's closing failed.
- **Criterion 3, complete answers stay closed: pass** (0.96 and 0.99).
  - With the large memory, 5 complete lamp answers were reopened.
  - In Crafter, one make was reopened on grass in view.
  - Doubting every unvaried fact instead closes 0–44%.
- **Diagnostic: stating the co-changes' conditions.** If the needed facts
  were stated as conditions rather than doubted:
  - lamps lighting's recall rises to 1.0 with training memory alone
    (from 0.65), at a cost of 40 false conditions over 300 events;
  - a lamp going out's rises to 0.63 (from 0.59);
  - Crafter's makes are unchanged, at a cost of about 180 false.

## 8. Decision

**Keep** (2026-10-18). Memory's closing for step 2 is this rule: an answer
is closed when the kept facts leave only successes, on at least 29 cases,
and no fact a co-occurring change needs went untested among them.
- **What it gives experiment:** a short list, about one fact per open
  answer.
- **One caution.** In world 1 the doubted fact (the box's destination)
  can't be varied by the agent: walls and lamps don't move. No experiment
  settles it. Deduction can: the lamp lights through the box's move, and
  memory knows from 6,545 pushes that a push needs a free destination.
  The diagnostic shows that inheriting the co-change's conditions would
  find lamps lighting's conditions from training memory alone (recall
  1.0), at a cost in precision.
- **Next, for the user:**
  1. **Conditions through co-changes (P21 chains):** state a co-change's
     conditions as the event's own, with a test that keeps precision.
  2. **Experiments that set up a doubted fact** (rung 4).
  3. **Craftium follow-up** of step 2.
