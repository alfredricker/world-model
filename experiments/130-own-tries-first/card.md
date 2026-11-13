---
id: "130"
title: own tries first — version 27's rule carried to og3: a touch that failed is not recalled again, one that worked is preferred
rung: 3
serves: [P4, P12, P21]
status: done
verdict: pass
arch_version: og4
date: 2026-11-11
---

# 130: own tries first

Roadmap step 3 (skills), rung 3. og3 recalls a remembered touch by how
well it matches now and the goal, and learns nothing from doing it:
a touch that has just failed can be recalled again, and 8 of task 3's
14 failures in [card 129.2](../129.2-appearance-per-place-aim-per-group/card.md)
are drawers retried until the steps ran out. The user (2026-11-11)
pointed out that version 27 already learns from failed attempts; this
card carries that rule over.

Biological counterpart: as in version 27's arm C (cards 095.1–099):
recent personal experience outweighs general knowledge of what an
action does (instance-based learning, Logan 1988).

## 1. Question

Does og3 solve more, in fewer touches, when its own tries of a
remembered touch decide which touch it recalls? P4 (memory), P12, P21.

## 2. What changes

One component: which of the matching touches is redone.

```
og3:  one of the 3 best matches, at random; nothing is kept about how it went
130:  after each touch the scene is read again: a move worked if its variable no longer
      differs from the goal, a press if the condition's state changed
      this episode: a touch that failed is not recalled again while others remain
      long term:    among the 10 best matches, the touch with the highest
                    P = (own successes + β·p0) / (own tries + β)
                    p0 = the agent's own success rate for that variable; β = 1
```

- **Version 27's form:** P = (N_own + β·P_C) / (N_own + β), with its
  fit that one try decides. Here β is set to 1, not fitted.
- **Practice:** the long-term counts come from 150 self-made problems
  the agent solves itself; on the benchmark they are frozen and only
  the episode's own failures are added.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The agent can tell whether its act worked by looking again | holds | holds | holds | holds | the planner's own readings |
| A remembered act that worked once tends to work again | holds for the same target | holds | holds | holds for fixed fixtures | slides: 72% after a success, 24% after a failure |
| The agent may practise before it is tested | holds | holds | few tries allowed | costly | 150 self-made problems |

## 3. Dependencies

Card 129.2 (og3). Version 27's arm C (ARCHITECTURE; cards 095.1–099)
for the rule.

## 4. Data check

From og3's earlier logs (827 slide touches): overall 39% worked; of 57
touches tried twice, the second try worked in 72% after a first
success and 24% after a first failure; 20 touches worked at least 80%
of the time over 3 or more tries. So a touch's success repeats and
there is something to learn.

## 5. Feasibility gate

Run 2026-11-11 (`tools/card130/plan.py`; each run under 5 minutes).
Practice: 150 mixed self-made problems in three batches, solved 86%,
92%, 94% in turn; 626 tries of 141 touches recorded.
Then 50 fresh problems of each kind, counts frozen, against og3 on
the same problems:

| | Own tries | og3 |
|---|---|---|
| Locks, slides and cube: solved | 92% | 88% |
| Locks and slides: solved | 94% | 100% |
| Slide touches that worked (both sets) | 53%, 50% | 39%, 42% |
| Opening the drawer, per try (truth) | 39%, 33% | 21%, 22% |
| Rounds per episode | 5.0, 4.3 | 5.7, 4.3 |

Own tries make a slide touch work more often (about 50% against 40%),
as the data check promised. Problems solved do not differ beyond
chance (about 6 points with 50 problems). Opening the drawer stays
poor: choosing better among replays does not make a replay react to
the drawer. Gate passed as a mechanism; the gain is modest.

## 6. Success criteria and prediction

Agreed with the user 2026-11-11 ("i approve"), before the run. Benchmark tasks 1–3, 50 episodes
each, run once, counts frozen:
1. Task 3 at least 80% (og3: 72%; og2: 86%).
2. Tasks 1 and 2 not more than 6 points under og3 (84%, 90%).
3. Slide touches work more often than og3's in its own run of the
   same tasks (the agent's own measure).

**Prediction.** 3 holds. 1 is uncertain: the gate shows fewer wasted
touches but no clear gain in problems solved, and the drawer's opening
remains about one try in three.

**Budget.** No training. Each run under 5 minutes, capped.

## 7. Result

Run 2026-11-11, once (`tools/card130/plan.py --task k`, seed 0, the
counts frozen as the practice phase left them; each run under 3
minutes, capped). og3's numbers are its own run of the same tasks
(card 129.2); touches are scored from both logs by
`tools/card130/score.py`: a move worked if the next reading no longer
shows its variable differing from the goal.

| | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Solved, own tries first | 100% | 86% | 82% |
| Solved, og3 | 84% | 90% | 72% |
| Slide touches that worked, own tries first | 56% (94 of 167) | 58% (105 of 182) | 38% (102 of 266) |
| Slide touches that worked, og3 | 38% (98 of 261) | 70% (103 of 148) | 36% (103 of 286) |
| Opening the drawer, per try, own / og3 | 45% / 24% | not asked | 36% / 28% |
| Window touches that worked, own / og3 | 77% / 77% | 52% / 65% | 42% / 61% |

1. Task 3 at least 80%: **82%, passed.**
2. Tasks 1 and 2 no more than 6 points under og3: **100% and 86%
   (4 under), passed.**
3. Slide touches work more often than og3's: **49% against 44% over
   the three tasks together (301 of 615, 304 of 695), passed
   narrowly and not evenly.** The gain is the drawer's opening; in
   task 2 the slides did worse (58% against 70%).

With 50 episodes a difference under about 7 points is within chance,
so only task 1's gain (16 points) is clear by itself. With tasks 4
and 5 at 0% the five-task mean is 53.6% (og3 49.2%, og2 53.2%); the
published 54.4% is not a fair comparison (card 128).

**What still fails.** Of the 16 failures in tasks 2 and 3, 12 stopped
with the window left 4 to 10 cm open and read as closed, and 4 never
opened the drawer in 750 steps (og3: 10 and 9). The first kind is a
reading fault, and it poisons this card's rule: a touch that leaves
the window short is counted as having worked, so such touches are
kept and preferred. That is the likely reason window touches work
less often here than in og3 in tasks 2 and 3.

## 8. Decision

**Keep.** Version 27's rule carries over: a failed touch is not
recalled again in the episode, and the agent's own counts choose
among matches. Opening the drawer works in 36–45% of tries against
24–28%, task 1 goes from 84% to 100% and task 3 from 72% to 82%.
Limits: the rule trusts the agent's reading of "worked", which is
wrong for a slide left a few centimetres short; β is set to 1, not
fitted; counts come from 150 practice problems of one seed; one
benchmark seed. Not fixed by this card: a replay does not react to
the drawer while it runs. Adopted as version og4 (the user,
2026-11-11).
