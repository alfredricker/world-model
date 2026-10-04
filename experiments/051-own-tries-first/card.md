---
id: "051"
title: recall through conditions, with a query's own tries first
rung: 0
serves: [P10, P11, P12, P19, P14]
status: done
verdict: pass
arch_version: 8
date: 2026-10-03
---

# 051: recall through conditions, with a query's own tries first

The revision of [card 049](../049-recall-through-conditions/card.md),
asked for by the user on 2026-10-03 ("run that revision, ideally on a
trial not as long as the main run"). Card 050 waits on this card.

## 1. Question

Card 049's recall failed criterion 2 in one seed. In seed 403's
cluttered world, the agent tried to pick up the red key while holding a
purple key 180 times. Every try failed, and the prediction never changed,
because its own failed tries counted no more than each of the thousands
of stored successes they resembled. If a query's own tries come first,
with the condition-weighted neighbours only as their prior, does one
failed try correct what recall inferred, while card 049's gains stay?
Before rung 1. Serves:
- P10 and P19, the situation's own experience used at once;
- P11, a contradicted prediction is revised for that situation only;
- P12, a failed try means "not under these conditions" (the user,
  2026-10-01), so the planner looks for what to change;
- P14, recovery after a failed act.

## 2. What changes

One part of recall for pick up, toggle and drop: how a query's own tries
combine with its neighbours.

```
card 049:  P = (Σ_i w_i n_i + a/L) / (Σ_i w_i |n_i| + a)              all stored tries as neighbours
card 051:  P = (N_own + α P_nb) / (|N_own| + α),  P_nb = card 049's P
           N_own: tries with the same front and held code tuples and the same admitted view values
```

- **Own situation.** The identity level (codes, as card 042), read only
  where the admitted conditions read. Tokens in view that no condition
  reads are still ignored, so card 049's protection against vetoes
  stays.
- **α** is fitted after admission by leaving one try out, as card 038
  fitted its back-off (MacKay and Peto 1995).
- **Unchanged:** admission, the candidates, the planner, tokens, walking.
- **Code:** `tools/card051/own.py`, a subclass of card 049's recall.

## 3. Dependencies

Card 049's recall (gate passed; criteria 1 and 3 passed in 5 of 5).
Card 038's back-off fit (kept with version 6 to 8).

## 4. Data check

Own situations in memory: 50 per action and world (front and held code
pairs), most with hundreds of tries. New tiles (yellow, purple) have none
until tried, so their first prediction is the neighbours' alone.

## 5. Feasibility gate

The upper bound and baselines are card 049's. A trial (shakedown) on the
cases card 049 failed or was weakest, under 10 minutes:
- seed 403, cluttered world, 100 layouts (card 049: 91%);
- seed 403, familiar worlds, 30 layouts (nothing lost);
- seed 401, chained rooms with one door, 100 layouts (card 049: 98%);
- seed 401, new-colour switch door, 6 layouts per colour (card 049: 14%).

Result of the trial (2026-10-03, about 9 minutes, `runs/051_trial_*`):
- **α = 0.00012**, the smallest value on the grid, for every action: a
  situation's own tries decide as soon as there is one, as in card 038.
- **Seed 403, cluttered world: 98%** (card 049: 91%; version 8: 95%).
  Of the two failures, layout 47 fails under every recall, version 8's
  included. Layout 66, traced: the agent picks up the purple key, tries
  the red door once, and recall now says "not with this key", so it drops
  the purple key: the recovery card 049 lacked. Then it picks the purple
  key up and drops it again for the rest of the episode, the same
  alternation as card 049's two doors (the planner's subgoal order).
- **Seed 403, familiar worlds: all four 100%,** steps 0.93–0.97 times
  card 029's, criterion 1's shape passes.
- **Seed 401, chained rooms with one door: 98%** at 1.08, as card 049.
- **Seed 401, new-colour switch door: 0 of 6 per colour** (card 049: 14
  of 100). Not a measurable change: under both recalls the agent acts at
  random 97% of the time in this seed, and card 049's successes were
  random walks (58 steps on average). Recall predicts that toggling the
  new door does nothing (341 held-out errors); untraced.

## 6. Success criteria and prediction

Card 049's three criteria, on seeds 400–404, in a reduced main run
(approved by the user on 2026-10-04: "I don't think we need to be as
thorough as other cards"):
1. **Nothing lost:** the four familiar worlds, 30 layouts each, ≥ 99%
   with steps within 5% of card 029's. Unseen rooms and the new-colour
   tests are not run; card 049 passed them, and this change touches only
   situations with tries of their own.
2. **Irrelevant tokens cannot veto:** chained rooms with one door, ≥ 95%
   at ≤ 1.15 times the shortest route; the cluttered world, ≥ 95%; 100
   layouts each.
3. **The right conditions:** as card 049 (admission is unchanged).

Each criterion in 5 of 5 seeds. Reported: the fitted α, and failed tries
repeated in the same situation. Two doors are not run (the planner's
subgoal order, card 052).

**Prediction.** α near zero, as in card 038, so one failed try flips the
prediction. Seed 403's cluttered world reaches 100%; the other results
stay as in card 049. Seed 401's new-colour test: uncertain.

**Budget.** All fifteen runs (five seeds by three tests) in parallel:
about 15 minutes (`runs/051_small.sh`).

## 7. Result

Reduced main run 2026-10-04 (`runs/051_small.sh`, fifteen runs in
parallel, about 15 minutes; one run, chained rooms seed 400, ran out of
GPU memory and was rerun alone). Numbers: [results.json](results.json).

| Seed | Familiar worlds, 30 layouts each (steps vs card 029) | Chained, one door | Cluttered (card 049) | Toggle at a door reads (key world; switch world) |
|---|---|---|---|---|
| 400 | 100% in all four (0.93–0.97) | 98%, 1.08 | 100% (100%) | relation part 2; switch on |
| 401 | 100% in all four (0.93–0.97) | 98%, 1.08 | 100% (100%) | held part 3, relation part 0; switch on |
| 402 | 100% in all four (0.93–0.97) | 98%, 1.08 | 100% (100%) | relation part 0; switch off |
| 403 | 100% in all four (0.93–0.97) | 98%, 1.08 | 98% (91%) | held part 0, relation part 0; switch on |
| 404 | 100% in all four (0.93–0.97) | 98%, 1.08 | 100% (100%) | held part 2; switch off |

- **Criterion 1: pass, 5 of 5.** All four familiar worlds 100%, effects
  exact, steps 0.93–0.97 times card 029's.
- **Criterion 2: pass, 5 of 5.** Chained rooms with one door: 98% at
  1.08 in every seed; the two failing layouts (78, 88) fail under every
  recall and the upper bound. Cluttered world: 98–100% (card 049: 91% in
  seed 403). Seed 403's two failures are the trial's: layout 47 fails
  under every recall, and layout 66 is the planner's pick-up and drop
  alternation (card 052, change 5).
- **Criterion 3: pass, 5 of 5.** The same conditions as card 049 in every
  seed (admission is unchanged).
- **α = 0.00012 in every seed and action:** a situation's own tries
  decide as soon as there is one.
- **Not measured:** the count of failed tries repeated in the same
  situation (needs per-step logs, which the runs did not keep); times per
  layout, which fifteen parallel runs make incomparable with card 049's.

## 8. Decision

**Proposed: keep** (awaiting the user). With card 049, this is recall
through the conditions that matter, with a query's own tries first: no
vetoes from irrelevant tokens, the right conditions in every seed,
nothing lost in the familiar worlds, and one failed try corrects recall.
Keeping it makes cards 049 and 051 architecture version 9. Left for card
052: the planner's subgoal order (two doors, cluttered layout 66),
the cost per step, and "unknown" read as "fails" (seed 401's new-colour
door).
