# Overnight 2026-10-08

The user's objective (2026-10-08, going to bed): work on card 092 without
stopping; if tier 2 is below 79% with a significant p-value, propose at
most 2 well-principled revisions, approve them and keep working; runs over
30 minutes allowed with memory caps.

## Before 092

- **091 kept** (the user's decision), version 19. Tier 3's first attempt
  was stopped by the 44 GB memory cap: `routed.py` built a queries × keys ×
  32 array per worker against tier 3's 54,000 keys. Fixed with
  `torch.cdist`; a 2-episode smoke peaked at 5 GB. Tier 3 (30 seeds, an
  hour each) rerun at 23:12 under the cap.

## Card 092: view effects in recall, queried by goal condition

- **Approved** under the overnight authority; literature (STRIPS,
  hindsight experience replay, neural episodic control, case-based
  planning) set as LITERATURE.md's current focus.
- **Data check:** tier 2's memory holds about 260 toggles of a locked
  door after which a box came into view (of 67,000 toggles); tier 1's
  538 after which the goal did; tier 3 many more, with some noise.
- **Implemented** (`tools/card092/reveal.py`, `WM_REVEAL`): when the
  goal's tile is in no place of the believed map, exploration and then
  "open a door with unseen places behind it" (card 060's step, its key
  and blockers chained by the planner) come before the guess-trying
  fallback, doors ranked by recall's view effect.
- **Found while debugging the gate:** the locked green door is never a
  candidate, because "green door, open" has no forward tries in tier 2's
  memory and both version 18's and the router's forward priors predict
  it blocked. The router's forward vote is the weak part (66 training
  keys; open and closed doors not separated). Card 087's property
  network predicts every open door walkable (0.68–1.0) and every locked
  or closed door, key, ball and box blocked, so forward's prior is card
  087's (card 091's own rule: per action, whichever predicts held-out
  tries better). 092 is measured against version 19 + card 087's forward
  prior, so the only difference is the reveal step.

## Card 091's tier 3 (after the user's keep)

**0/30**, against version 18's 5/30. 28 episodes spend most of their
3,600 steps turning in place before a door (95% of steps "explore";
version 18 about 1%). Version 18's flags on the same 2 seeds do not
loop; adding card 087's forward prior to the router does not stop it, so
the router's pick up, drop or toggle predictions cause it. Recorded on
card 091 and in ARCHITECTURE; traced below.

## Card 092's gate

- **Upper bound: passes.** With the true view effect written by hand
  (opening a locked door shows the box), 17 of card 091's 21 tier 2
  failures are solved, most in 26–46 steps (`runs/092/gate_oracle_*`).
  The 4 left: 1002013, 1002015, 1002048 (the reveal step acts for
  about 240–290 steps without success) and 1002088 (box seen, then
  fails).
- **Effect recall: passes.** Held-out tier 2 toggles: recall −0.0035
  per toggle and kind against each kind's frequency −0.0051; on the 48
  where a kind came into view, −3.3 against −7.3.

## Card 092's tiers 1–2

| Tier | 091 (version 19) | Base: 19 + 087's forward prior | 092: base + reveal |
|---|---|---|---|
| 1 | 100% | 100% | 100% |
| 2 | 79% | **96%** | **95%**, 56 steps when solved (base 78), random 0.7% (base 12%) |

Tier 2 is far above 79% (17 seeds won, 1 lost against card 091), so no
revision was triggered. Most of the gain is card 087's forward prior
together with the router (base against 091: 17 won, 0 lost); the reveal
step adds speed, not successes, on tier 2.

## Card 091.1: the router's vote within the same tiles first

Tier 3's loop traced to the router: before the locked blue door, its
P(the door changes) with an empty hand ranges 0.27–0.53 across views,
although memory holds 1,544 such toggles and none opened it, because the
vote runs over all 54,000 toggle keys (blue keys did open blue doors).
Version 18 asked first whether the same tiles were tried (card 042,
codes for identity). Card 091.1 (`WM_ROUTER_ID=1`): the vote only over
stored keys with the query's front and held codes when there are any.
Smoke: the loop is gone (25 and 26 explore steps on 2 seeds, as version
18). Queued: tiers 1–2 for both arms with 091.1, then tier 3 for each
arm (30 seeds, an hour each, 44 GB cap), `runs/092/night.sh`.

## Tiers 1–2 with card 091.1

| Tier 2 (100) | Success | Steps when solved | Random |
|---|---|---|---|
| Base (19 + 087's forward prior) | 96% | 78.1 | 12% |
| 092 (base + reveal) | 95% | 56.4 | 0.7% |
| Base + 091.1 | 95% | 119.4 | 8.7% |
| 092 + 091.1 | 95% | **64.3** | 0.4% |

Tier 1 is 100% in every arm. With 091.1 the two arms fail the same 5
seeds; the reveal step halves the steps when solved (64 against 119).
091.1 costs steps on tier 2 (base 78 → 119) and swaps which seeds fail
(1002017 solved without it, not with it: a 6-step loop holding the red
key before the green box, the plan disappearing when it faces the box;
recall's own prediction for that pick up is stable and right, so the
cause is elsewhere in planning; not traced further tonight).

## Tier 3 runs, first attempts

- **Reveal arm, 00:34–01:11, stopped:** 20 episodes reached the hour at
  58–77 steps (about 55 s per step; version 18: 1.7 s).
- **Cause:** each forked worker ran 24 math threads (load 137 on 24
  cores). The router's per-query vote multiplies a (queries × 54,000)
  matrix, which the math library spreads over every core in every
  worker. A single process with the reveal step took about 7 s per step
  while the CPU was shared (planning 3.7 s, the fallback's reveal and
  guesses the rest).
- **Rerun** from 01:50 with one math thread per worker
  (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`), reveal arm
  then base arm (`runs/092/night2.sh`). Version 18's 5/30 ran without
  this setting; its speed then was 1.7 s per step.
- **One thread per worker was not enough:** the rerun reached 73–100
  steps per hour (about 40 s per step). Measured in one process: the
  router's vote compared every query with all 54,000 stored keys even
  when card 091.1 then kept only the same-tile group. Now the vote is
  computed over that group alone when it exists (the same answers;
  planning time 103 s → 31 s over the same 200 s of an episode). A
  second change, skipping the fallback's repeated exploration after the
  reveal step found nothing, changed behaviour (random actions appeared)
  and was reverted.
- **Final tier 3 queue** from 02:57: reveal arm, then base arm, both on
  the faster router code, one math thread per worker (`runs/092/night2.sh`).
  Earlier partial tier 3 logs are kept as `*_slow*.log`,
  `*_oversubscribed.log`.

## Tier 3: which component is slow

- **Reveal arm stopped at 20 of 30** (04:00): 0 of 20, every episode at
  the hour with 67–110 steps.
- **Speed under equal load** (20 episodes in parallel, 300 s each,
  `runs/092/speed.sh`): version 18 2,970 steps in all; base arm (card
  091.1's router + card 087's forward prior) 3,044; reveal arm 824. The
  router is as fast as version 18; the reveal step is 3.6 times slower.
- **Likely cause:** my ranked "open a door to look" dropped card 060's
  rule of trying the last step's door first while it still gives a plan,
  so every step re-planned through every candidate door. Restored.
- **Base arm's full tier 3** (30 seeds, an hour each) started at 04:25,
  a fair comparison with version 18's 5/30 at the same speed.
- **Restoring card 060's rule did not help** (reverted, so the code is
  the one measured on tier 2). Single processes side by side, 300 s:
  base 144 and 93 steps, reveal 34 and 34; the fallback takes 250–290
  s in both, about 1.8 s per step for base and 8.5 for reveal. With
  the goal unseen, which on tier 3 is nearly the whole episode, the
  reveal step searches the map for a frontier pose at every step, where
  version 18 explored about 25 times per episode and otherwise continued
  a guess cheaply. Making that search incremental between steps is a
  design change for the next card, not tonight's.

## Tier 3 result and decisions (06:05)

| Version | Tier 1 | Tier 2 | Tier 3 (30, an hour each) |
|---|---|---|---|
| 18 | 100% | 59% | 5/30 |
| 19 (091) | 100% | 79% | 0/30 (loops) |
| **20 (091.1 + 087's forward prior)** | **100%** | **95%** | **14/30** (10 won, 1 lost against 18; p = 0.012) |
| 20 + 092's reveal step | 100% | 95%, half the steps | 0/20 (too slow) |

- **091.1: keep → version 20**, with card 087's forward prior
  (ARCHITECTURE, change log). Taken overnight under the user's
  authority; the user may revert.
- **092: revise.** The reveal step plans from the first step and halves
  tier 2's steps, but adds no successes there and its per-step map
  search makes tier 3 too slow for the hour. Proposed revision (not run):
  keep its target between steps and search the frontier incrementally,
  gated on tier 3 speed under 20 parallel episodes.
- **No revision cards were auto-approved:** tier 2 never fell below 79%.
- **Cards made tonight:** 091.1 (and 092's runs). Under the 10-card limit.

## For the morning

1. Confirm or revert version 20 (091.1 + 087's forward prior).
2. 092's revision: incremental frontier search for the reveal step.
3. Tier 3's remaining 16 failures: 15 reach the hour at 1,500–2,400 steps
   (speed is now the limit as much as competence).
4. Tier 2's 5 failures: the traced one (1002017) loops 6 steps holding a
   key before the box although recall's prediction is right.
5. Nothing is committed.
