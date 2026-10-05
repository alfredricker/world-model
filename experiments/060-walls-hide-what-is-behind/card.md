---
id: "060"
title: walls hide what is behind them
rung: 1
serves: [P15, P2, P12, C5]
status: done
verdict: pass
arch_version: 11
date: 2026-10-05
---

# 060: walls hide what is behind them

Drafted and started overnight on 2026-10-05, after card 057 (AGENTS.md,
"Overnight sessions"; the user's objective: a view smaller than the
map). The numbers below were fixed before the main run; the smoke test
that shaped arm B is reported in section 7.

## 1. Question

With MiniGrid's default occlusion (walls and closed doors hide what lies
behind them) on top of version 11's 7 × 7 view, does the agent still
reach its goals when the goal square is hidden in the other room until
the door is opened? P15 (act to see what it does not know), P2 (belief
about what is out of view), P12 (seeing is a condition like any other),
C5 (the same tasks, less in view).

## 2. What changes

One component, exploration's target (version 11's step 4, last point).

| | Version 11 (arm A) | Arm B |
|---|---|---|
| View | 7 × 7 with occlusion (MiniGrid's `process_vis`) | the same |
| No chain: where to look | the nearest placement from which any never-seen place would be in view | the nearest placement viewing a never-seen place next to a known walkable one (Yamauchi's frontier); if there is none, a known token next to never-seen places that recall says can be made walkable (a door) becomes the condition ("walk", j), pursued like any other (its key first) |

Places behind outer walls are never seen; under arm A they stay targets
for ever. Arm B looks only where looking can succeed, and opens a door
to see beyond it.

## 3. Dependencies

Version 11 (card 057); card 045's ("walk", j) conditions; card 054's
encoder. Literature: Yamauchi 1997 (frontier exploration, not in papi);
MiniGrid's visibility rule (Chevalier-Boisvert et al. 2023).

## 4. Data check

Version 10's 30 test layouts per familiar world: the goal square is
always in the right room behind the door, so with occlusion it is
hidden at the start in every layout. Chained rooms with one door: 100
layouts, the goal room behind a locked door.

## 5. Feasibility gate

- **Upper bound:** version 11 without occlusion (card 057: 100%).
- **Trivial baseline:** arm A.

## 6. Success criteria and prediction

Encoder of seed 399 (the four encoders behave identically).
1. **Familiar worlds:** arm B ≥ 95% in every world, steps at most 1.5 ×
   the full view's.
2. **Chained rooms with one door:** arm B ≥ 90%.
3. **Arm B above arm A** on both tests, and no wrong remembered tile at
   the end of any episode.

**Prediction.** Arm A near 0%: the goal is never seen, so no chain
exists, and exploration chases places behind outer walls. Arm B near
100% at about 1.1–1.3 × the full view's steps.

**Decision rules.** Keep (version 12) if all three hold; one declared
revision if one fails; stop if arm B is below 80%.

**Budget.** About 10 minutes.

## 7. Result

**Smoke test first** (5 key-world and 5 both-world layouts, arm B as
first written: frontier only): 0%. Traced: at the start the only
never-seen places lie behind the locked door, none next to a known
walkable place, so arm B had nowhere to look and took random actions.
Before the main run, arm B got its second half (section 2: open a door
next to never-seen places, as the condition ("walk", j)); then 100% in
those 10 layouts.

**Main run, seed 399** (`runs/060/{fam,chain}_{A,B}_399.json`; full view
without occlusion: card 054's version 10 on the same encoder).

| | Full view | Arm A: version 11's exploration | Arm B |
|---|---|---|---|
| Key world: success, steps | 100%, 15.9 | 0% | 100%, 18.1 (1.14×) |
| Switch world | 100%, 16.7 | 3.3% | 100%, 19.1 (1.15×) |
| Either world | 100%, 14.4 | 3.3% | 100%, 17.1 (1.19×) |
| Both world | 100%, 21.0 | 0% | 100%, 23.5 (1.12×) |
| Chained rooms, one door (× shortest) | 100% (1.00) | 4% (8.4) | **71%** (1.66) |
| Episodes ending with a wrong remembered tile | – | 0 | 0 |

Arm A fails as predicted: it keeps walking toward places behind the
outer walls, which it can never see. Criteria 1 and 3 hold; criterion 2
fails. All 29 failed chained-room episodes explored for their whole 200
steps. Traced (layout 0): two doors next to never-seen places were
equally near, and turning toward one made the other the nearer, so the
agent turned left, then right, for the whole episode (card 058's
dithering, here in where to look).

**Declared revision:** the door chosen to look past is kept between
steps while it still gives a plan and still has never-seen places next
to it (`--keep-look`; version 10 already keeps an achiever between
steps). Layout 0 then succeeds in 17 steps (shortest 15).

**Revision** (`runs/060/{fam,chain}_Bk_399.json`): chained rooms 100%
(100 of 100) at 1.61 × the shortest route (median 1.27, worst 3.6; the
agent must find the key and the door before it knows where the goal is,
while the shortest route knows the map). Familiar worlds unchanged:
100% at 1.12–1.19 × the full view's steps. No random step, and no wrong
remembered tile in any episode. All three criteria hold.

## 8. Decision

**Keep** (version 12: walls and closed doors hide what is behind them).
When walls and closed doors hide what lies behind them, the agent looks
only where looking can succeed: at never-seen places next to places it
knows it can walk on. When there are none, it opens a door next to
never-seen places, as the condition ("walk", j) planned like any other
(its key first), and keeps that choice between steps. Seeing is then a
condition the agent works toward (P12, P15), not a separate routine.
Open: the starting memory is still learned from full views (card 057's
exception), and the rooms are small next to the view; only seed 399 was
run, since the four encoders gave identical results in card 057.
