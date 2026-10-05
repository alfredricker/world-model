---
id: "060"
title: walls hide what is behind them
rung: 1
serves: [P15, P2, P12, C5]
status: approved
verdict:
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

## 8. Decision
