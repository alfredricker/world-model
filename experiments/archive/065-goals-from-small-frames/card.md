---
id: "065"
title: goals from frames seen through the small view
rung: 1
serves: [P16, P12, P2, C1, C2, P18]
status: done
verdict: pass
arch_version: 14
date: 2026-10-05
---

# 065: goals from frames seen through the small view

Drafted and started overnight on 2026-10-05 (AGENTS.md, "Overnight
sessions"), the last card of the night: it joins the objective's second
and third parts. The numbers below were fixed before the main run; the
smoke test (3 key-world layouts) is reported in section 7.

## 1. Question

Card 056 inferred goals from example frames of the whole map and reached
them with version 10. When the example frames, the agent's own
experience frames and its acting all go through MiniGrid's 7 × 7
occluded view (version 14), does it still infer each goal from five
frames and reach it as well as a goal written in? P16 (goals from
examples), P12, P2, C1 (goals given as observations), C2 (one view for
everything), P18 (the night's parts working together).

## 2. What changes

One component: what the goal frames and the base-rate frames show.

| | Card 056 (version 10) | This card (version 14) |
|---|---|---|
| A goal frame | the 13 × 13 window, one random frame per episode where the goal holds | the 7 × 7 occluded view, the frame in which the goal became true, one per episode (a small view shows a goal only where it is: a door opening is in front of the agent) |
| Experience frames (the size principle's base rates) | stored frames, whole window | the same frames as the small view showed them |
| Acting | version 10 | version 14 |

Inference is card 056's (Bayesian concept learning with the size
principle, five frames, features "holds h" and "shows u").

## 3. Dependencies

Cards 056 (with its revision: one frame per episode), 057, 060–062;
card 054's encoder (seed 399).

## 4. Data check

Smoke test, key world: goal frame pools 315–361 per held-key colour,
514 for the switch, but only 13–19 per door colour (random play opens
the door rarely, as in card 059); vocabulary under the small view: 19
features.

## 5. Feasibility gate

- **Upper bound:** the goal written in, under the same small view.
- **Trivial baseline:** card 056 with the whole map (frames 100% in
  every world after its revision, 1.04–1.09 × the shortest route).

## 6. Success criteria and prediction

Encoder of seed 399, 30 test layouts per familiar world, four goals
each (the matching key held, the other key held, the door open, the
switch on):
1. **Reached:** goals from frames ≥ 95% in every world.
2. **As well as written in:** within 3 points of the written-in goal
   under the same view, in every world.

Reported: how often the inferred goal equals the written-in one, the
companion features it adds, steps against the shortest route.

**Prediction.** Both hold. Door goals carry a companion ("holds the
key"), which in the key world is reachable and harmless; in the switch
world the door opens without a key, so the companion should not appear.

**Decision rules.** Keep if both hold; one declared revision if one
fails; stop if goals from frames fall below 80% in any world.

**Budget.** About 10 minutes.

## 7. Result

**Smoke test** (3 key-world layouts): 12 of 12 episodes with goals from
frames and with goals written in; the tool ran unchanged into the main
run.

**Main run** (`runs/065/main_399.json`; card 056's full-view numbers
from `runs/056/rev_399.json`).

| World | From frames, small view (× shortest) | Written in, small view | Card 056, frames, full view | Inferred = written in |
|---|---|---|---|---|
| Key | **100%** (1.10) | 100% (1.10) | 100% (1.04) | 73% |
| Switch | **100%** (1.15) | 100% (1.14) | 100% (1.08) | 94% |
| Either | **100%** (1.28) | 100% (1.14) | 100% (1.07) | 87% |
| Both | **98.3%** (1.15) | 99.2% (1.15) | 100% (1.09) | 73% |

Both criteria hold: at least 98.3% in every world, and within 0.9 points
of the written-in goal. Where the inferred goal differs from the
written-in one it adds a companion seen in every example frame: "holds
the key" on door goals in the key and both worlds (the door opens only
with the key there; in the switch world it does not appear, as
predicted), and "shows the goal square" on some door goals, since the
goal comes into view when the door opens. Reaching the goal square in
view costs extra looking: the either world takes 1.28 × the shortest
route against 1.14 × with the goal written in. The both world's two
misses (the other key, the door) and its random steps (11–14% in both
arms) are version 14's, not the inference's: the written-in door goal
also fails once in 30 there.

Goal frames of a door opening are few under random play (4–19 per door
colour in the key and both worlds, 48–67 in the others), the same
thinness card 059 met.

## 8. Decision

**Keep.** Goals given as five example frames, seen through the same
7 × 7 occluded view as everything else, are inferred and reached as
well as goals written in. The frame to show is the one in which the goal
became true, since a small view shows a goal only where it is. Open: the
both world's random steps under the small view (version 14), and too
few door-opening frames from random play (demonstrations, pinned).
