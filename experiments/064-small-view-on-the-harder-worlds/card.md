---
id: "064"
title: the small view on the harder worlds
rung: 1
serves: [C5, P2, P15, P12]
status: done
verdict: pass
arch_version: 14
date: 2026-10-05
---

# 064: the small view on the harder worlds

Drafted and started overnight on 2026-10-05 (AGENTS.md, "Overnight
sessions"; the user's objective: a view smaller than the map), as the
retention check C5 asks for. The numbers below were fixed before any
run.

## 1. Question

Version 14 (the 7 × 7 occluded view, frontier exploration, a door
opened to look past it, memory learned under the small view) was tested
on the familiar worlds and on the chained rooms with one door. Does it
keep version 10's results on the two harder worlds: the chained rooms
with two doors (card 048: 13 × 8, the goal behind two locked doors, each
key in a different room), and the key world cluttered with extra keys,
switches and vases (card 049)? C5 (earlier capabilities survive), P2,
P15, P12.

## 2. What changes

Nothing in the agent: version 14 as kept, on two more test worlds.

| | Version 10 (full view, card 054's encoder) | This card |
|---|---|---|
| View and memory | 13 × 13 window, the whole map | version 14 |
| Two doors | 100% at 1.00 × the shortest route (card 054, step D, seed 399; version 10 with its codebook encoder: 30/30 in 5 of 5 seeds) | – |
| Cluttered key world | 100% at 1.06 × (card 054, seed 399) | – |

## 3. Dependencies

Version 14 (cards 057, 060–062); card 048's chained rooms, card 049's
cluttered world; card 054's encoder (seed 399).

## 4. Data check

Two doors: card 048's 30 test layouts; cluttered: card 049's first 100
layouts. With occlusion the goal room is unseen at the start in every
two-door layout, and the second key's room is behind the first door.

## 5. Feasibility gate

- **Upper bound:** version 10 with the full view (section 2).
- **Trivial baseline:** none beyond version 11's exploration, which
  failed under occlusion (card 060: 0–4%).

## 6. Success criteria and prediction

Encoder of seed 399:
1. **Two doors:** ≥ 90% of 30 layouts.
2. **Cluttered:** ≥ 95% of 100 layouts.
3. No wrong remembered tile at the end of any episode.

Reported: steps against the shortest route (the shortest route knows the
map; the agent must explore).

**Prediction.** Cluttered passes (one room, like the familiar worlds).
Two doors is doubtful: the second door is found only after the first is
opened, and opening a door to look past it competes with the keys'
conditions.

**Decision rules.** Keep (version 14 confirmed on these worlds) if all
three hold; one declared revision if one fails; stop if either world is
below 70%.

**Budget.** About 10 minutes.

## 7. Result

`runs/064/two_399.json`, `clutter_399.json`.

| | Version 10, full view | Version 14 |
|---|---|---|
| Chained rooms, two doors (30 layouts) | 100% (1.00 × shortest) | **100%** (1.37 ×; median 1.35) |
| Cluttered key world (100 layouts) | 100% (1.06 ×) | **100%** (1.27 ×; median 1.15) |
| Episodes ending with a wrong remembered tile | – | 0 |
| Random steps | 0 | 0 (two doors), 0.15% (cluttered) |

All three criteria hold. The extra steps are the looking: the shortest
route knows the map, while the agent finds the keys and doors as they
come into view, and in the two-door rooms learns of the second door only
after opening the first.

## 8. Decision

**Keep** (version 14 confirmed on the harder worlds). Under a view that
sees a seventh of the chained rooms and nothing behind walls, with a
memory learned only from that view, the agent still reaches the goal
behind two locked doors in every layout, and ignores clutter as before.
