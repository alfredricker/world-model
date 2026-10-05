---
id: "068"
title: play starts in the play that builds memory
rung: 1
serves: [P6, P19, P12, C3]
status: done
verdict: fail
arch_version: 15
date: 2026-10-05
---

# 068: play starts in the play that builds memory

Asked for by the user on 2026-10-05 ("yes" to play starts as the next
card, after card 066's diagnosis of tier 2).

## 1. Question

When half of the random play that builds memory starts from a play start
(anywhere in the map, some doors open, sometimes holding a locked door's
key), does recall learn that the open door of every colour can be walked
through, so that every locked door becomes something the agent can open,
and does tier 2 improve? P6 (action consequences in every colour), P19
(rare consequences from few examples), P12 (a door becomes a condition
only when it is openable), C3 (the curriculum is declared).

## 2. What changes

One component: the play that builds memory.

| | Card 066 | This card |
|---|---|---|
| Episode start | the environment's own | half of episodes: a play start |
| Play start | – | every door open with probability 1/2; with probability 1/2 the agent holds the key of a door still locked (taken from the floor or from its box); the agent on an empty place anywhere, facing any way |
| Actions, steps, sampling, believed views | random play, about 3.2 million steps, card 034's sampling, card 062's views | the same |

This is CHARTER's declared curriculum for the chained rooms (2026-09-24),
carried to the MiniGrid tiers.

## 3. Dependencies

Card 066's harness and memory; the best version after card 067's
decision; card 047's openable things (a locked door is openable when
recall predicts that a pick up or toggle makes it a tile it can walk
onto).

## 4. Data check

Card 066's tier 2 memory: forward steps onto an open door among the
stored rows, per colour: red 2, green 0, blue 4, purple 5, yellow 8, grey
8; locked doors openable: 4 of 6 colours (green and blue not). This
card's counts are reported before the main run.

## 5. Feasibility gate

- **Upper bound:** none needed for the data question (counts and
  openable per colour are read directly).
- **Trivial baseline:** card 066's memory (4 of 6 colours openable).

## 6. Success criteria and prediction

The best version after card 067; card 066's seeds (200 episodes of tier
1, 100 of tier 2, 5 minutes at most each); tier 3 as card 066 (memory
not yet buildable).

1. **Tier 1:** success ≥ 99% (CHARTER).
2. **Tier 2:** not worse than the best version (paired, McNemar's test,
   p < 0.05).
3. **Openable:** with tier 2's memory, the locked door of each of the six
   colours is openable (6 of 6, from 4 of 6).

**Prediction.** 3 holds. Tier 2 better only a little: card 067's traces
show the agent reaching the box and then failing on the hand (recall
offers "pick up the open door" or "hold the ball" as ways to free the
hand).

**Decision rules.** Keep if all three hold; revise once if 3 fails;
stop if tier 1 falls below 99%.

**Budget.** Memory about a minute per tier; tier 1 about a minute; tier
2 at most about 25 minutes.

## 7. Result

Memory (`runs/068/memory_tier{1,2}.json`): tier 2, 2,755 of 5,555
episodes are play starts (681 holding a key); open doors walked onto
174–200 times per colour (stored rows 16–29, from 0–8); 949 doors opened
(card 066: 416). Tier 1: 2,507 play starts.

Recall's view-set fit (card 039's and card 042's) needed a 7 GB tensor on
this memory and ran out of GPU memory. It now computes the same distances
with one fused GPU kernel and passes the gradient through each token's
nearest match only (`tools/card039/slot_planner.py`, `chamfer_a2b`):
values within 2 × 10⁻⁷ of the old formula, gradients equal up to ties;
0.56 GB and 15 ms per step at tier 2's size. Tier 1 ran before and after
the change: the same outcome and steps in 200 of 200 episodes.

| | Best version (card 067) | This card |
|---|---|---|
| Tier 1 | 100% (200), 20.8 steps | **100%** (200), 20.8 steps |
| Tier 2 | 0% (100) | **0%** (100); no episode differs |
| Tier 2 setup | 143 s | 746 s |
| Locked doors openable, tier 2 | 4 of 6 | **5 of 6** (blue not) |

- **Criterion 1: pass.** **Criterion 2: pass** (no episode solved by one
  and not the other; McNemar's test, p = 1). **Criterion 3: not met**,
  5 of 6.
- **Why blue is not openable** (`runs/068/openable_why_tier2.log`,
  card 047's test situation by situation). Recall predicts that a toggle
  changes the locked door in every colour, then predicts the tile it
  becomes with card 038's ways (keep, copy, shift or set, per part). For
  red, green, purple and grey that tile is the open door of the colour,
  which forward recall knows is walkable. For blue and yellow it is a
  vector no catalogue tile has (a fresh token); forward recall calls
  yellow's walkable and blue's not. Every colour has the stored situations
  and walkability the card set out to give, so what fails is the
  prediction of the tile a door becomes, not memory.
- Seen on the way: in the opened situations recall also predicts that a
  locked door opens when a ball is held (most situations hold a ball),
  card 066's "hand" cause.

## 8. Decision

**Revise**, as declared for criterion 3. The diagnosis puts the cause
outside this card's component: no change to the play fixes the tile
predicted for a toggled blue door. The revision is therefore a card on
that prediction (card 038's ways for a door that opens), run with this
card's memory; it needs the user's approval. Version 15 is unchanged
until then.
