---
id: "068"
title: play starts in the play that builds memory
rung: 1
serves: [P6, P19, P12, C3]
status: approved
verdict:
arch_version: 14
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
hand), and recall still lets other keys open some doors (card 063).

**Decision rules.** Keep if all three hold; revise once if 3 fails;
stop if tier 1 falls below 99%.

**Budget.** Memory about a minute per tier; tier 1 about a minute; tier
2 at most about 25 minutes.
