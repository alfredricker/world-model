---
id: "096.1"
title: how many occurrences arm C needs, and where recall alone falls short (offline)
rung: 1
serves: [P19, P10, P6]
status: done
verdict: pass
arch_version: 24
date: 2026-10-10
---

# 096.1: few occurrences

The first step of [card 096](../096-structure-memory-by-consolidation/card.md)'s series, written
and run overnight (2026-10-10) under the user's overnight authority; new
card 3 of the night. Card 096 planned its first step on tier 2's five
failures, which no longer exist (version 23: 100/100). The literature
review (card 096, section 3) proposed the same question on GOAL P19's
measure instead, which is not saturated.

## 1. Question

Does recall alone (arm C: arm B's vote over stored token instances, arm
A's network as its prior) predict an interaction it has seen 0, 1, 3,
10, 30 or 100 times, or is a structure component needed, and where?
P19 (important consequences from few examples: "accuracy against the
number of distinct occurrences"), P10, P6. Nothing in the agent changes.

## 2. What is measured

An **event** is one (action, token in front, token in the hand) of the
stored tries of tiers 2 and 3 (memory 066), and its outcome what its
tries did. In version 24's store (front and hand only) an event's tries
are one group, at distance 0 from its own query, together with any
token that admission does not tell apart from it. "Memory saw it n
times" is then computed exactly: the vote without that group, plus n
tries of it (`tools/card096.1/fewshot.py`). Right: both the front and the
hand predicted on the right side of 0.5, and every predicted changed
token is the event's. Each event counts once, however often it occurs.
Arm A was trained on tiers 1, 2 and a decoy world, never tier 3, so on
tier 3 its prior comes from other worlds.

## 3. Result

Share of events right (`runs/096.1/fewshot_t2.json`, `fewshot_tier3.json`):

| Tier | Action (events) | Arm | 0 | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|
| 2 | pick up (183) | B (vote alone) | 94.5 | 99.5 | 99.5 | 100.0 | 100.0 | 100.0 |
| 2 | pick up (183) | A (network alone) | 79.8 | 79.8 | 79.8 | 79.8 | 79.8 | 79.8 |
| 2 | pick up (183) | **C (version 24)** | 96.2 | 99.5 | 99.5 | 100.0 | 100.0 | 100.0 |
| 2 | drop (186) | B (vote alone) | 90.3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2 | drop (186) | A (network alone) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2 | drop (186) | **C (version 24)** | 96.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2 | toggle (211) | B (vote alone) | 95.7 | 97.2 | 97.6 | 100.0 | 100.0 | 100.0 |
| 2 | toggle (211) | A (network alone) | 70.1 | 70.1 | 70.1 | 70.1 | 70.1 | 70.1 |
| 2 | toggle (211) | **C (version 24)** | 83.9 | 97.2 | 97.6 | 100.0 | 100.0 | 100.0 |
| 3 | pick up (237) | B (vote alone) | 61.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 3 | pick up (237) | A (network alone) | 86.1 | 86.1 | 86.1 | 86.1 | 86.1 | 86.1 |
| 3 | pick up (237) | **C (version 24)** | 92.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 3 | drop (238) | B (vote alone) | 82.3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 3 | drop (238) | A (network alone) | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 |
| 3 | drop (238) | **C (version 24)** | 99.2 | 99.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| 3 | toggle (246) | B (vote alone) | 76.4 | 96.8 | 98.4 | 99.2 | 99.6 | 99.6 |
| 3 | toggle (246) | A (network alone) | 48.8 | 48.8 | 48.8 | 48.8 | 48.8 | 48.8 |
| 3 | toggle (246) | **C (version 24)** | 58.1 | 96.8 | 98.4 | 99.2 | 99.6 | 99.6 |

Most events change nothing (a pick up facing a wall). GOAL P19 is about
consequences, so the same curve on the events that change something
(`tools/card096.1/split.py`, `runs/096.1/split.json`; version 24's arm A
and α):

| Tier | Action (change events) | A alone | B alone, 0 | **C, 0** | C, 1 | C, 3 |
|---|---|---|---|---|---|---|
| 2 | pick up (18) | 100 | 94 | **94** | 94 | 94 |
| 2 | drop (12) | 100 | 0 | **100** | 100 | 100 |
| 2 | toggle (59) | 0 | 88 | **46** | 90 | 92 |
| 3 | pick up (9) | 100 | 78 | **100** | 100 | 100 |
| 3 | drop (8) | 88 | 0 | **88** | 88 | 100 |
| 3 | toggle (123) | 0 | 68 | **28** | 94 | 97 |

- **Toggles are the gap.** A door or box toggle never seen is predicted
  by arm C on 29–46% of change events, below the vote alone (68–88%);
  after one occurrence 90–94%, after three 92–97%. GOAL P19's own
  example (a locked door opened a few times, predicted for a new door)
  fails on tier 3: the same door with its key is still predicted
  locked after one occurrence (P 0.13–0.34).
- **Arm A is right on 0% of toggle changes** and on 88–100% of pick up
  and drop changes. It was trained on whole states but is queried in the
  agent with two tokens (front and hand); its two-token answers vary
  with the training run (card 096.2 found retrained copies at 0–92%).
  With α = 0.01 it still decides wherever the vote's neighbours are far,
  which pulls toggles toward "nothing happens".
- **Never-seen events that change nothing** fail by kind (doors
  predicted pickable) and state (the nearest neighbours of a closed door
  are locked ones): arm C 88–99%.

## Check for step 096.4 (forgetting)

Card 096's step 096.4 asked whether 095.2's forgetting (no-change groups
dropped where their neighbours already predict them) distorts the counts
that few-occurrence predictions rest on. The split curve recomputed on
stores built without forgetting (`tools/card096.1/noforget.py`,
`runs/096.1/split_noforget.json`) is identical at n = 0, 1 and 3 for
every action on both tiers. But in version 24's store (front and hand
only) forgetting drops almost nothing: 0–2.4% of groups per action (tier
3: 14,793 of 14,793 kept for pick up, 14,474 of 14,665 for drop, 19,228
of 19,236 for toggle), and the stores are the same size (173 and 566
MB). 095.2's 18-fold saving was on the whole-state store before 095.3
narrowed it. So the question does not arise in version 24; it returns
when a store again holds whole states (Crafter).

## 4. Decision

**Keep** (the measure; overnight authority). A structure component is
needed for never-seen events, and for toggles also at one to three
occurrences. The failures are what card 096 asks structure to supply
(kind, state, the key–door relation), and they are what Crafter will
meet constantly. The prior is the place for it, but first it must be
trained on the state it is queried with (card 096.2's finding). P19's
curve is to be reported on events that change something.
