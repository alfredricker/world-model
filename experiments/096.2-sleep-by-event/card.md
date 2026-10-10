---
id: "096.2"
title: a sleep that replays stored tries into arm A by event, its weight fitted on events memory lacks
rung: 1
serves: [P19, P10, P6]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 096.2: sleep by event

A step of [card 096](../096-structure-memory-by-consolidation/card.md)'s series, written and
approved overnight (2026-10-10, overnight authority; new card 4 of the
night). Offline first; the agent only if the offline criteria hold.

## 1. Question

[Card 096.1](../096.1-few-occurrences/card.md) found that arm C predicts any event memory has
seen once (97–100%), but fails on never-seen events by kind, state and
the key–door relation, and that arm A, its prior, is weak exactly there
(49–86% per distinct event on pick up and toggle). Two causes are in how
arm A is made: it is trained by sampling tries, so frequent events
swamp rare ones, and its weight α was fitted per try at full data, where
the prior hardly matters. If a sleep replays stored tries into arm A by
event (each event equally, as complementary learning systems replay
rare events marked so they are not swamped), over the agent's own world
too, and α is fitted where the prior is used (events memory lacks), does
arm C predict never-seen events better, without losing anything where
memory has the event? P19, P10, P6.

## 2. What changes

One component: how arm A, the vote's prior, is consolidated and weighted.

| | Version 24 | This card |
|---|---|---|
| Arm A's replay | tries sampled by their weight (card 095.1) | each event (action, token in front, hand) equally, then a random try of it |
| Worlds replayed | tiers 1, 2, a decoy world | the same, plus the agent's own world (tier 3) |
| α | fitted per try at full data (0.01 for most actions) | fitted by the likelihood of events memory lacks |

## 3. Dependencies

Cards 095.1 (arm A and its training), 095.3 (the store), 096.1 (the
exact few-occurrence computation). Literature: LITERATURE.md's current
focus (complementary learning systems; prioritised replay's change of
distribution; Griffiths et al. 2007 on α as the base's weight).

## 4. Data check

Events per world (096.1): tier 2 183 / 186 / 211, tier 3 237 / 238 / 246
(pick up / drop / toggle). 20% of each world's events per action are
held out of every arm A trained here, with all their tries.

## 5. Feasibility gate

- **Upper bound:** arm C with the event in memory once (096.1: 97–100%).
- **Trivial baseline:** version 24's recipe retrained with the same
  events held out (arm A by try, α as fitted in 095.1), and the vote
  alone.

## 6. Success criteria and prediction

On the held-out events of tiers 2 and 3 (`tools/card096.2/sleep.py`):
1. **Never seen (n = 0):** arm C with the sleep (by event, own world,
   α fitted on half the held-out events and scored on the other half)
   right on more held-out events than version 24's recipe, pooled over
   actions and tiers, sign test p < 0.05; and not below the vote alone
   on any action.
2. **Seen once or three times:** not below version 24's recipe on any
   action (the P19 points the agent mostly lives at).
3. **Where memory has the event:** at full counts, on every event, right
   share not below version 24's on any action.

Then, if 1–3 hold: the sleep's arm A and α in the agent, CHARTER's
tiers (tier 1 ≥ 99%, tiers 2–3 no worse) and the fresh seeds.

**Prediction.** By event replay lifts arm A on rare events (kinds and
states) to near the vote's level or above; the key–door relation stays
hard (five matching pairs to learn from, the sixth held out). Never-seen
events gain most on tier 3, where arm A is now a prior from other
worlds. The agent's tiers are unchanged (every event is in memory).

**Budget.** Three arm A trainings of about a minute each, evaluation a
few minutes; the agent's tiers 10 minutes.

## 7. Result (offline)

`runs/096.2/sleep.json`. Held-out events never seen (n = 0), share right
(held-out events per column: 37 / 37 / 42 / 47 / 48 / 49):

| | T2 pick up | T2 drop | T2 toggle | T3 pick up | T3 drop | T3 toggle |
|---|---|---|---|---|---|---|
| Vote alone | 97.3 | 97.3 | 95.2 | 61.7 | 81.2 | 77.5 |
| Version 24's recipe (by try, α 095.1) | 100.0 | 97.3 | 83.3 | 91.5 | 95.8 | 53.1 |
| By try, α refitted | 97.3 | 97.3 | 92.9 | 93.6 | 95.8 | 71.4 |
| By event, α refitted | 100.0 | 94.6 | 88.1 | 100.0 | 81.2 | 71.4 |
| **Sleep: by event + own world, α refitted** | 97.3 | 97.3 | 90.5 | 97.9 | 100.0 | 77.5 |
| Sleep's arm A alone | 97.3 | 100.0 | 69.0 | 95.7 | 100.0 | 75.5 |

Sleep against version 24's recipe at n = 0: **20 events won, 1 lost**
(p ≈ 0.00002). At n = 1 and 3 every action is equal or better (tier 3
drop 100% against 95.8%); at full counts every action is equal (tier 3
toggle 99.6% for both, the rest 100%).

What each part does: refitting α on events memory lacks helps toggles
most (version 24's recipe 53% → 71% on tier 3); replay by event helps
pick up (tier 3 94% → 100%); the agent's own world helps drop and toggle
(tier 3 81% → 100% and 71% → 78%). The cross-fitted α differ by fold
(tier 2 drop: 0.001 and 10,000): with about 20 events per fold, α is
loosely determined.

1. **Never seen:** pooled, met (20 won, 1 lost); "not below the vote
   alone on any action" **not met** on tier 2 toggles (90.5% against
   95.2%, 2 of 42 events; version 24's recipe is 83.3%).
2. **Seen once or three times:** met.
3. **Where memory has the event:** met.

Verdict: fail on criterion 1's second clause. The prior is still worse
than no prior on tier 2's held-out toggles, less so than version 24's.

## 8. Result in the agent, and what the offline result was

The sleep's arm A (by event, all four worlds, nothing held out) with α
fitted on all held-out events (`runs/096.2/alpha.json`: drop 10,000,
the grid's top) in version 24: tier 1 200/200 (identical); **tier 2
89/100** (version 24: 100/100), 70 steps against 30; tier 3 30/30 in
**769 steps** against 129. Either part alone solves the 11 lost tier 2
seeds; together, on 1002029, the agent picks up the key, can no longer
plan to drop it, and walks into the ball blocking the door for 570
steps. Queried with two tokens, the deployed arm A predicts that a key
dropped onto the floor stays in the hand (P 0.30 and 0.02), and α =
10,000 lets it override memory's hundreds of tries.

Splitting the offline result by event kind showed why it misled:
- most held-out events change nothing, and "nothing happens" wins them;
  the per-event share rewarded a prior that leans that way;
- every arm A retrained here is erratic when queried with two tokens
  (front and hand), though trained on whole states: retrained by try,
  version 24's own recipe gives 0% of tier 2's pick up changes, against
  100% for version 24's copy; by event, 42–92%;
- α fitted on 20–50 held-out events runs to the grid's edge wherever the
  prior happens to fit them.

## 9. Decision

**Stop.** Replay by event is not shown to help, because the prior it
trains is not queried in the form it was trained on. The lessons go to
LESSONS.md. The next step for card 096: train arm A on the two-token
states the agent asks it about (front and hand, 095.3's gate), check
that copies trained from different seeds agree, and judge P19's curve
on events that change something. α is fitted per try at full data as
before until that prior is reliable.
