---
id: "096.3"
title: arm A trained on the state the agent asks it about (front and hand)
rung: 1
serves: [P19, P6, P17]
status: done
verdict: fail
arch_version: 24
date: 2026-10-10
---

# 096.3: the prior on the queried state

A step of [card 096](../096-structure-memory-by-consolidation/card.md)'s series, written and
approved overnight (2026-10-10, overnight authority; new card 5 of the
night), from [096.2](../096.2-sleep-by-event/card.md)'s finding.

## 1. Question

Version 24 asks arm A (arm C's prior) about two tokens, the one in front
and the hand (095.3's gate), but arm A was trained on whole states
(about 19 tokens). Its two-token answers depend on the training run:
version 24's copy is right on every pick up and drop change and on none
of the toggle changes; copies retrained by the same recipe are right on
0–92%. If arm A is trained on the two-token state it is asked about,
are its answers stable across seeds, and does arm C predict never-seen
events that change something (GOAL P19's important consequences) at
least as well as the vote alone, with nothing lost where memory has the
event? P19, P6, P17.

## 2. What changes

One component: arm A's training input. Each stored interaction try
becomes (action, front token, hand token) → (front changed, hand
changed, what each became), weighted as before; the same network, loss,
worlds (tiers 1, 2, decoy; never tier 3), steps and α as version 24.
`tools/card096.3/prior.py`.

## 3. Dependencies

Cards 095.1 (arm A), 095.3 (the two-token query), 096.1 (the exact
few-occurrence computation and its split by change), 096.2 (held-out
events, the baselines).

## 4. Data check

Held-out change events (096.2's draw): tier 2 pick up 4, drop 1, toggle
13; tier 3 4, 2, 24. Few for pick up and drop; toggles carry the test.

## 5. Feasibility gate

- **Upper bound:** arm C with the event seen once (096.1: change events
  88–100%; toggles 90–94%).
- **Trivial baseline:** the vote alone (toggle changes never seen:
  68–88%); arm A on whole states with the same events held out (096.2's
  "try" copy).

## 6. Success criteria and prediction

1. **Stable:** the three seeds' copies give the same side of 0.5 for
   both tokens on at least 95% of held-out events, change and no-change.
2. **Never seen, change events:** arm C with the new prior (seed 95)
   right on at least as many held-out toggle changes as the vote alone,
   on tiers 2 and 3, and on more than with the whole-state copy.
3. **Where memory has the event:** at full counts, every event, arm C
   right on a share not below version 24's on any action.

Then, if 1–3 hold, in the agent (copy trained on every event): tier 1
≥ 99%, tiers 2–3 and the fresh seeds not worse than version 24.

**Prediction.** Stable (the two-token input is what it is trained on);
toggle changes never seen rise above the vote alone where the kind and
state carry it (closed and open doors, boxes) but not for the key–door
colour relation; no change where memory has the event (α = 0.01).

**Budget.** Three trainings under a minute each, evaluation a few
minutes; the agent 10 minutes.

## 7. Result

`runs/096.3/prior.json`. Held-out events never seen (n = 0), arm C with
version 24's α; "agree": the three seeds give the same side for both
tokens.

| | Seeds agree | A alone (3 seeds) | C, new prior (seed 95) | C, whole-state copy (096.2) | C, version 24's copy | Vote alone |
|---|---|---|---|---|---|---|
| Tier 2 toggle changes (13) | 54% | 0–15% | 77% | 46% | 46% | 85% |
| Tier 3 toggle changes (24) | 42% | 0–33% | 58% | 13% | 17% | 63% |
| Tier 2–3 pick up and drop changes (11) | 50–100% | 50–100% | 50–100% | 0–100% | 50–100% | 0–100% |
| Every no-change group (212) | 97–100% | 97–100% | 92–100% | 92–100% | 92–100% | 61–100% |

At full counts every action is equal to version 24 (100%; tier 3 toggle
99.6%). Seen once, toggle changes are 85% and 92% for every prior.

What the copies miss (held-out toggle changes): every box (toggling a
box always opens it; all seeds predict nothing happens, P ≤ 0.03), every
locked door with its own key, and, for seed 96, every toggle change. The
vote alone gets every box right from far neighbours (N ≈ 0.004–0.09).

1. **Stable:** not met (toggle changes 42–54%).
2. **Never seen, toggle changes:** not met against the vote alone (77%
   and 58% against 85% and 63%); met against the whole-state copy.
3. **Where memory has the event:** met.

## 8. Decision

**Stop.** Training arm A on the queried form removes most of the damage
the whole-state copies did (tier 3 toggle changes 13–17% → 58%), but a
network prior fitted to about 600 events per world is unstable on the
rare events that matter and below the vote's far neighbours. In
MiniGrid, with the full memory, the agent never needs it: every event
the planner asks about is in memory, and one occurrence is enough for
85–100%. Card 096's question (where structure is needed) is answered for
MiniGrid: for never-seen toggles of known kinds the vote already carries
63–85%, and what remains is the key–door colour relation (the user paused
colour transfer until Crafter) and the vote's weighting of state against
colour. Taken up again when Crafter's new combinations make the prior
matter; until then version 24's arm A and α stay.
