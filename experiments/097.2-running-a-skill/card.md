---
id: "097.2"
title: an intention held between steps, reconsidered on events (card 093's design)
rung: 1
serves: [P21, P14, P17, P12]
status: done
verdict: pass
arch_version: 22
date: 2026-10-10
---

# 097.2: an intention held between steps

A step of [card 097](../097-procedural-memory/card.md)'s series, from
[card 093](../093-working-memory/card.md)'s design (the user set 093 aside
on 2026-10-09 and moved its intention and triggers here). Drafted and
approved overnight on 2026-10-10 under the user's overnight authority.
The user had placed this series after cards 094 and 096; it is taken up
now because version 22's tier 3 failures are exactly the defect it
addresses, and it reads nothing of card 094's belief, so it will not be
rebuilt when 094 and 096 land.

## 1. Question

Version 22 ([095.7](../095.7-exploring-uncertain-results/card.md)) plans
from the top at every step. Its five tier 3 failures loop for 3,500
steps: to toggle a door with a ball in front of it, the agent drops the
key it carries to free its hand, the key lands on its route, and the
next step's plan, made from the top again, picks the key back up (traced
on 1003000, 1003018, 1003023). If the agent keeps the chain of conditions
its last deliberation chose and continues it until an event makes it
stale, does it stop looping, with no tier worse? P21 (a held intention
continued cheaply, deliberation when needed), P14, P17, P12.

## 2. What changes

One component: working memory's intention, which decides whether a step
deliberates or continues.

```
version 22, every step:  choose (from the goal) → else fallback → action
this card, every step:   any trigger?  deliberate as version 22; keep its chain of conditions
                         none:         continue: plan the deepest condition of the chain not yet met,
                                       the met ones below it protected → action
```

- **The chain:** the conditions of the chosen plan's trace, top first
  (the goal, or card 060's "this door can be walked onto"), down to the
  need the action serves.
- **Continuing:** the planner's own solve on the deepest unmet condition
  (face needs are walked to by their parent), with the chain above it as
  the chain being pursued and the met conditions below it protected
  (card 029's protection), so a step does not undo what the chain
  already achieved. The chain below takes the new sub-plan.
- **Triggers** (card 093's four): the top condition holds; the pose,
  front or hand differ from the last action's prediction; a token the
  agent can act on (not walkable) is newly in the facts; planning the
  deepest unmet condition finds nothing.
- **Unchanged:** recall, the planner's search and conditions, walking,
  exploration, 095.7's step, card 072's guesses (planned inside the
  fallback as before).

## 3. Dependencies

Version 22; cards 029 (protection), 060 (the door kept between steps),
072, 074. Literature (card 093): execution monitoring with triangle
tables (Fikes, Hart and Nilsson 1972); intentions as commitments that
filter reconsideration (Bratman 1987; Rao and Georgeff 1995); when to
reconsider (Kinny and Georgeff 1991).

## 4. Data check

Version 22's tier 3 failures (5) all loop on the drop and pick up
described above; 1003009 loops in exploration the same way.

## 5. Feasibility gate

- **Upper bound:** the seven looping seeds (1003000, 002, 006, 009, 013,
  020, 023): at least 4 solved.
- **Trivial baseline:** version 22 (0 of the seven).

## 6. Success criteria and prediction

1. **Tier 3:** better than version 22's 24/30 on the same seeds (more won
   than lost, sign test).
2. **Tiers 1–2:** tier 1 ≥ 99%; tier 2 not worse than 100/100.
3. **Fresh seeds:** on 100 tier 3 seeds never traced (1003100–1003199),
   better than version 21 on the same seeds, since the gates of 095.7 and
   this card used seeds from the test set.

**Prediction.** Tier 3 28–30/30; fresh seeds 85–95%. Tiers 1–2
unchanged.

**Budget.** Gate 5 minutes; tiers 1–3 10 minutes; fresh seeds about 30
minutes for both versions.

## Gate result

First form (the old chain kept as it was, nothing protected): 1/7, worse;
the continued chain re-achieved a condition the new sub-plan had just
given up (pick the box back up). Fixed before the main run as stated in
section 2 (the chain takes the new sub-plan; met needs below protected):
**7/7**, in 63–211 steps (`runs/097.2/gate_tier3.json`).

## 7. Result

`tools/card097.2/intent.py` (`WM_INTENT=1`, with 095.7's `WM_INFORM=1`);
`runs/097.2/intent_tier*.json`, `fresh_*_tier3.json`.

| | Tier 1 | Tier 2 | Tier 3 (test seeds) | Fresh 100 seeds | s per step (tier 3) |
|---|---|---|---|---|---|
| Version 21 | 200/200 | 100/100, 55 steps | 24/30, 1,677 steps | 88/100, 1,415 steps, return 0.57 | 0.038 |
| Version 22 | 200/200 | 100/100, 30.7 | 24/30, 264 | | 0.016 |
| **This card** | **200/200**, 24.3 | **100/100**, 30.2 | **30/30**, 129 | **99/100**, 144, return **0.95** | 0.035 |

Return is MiniGrid's (1 − 0.9 × steps / 3,600 on success, 0 otherwise).
Tier 3 against version 22: 6 won, 0 lost (p = 0.03). Fresh seeds against
version 21: 12 won, 1 lost (1003160), p = 0.003. Of tier 3's steps, 67%
continued an intention (2,596 of 3,863); the triggers that ended them:
cannot continue 262, newly seen 198, done 82, surprise 0 (the pose,
front and hand were always as predicted).

1. **Tier 3:** met (30/30 against 24/30).
2. **Tiers 1–2:** met (200/200; 100/100).
3. **Fresh seeds:** met (99/100 against 88/100, p = 0.003).

## 8. Decision

**Keep** (overnight authority): architecture version 23. The intention
removes the loops that re-planning from the top caused, tier 3 is solved
on every test seed and 99 of 100 fresh ones. Not tested here: whether the
surprise trigger works (it never fired), and the intention's cost saving
(steps were already cheap after 095.3).
