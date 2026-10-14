---
id: "095.5"
title: the planner's conditions read from arm C's store, the last reader of version 20's situations retired
rung: 1
serves: [P12, P6, P17]
status: done
verdict: pass
arch_version: 23
date: 2026-10-10
---

# 095.5: conditions from the store that predicts

A step of [card 095](../095-state-change-prediction/card.md)'s series.
Drafted from the question card overnight (2026-10-10, after the user
asked to finish cards 091–097 before Crafter) and approved under the
overnight authority. The question card asked for two things; this card
does the first, and the second is moved to Crafter (see the end).

## 1. Question

Since 095.3, arm C predicts what pick up, toggle and drop do, but the
planner still asks version 20's stored situations (the old interaction
stores) which hands to try when it looks for an action's conditions
(card 074.1's conditions, through `templates`) and for what can be made
walkable (card 047's `openable`, through `opened`). Arm C then judges
every candidate. If the candidates come instead from arm C's own store
(the hands under which an action changed a token), does the planner find
the same conditions, with one memory for predicting and for conditions
and no tier worse? P12 (conditions inferred from what experience
shows), P6, P17.

## 2. What changes

One component: where the candidate hands come from.

| | Version 23 | This card |
|---|---|---|
| Conditions of doing a on u | hands of stored tries on tokens like u (`templates`), each judged by arm C | hands under which arm C's store saw a change a token, this episode's own tries that changed something, card 072's guess; each judged by arm C |
| Can u be made walkable | hands of stored tries that made a token walkable (`opened`), judged by arm C | the same candidates, judged by arm C |
| Condition form | card 043's two parts (hand, view) | the hand only: arm C admitted no other place (095.3's gate), so a view part cannot arise |

Code: `tools/card095.5/conds.py` (`WM_COND=1`; `WM_COND=shadow` acts as
version 23 and compares both answers at every call).

## 3. Dependencies

Version 23; arm C's store (095.2–095.3); card 074.1's conditions; card
047's openable; card 072's guesses.

## 4. Data check

Candidate hands per action in arm C's store (tier 3): pick up 1 (the
empty hand), drop 8, toggle 9; tier 2: 1, 12, 10. Every candidate costs
one cached query to arm C.

## 5. Feasibility gate

- **Upper bound (shadow):** both readers answer at every call of
  version 23's run; the answers must agree, or the differences be traced
  to a case where arm C's candidates are right.
- **Trivial baseline:** version 23 (the readers as they are).

**Passed** (`runs/095.5/shadow_tier*.json`): tier 2, 20 seeds, 11,331
condition calls and 364 openable calls; tier 3, 30 seeds, 331,009 and
9,099; **0 differ** on either. Version 23's episodes unchanged (tier 3
30/30 in 128.8 steps).

## 6. Success criteria and prediction

1. **Same conditions:** in shadow, no difference on tiers 2 and 3 that
   is not traced to arm C's candidates being right.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than 100/100;
   tier 3 not worse than 30/30; fresh seeds (1003100–199) not worse
   than 99/100.
3. **Retired:** version 20's situations read 0 times; time per step not
   above version 23's (tier 3: 0.035 s).

**Prediction.** The same answers everywhere, so the same episodes. The
gain is one store answering both questions; the old interaction stores
then have no reader in the agent (only writers), and could be dropped
from memory in a later step.

**Budget.** Shadow 10 minutes; tiers and fresh seeds 15 minutes.

## 7. Result

`runs/095.5/cond_tier*.json`, `fresh_cond_tier3.json` (`WM_COND=1`).

| | Tier 1 | Tier 2 | Tier 3 | Fresh 100 seeds | s per step (tier 3) | Version 20's situations read |
|---|---|---|---|---|---|---|
| Version 23 | 200/200, 24.3 | 100/100, 30.2 | 30/30, 128.8 | 99/100, 144.5 | 0.035 | every condition search |
| **This card** | **200/200**, 24.3 | **100/100**, 30.2 | **30/30**, 128.8 | **99/100**, 144.5 | **0.034** | **0** |

Every episode of every tier, fresh seeds included, is step for step the
same as version 23's (430 of 430).

1. **Same conditions:** met (0 differences in 342,340 condition and
   9,463 openable calls).
2. **Tiers:** met (no tier worse; identical).
3. **Retired:** met (0 reads; time per step 0.034 against 0.035 s).

## 8. Decision

**Keep** (overnight authority): architecture version 24. The planner's
conditions now come from the same store that predicts, so arm C answers
both "what will this do" and "what would make it work". Version 20's
interaction stores have no reader left in the agent but are still built
at setup: toggle's admission alone is 84 s of tier 3's 125 s, and card
091's router (5 s) is never queried. Dropping them needs arm C's own fit
of β (how much this episode's tries outweigh the store; version 20's
fit is borrowed today); a later step.

## Moved to Crafter

The question card also asked for goals and conditions over any change in
the predicted state, since Crafter's goals are inventory counts and
tokens near the agent. In MiniGrid the only state that changes under an
interaction is the front and the hand (card 095, section 3), so the
general form cannot be tested here, and building it before Crafter's
state exists would mean building it twice. It becomes a step of the
Crafter series, after 098's tokens.
