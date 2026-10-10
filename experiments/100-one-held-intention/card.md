---
id: "100"
title: one held intention, the only memory of choices between steps
rung: 1
serves: [P14, P21, C2]
status: done
verdict: pass
arch_version: 25
date: 2026-10-10
---

# 100: one held intention

Drafted 2026-10-10 at the user's request, from the same audit as
[card 099](../099-version-24-without-dead-parts/card.md), which it
follows. It runs on version 25 if 099 is kept, otherwise on version 24.

## 1. Question

Version 24 keeps the agent's choices between steps in four places, added
one card at a time, and the last of them (097.2's intention) covers what
the other three were for. If the intention is the only memory of choices
between steps, do the tiers stay as they are? P14 (carry out a plan and
recover), P21, C2 (one representation per question).

## 2. What changes

One component: what the planner remembers of its own choices from one
step to the next.

| Kept between steps | Since | What it does in version 24 | How often |
|---|---|---|---|
| The achiever per condition (`commit`, card 051) | 10-04 | Reset at every deliberation from that step's plan; the next deliberation tries the same achiever first while it still gives a plan | Every deliberation; not exported |
| The first need per (condition, achiever) (`_order_keep`, card 074.1) | 10-06 | Among an achiever's needs that no found order makes wait, the one pursued last time goes first; never cleared within an episode | `tie_kept` 544 / 1,112 / 2,157 on tiers 1 / 2 / 3 |
| The door to look behind (`look_j`, card 060) | 10-05 | With no plan and no frontier, the last step's door is tried first | Not exported |
| The intention (card 097.2) | 10-10 | The whole chain of conditions is held and continued at its deepest unmet condition; deliberation runs from the top only on a trigger: done, surprise, a newly seen token that can be acted on, or cannot continue | Continued on 3,802 / 2,281 / 2,596 of 4,859 / 3,016 / 3,863 steps |

The first three act only when the agent deliberates, and since 097.2 it
deliberates only on a trigger: 1,057 / 735 / 1,267 times.

There are two arms. The simpler one is kept unless it is worse.
- **Arm A, the intention alone.** The first three are removed. On a
  trigger, deliberation chooses by the planner's own ranking: fewest
  unmet needs then nearest for achievers; found orders, then fewer pick
  ups, toggles and drops, then steps for needs; nearest first for doors.
- **Arm B, the intention as the default when reconsidering** (Bratman
  1987: intentions resist reconsideration). The intention also stores
  the achiever it chose for each condition of its chain (the plan's
  `ops`). On a trigger, deliberation runs from the top as now. Wherever
  it chooses among alternatives, the dropped intention's choice comes
  first while it still gives a plan:
  - an achiever for a condition;
  - which of an achiever's needs goes first, among those no found order
    makes wait;
  - which door to look behind.

  This is the rule of `commit`, `_order_keep` and `look_j`, read from
  one held chain instead of three stores.

What differs from version 24 in arm B: `_order_keep` remembers every
(condition, achiever) of the episode, and the intention only the last
chain. `commit` is cleared when a step falls back, while the intention
keeps the fallback's door chain.

**Not changed:** the four triggers; card 074's found orders, which are
always followed; card 072 and 095.7's fallbacks.

Code: `tools/card100/held.py` (to write), with `WM_HELD=A|B|shadow`. In
shadow mode the agent acts as version 25, and at every deliberation it
records the choice each arm would make and whether any keeper changed
the choice from the ranking alone.

## 3. Dependencies

- Card 097.2's intention.
- Card 051's `commit`, card 074.1's ties, card 060's kept door. These are the rules being merged; all passed in their cards.
- Card 099, for the version this runs on.

## 4. Data check

From the shadow run, per tier:
- deliberations;
- deliberations where a keeper's choice differs from the ranking alone, per keeper;
- of those, the share where the dropped intention's choice equals the keeper's.

`tie_kept` already shows thousands of kept ties. It does not show how
many differed from the ranking.

## 5. Feasibility gate

- **Upper bound:** in shadow, arm B's choice equals version 25's at every deliberation of tiers 2 and 3. Then arm B gives the same episodes.
- **Trivial baseline:** arm A's choices against version 25's in the same shadow run. If they never differ, the keepers do nothing and arm A is the answer without a main run.

Result of the gate, before the main run:

**Run differently from the plan.** A shadow run could not record what
the ranking alone would choose where `commit` keeps an achiever. That
would mean running the search a second time, and the search's side
effects (card 074's test plans, the tie store, the caches) would change
the acting run. So the gate was arm A's episodes against version 25's.
That is the trivial baseline the card names, and arm A is a main-run arm
anyway.

**Arm A** (`runs/100/main_a.sh`, `runs/100/compare_a.json`):

| | Tier 1 | Tier 2 | Tier 3 | Fresh |
|---|---|---|---|---|
| Version 25 | 200/200, 24.3 | 100/100, 30.2 | 30/30, 128.8 | 99/100, 144.5 |
| Arm A | 200/200, 24.3 | 100/100, **37.9** | 30/30, 130.0 | 98/100, **165.3** |

The keepers do something: tier 2 takes 25% more steps and the fresh seeds
14% more. Arm A fails criterion 2.

**Diagnosis, which old keeper does it** (tier 2, arm A with one keeper
put back, `runs/100/diag_keep_*_tier2.json`): with card 074.1's kept
ties put back, 30.2 steps; with card 051's kept achievers, 37.9; with
card 060's kept door, 37.9. The kept ties carry it all. The other two
make no difference here, because the intention already holds the
achiever and the door between triggers.

**Arm B, narrowed by the diagnosis.** The kept achievers and the kept
door are removed as in arm A. Where card 074.1 breaks a tie among an
achiever's needs, the need that the intention's chain pursued after that
condition comes first. The chain is the intention held now, or, while the
agent deliberates, the one just dropped. A chain records each condition
followed by the need pursued first, even when that need was alone, as
card 074.1's store did. What is lost is card 074.1's memory of every
(condition, achiever) weighed earlier in the episode: only the last
chain's are kept.

## 6. Success criteria and prediction

1. **CHARTER's tiers:** tier 1 ≥ 99%; tiers 2 and 3 not worse than 100/100 and 30/30; the 100 fresh tier 3 seeds not worse than 99/100.
2. **No added wandering:** mean steps when successful no more than 5% above version 25's on tiers 2 and 3 and the fresh seeds. Re-choosing after a trigger shows up as steps before it shows up as failures (card 051's seed 403 alternated between two keys).
3. **One memory of choices:** `commit`, `_order_keep` and `look_j` no longer exist; the intention is the only state carried between steps besides belief and memory.

**Prediction.**
- Arm B gives version 25's choices in nearly every deliberation: the
  keepers' choices come from the last plan, which is the intention's
  chain, except where `_order_keep` remembers a (condition, achiever) the
  last chain did not use.
- Arm A differs after "newly seen" triggers (122 / 297 / 198), where the
  ranking can switch to another need of equal cost. On tier 3 this may
  cost steps. Whether it loses episodes is the open question.

**Budget.** Shadow takes about 10 minutes. Tiers 1–3 and the fresh seeds
take about 15 minutes per arm that needs them.

## 7. Result

Arm B: `runs/100/main_b.sh` and `b_tier2` (`WM_HELD=B`, on version 25's
flags), `runs/100/compare_b.json`. Code: `tools/card100/held.py`.

| | Tier 1 | Tier 2 | Tier 3 | Fresh 100 seeds |
|---|---|---|---|---|
| Version 25 | 200/200, 24.3 | 100/100, 30.2 | 30/30, 128.8 | 99/100, 144.5 |
| Arm A | 200/200, 24.3 | 100/100, 37.9 | 30/30, 130.0 | 98/100, 165.3 |
| **Arm B** | **200/200**, 24.3 | **100/100**, 30.2 | **30/30**, 128.9 | **98/100**, 141.2 |
| Arm B, episodes with the same steps and outcome as version 25 | 200/200 | 100/100 | 17/30 | 56/100 |

On tier 3 arm B takes more steps in 9 episodes and fewer in 4 (+4 steps
in all). On the fresh seeds it takes more steps in 24 and fewer in 19
(−14 in all).

**The fresh seed arm B loses** (1003184, which version 25 won in 457
steps), traced step by step (`runs/100/trace_*_1003184.log`):
- At step 156 the agent holds a green ball. It needs the grey key, and
  the key's pick up has two unmet needs: an empty hand, and facing the
  key.
- Version 25's tie store remembers choosing "empty the hand first" for
  that same (condition, achiever) earlier in the episode, so it drops the
  ball by the door.
- Arm B's last chain did not contain that condition. So the tie went by
  cost: it walks to the key holding the ball and drops it there.
- Both open the grey door. Arm B, from where it now stands, finds no plan.
  Card 072's trying fires (41 tries planned), then random actions (448),
  for the remaining 3,400 steps.
- Version 25's own loss (1003160: 1,754 inform actions) and arm A's
  (1003115: 1,833 random actions) are the same kind of failure, the
  planner finding no plan and the fallbacks looping. Arm B's narrower
  memory of ties took one episode onto such a path. It did not add a new
  kind of failure.

1. **CHARTER's tiers:** met. Tier 1 200/200; tier 2 100/100 (the same
   steps in every episode); tier 3 30/30; fresh seeds 98/100 against
   99/100, no different. Paired, arm B loses one seed that version 25 won
   and wins none that it lost.
2. **No added wandering:** met. Steps change by 0.0% on tier 2, +0.1% on
   tier 3 and −2.3% on the fresh seeds.
3. **One memory of choices:** met. `commit` reads empty, the kept door
   is not preferred, and card 074.1's ties read the intention's chain.
   The intention is the only state carried between steps besides belief
   and memory.

## 8. Decision

**Keep:** architecture version 26, arm B. The intention is the one
memory of choices between steps. On re-deliberation, its chain breaks
the ties among an achiever's needs. Card 051's kept achievers and card
060's kept door are gone, since the intention already holds both between
triggers.

Two things are recorded for the user to overrule:
- The decision rule said "stop if both lose episodes". It was read as
  criterion 1 (CHARTER's "worse", by test), which arm B meets. Read
  literally, both arms lost one fresh seed that version 25 won.
- Arm B was narrowed after the diagnosis. The kept achievers and the
  kept door were not rebuilt from the intention, because putting them
  back changed nothing on tier 2.

If more losses trace to the forgotten tie choices, the memory that is
missing is episodic: how the agent resolved the same conflict earlier in
the episode. It belongs with card 096.5's sequences of condition changes,
not as a fourth store.

Learned on the way: card 072's trying is not dead. It fired for the
first time on the tiers in arms A and B, on seeds where the planner
found no plan.

## Out of scope, noted by the audit

- **The surprise trigger has never fired:** 0 on every tier. It
  compares only planned acts with their prediction, and arm C was right
  on all of them. Tier 3's 59 wrong predictions (686 tries) all fell on
  steps without a held intention: inform's walks and the fallback. How
  the intention recovers from being wrong is therefore untested. A later
  card should test it, with a world where an act predicted to work does
  not.
- **Card 072's trying and card 095.7's inform** both act to learn when
  there is no plan. Inform acts where arm C is unsure what a token
  becomes; trying acts on a way arm C predicts will not work. Merging
  them into one rule (act where the result is least certain) is
  card 101.
