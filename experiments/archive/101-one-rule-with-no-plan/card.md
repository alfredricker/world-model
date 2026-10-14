---
id: "101"
title: one rule for acting with no plan, where the outcome is least known (own tries counted), the choice held
rung: 1
serves: [P15, P14, C2]
status: done
verdict: pass
arch_version: 27
date: 2026-10-10
---

# 101: one rule for acting with no plan

Drafted 2026-10-10 at the user's request ("propose card 101"). It follows
the audit of version 24 and cards 099–100. Two parts of version 26 answer
the same question, what to do when the planner finds no plan and nothing
unseen is left to look at: card 095.7's inform and card 072's trying.
Traces of the fresh seeds lost by versions 25 and 26 and by card 100's
arm A show the two parts looping.

## 1. Question

With no plan and nothing unseen to walk to, can one rule replace both
parts without any tier worse and without repeating acts that teach
nothing? The rule acts on the token, action and hand whose outcome arm C
knows least, counting this episode's own tries, and holds that choice
until the act is made. Serves P15 (seek experience that resolves
uncertainty), P14 and C2.

## 2. What changes

One component: the action taken when there is no plan and nothing unseen
to walk to. Exploring the unseen and the door next to never-seen places
(cards 057 and 060) still come first.

| | Version 26 | This card |
|---|---|---|
| Candidates | Inform: tokens in the believed 13 × 13 window under the present hand, where arm C predicts a change (P ≥ 0.5) and is unsure what the token becomes (certainty below 0.8). Trying, when inform has none: ways arm C predicts will not work, under the empty hand, the present hand or a token a pick up could give | Both sets: every (action, token, hand) with pick up or toggle, a non-walkable token in belief, and the hands trying considers |
| How uncertain | Inform: from the store alone, cached for the run, so this episode's tries never count. Trying: P(works) | U = 1 − max(P(no change), P(change) · c). P is arm C's prediction with this episode's own tries (β, card 099). c is 095.7's share from the store: how sure arm C is of what the token becomes |
| Which first | Inform: the nearest. Trying: P / (1 + steps of a plan to the goal under the hypothesis that it works) | The highest U / (1 + steps to make the act): get the hand if needed, face the token, act |
| Held | Inform: no; it is chosen again every step. Trying: its hypothesis is kept until tried or 100 steps | The intention holds the chosen act ("hold h", then "face, act") under card 097.2's triggers, until it is made or cannot be continued |
| When nothing is uncertain | Random actions | Random actions (unchanged) |

Own tries count for whether a token changes: after one try, P is 0 or 1
(β at the floor). They do not make the result certain. The store knows
that the same tokens can give different results: a grey box holds a
different key each time, so its toggle has certainty 0.18. One opened
box must not make the next box of that colour look known. A way that
failed once (P → 0) therefore has U = 0 and is not tried again. A box of
an appearance already opened keeps U ≈ 0.8. Trying's planning under a
hypothesis is no longer needed for the act itself, only to reach it.
That plan takes no step through the act's effect.

**The loops**, traced in card 100 (`runs/100/trace_*`), and
instrumented before any run of this card:
- **Seed 1003160**, lost by versions 25 and 26. The goal's plan drops
  the blue key on (13, 6), the cell before the blue door its route
  passes. Then no goal plan is left. Inform's target is a grey box,
  whose contents arm C cannot know. The walk to it crosses (13, 6), and
  the walking field counts the dropped key as crossable, so the walk's
  first act picks the key up. Holding the key, the goal's plan exists
  again, and it drops the key on (13, 6). This repeats 877 times.
  *Corrected:* the first draft said inform chose the key itself because
  own tries were not counted. Inform's own candidates show it never did.
- **Seed 1003184**, lost by version 26. After the boxes are opened,
  inform's nearest target changes with the agent's heading. It turns left
  and right in place for 3,400 steps, with 41 of trying's hypotheses and
  448 random actions among them.

**What this card can change in them.** Holding the chosen act ends
1003184's turning. On 1003160 the held act (open the grey box) would
continue after the key is picked up, so the goal's plan does not drop it
at once. Whether that wins the seed is open. The cause is the goal's
plan, which chooses where to drop without seeing that the dropped key
blocks its own route: the walking field counts the key as crossable, but
crossing it needs the hand the plan has just emptied. That is card 102's
question.

Code: `tools/card101/act.py`, `WM_NOPLAN=1`. It replaces the
fallback after explore and 060's door; card 072's and 095.7's hooks are
not installed.

## 3. Dependencies

- Version 26 (card 100).
- Arm C with own tries (cards 095.3 and 099).
- 095.7's measure of result certainty.
- Card 072's candidate hands.
- Card 097.2's intention and triggers.
- Card 060's exploration order, which is unchanged.

## 4. Data check

How often version 26 reaches this fallback (`runs/100/*b*`):

| | Steps | Inform actions | Trying (planned / made) | Random | Episodes using it |
|---|---|---|---|---|---|
| Tier 1 | 4,859 | 0 | 0 / 0 | 0 | 0 |
| Tier 2 | 3,016 | 0 | 0 / 0 | 0 | 0 |
| Tier 3 | 3,867 | 442 | 0 / 0 | 0 | 30 (opening boxes) |
| Fresh tier 3 | 21,037 | 3,429 (1,754 in seed 1003160) | 41 / 2 | 448 | 100 |

The rule acts on tier 3 only. There, the act that matters is opening
boxes whose contents arm C cannot know. Each box is opened once, so
counting own tries does not change it; the order of boxes may change.

## 5. Feasibility gate

- **Upper bound:** on the three lost fresh seeds (1003115, 1003160,
  1003184), the held act is continued until it is made, and on 1003184
  the turning ends. This checks that the code does what the card says,
  before the main run.
- **Trivial baseline:** version 26 itself on the same three seeds: the
  recurrence counts of criterion 2.

Result of the gate, before the main run:

**Passed after three changes found by tracing** (`runs/101/gate*_tier3.json`,
`trace*_1003184.log`; `tools/card101/act.py`). Seeds 1003115, 1003160 and
1003184. Criterion 2's count is the number of steps on which a belief
state is seen for more than its 50th time.

| | 1003115 | 1003160 | 1003184 |
|---|---|---|---|
| Version 26 (`WM_NOPLAN=measure`) | won, 211 steps, 0 | lost, 1,642 | lost, 1,766 |
| First build | won, 215, 0 | **won**, 201, 0 | lost, 2,674: two held acts alternate |
| + the act held before a deliberation comes first | won, 215, 0 | won, 212, 0 | lost, 2,871: self-defeating targets |
| + no hand of the token's own appearance | won, 215, 0 | won, 212, 0 | lost, 218: the act never made |
| + the act made without the prediction check | won, 215, 0 | won, 212, 0 | **won**, 1,856, 0 |

The three changes, each from a trace of 1003184:
1. **The held act comes first when the agent reconsiders,** while it is
   not made, still uncertain and still planned. "Newly seen" triggers
   fire as the agent turns. A ranking made from scratch each time then
   switched between two acts, each needing a different key in hand. This
   is the intention as the default when reconsidering, as card 100 did
   for ties.
2. **No hand of the token's own appearance.** Under "toggle the red key
   at place 93 while holding a red key", getting the hand picks up the
   target. The act then vanishes unmade.
3. **The act is made without card 029's prediction check.** Card 029's
   planner acts only when the model predicts the act achieves its
   condition. An act chosen because its outcome is uncertain is often
   one predicted not to work, so it was never made: the agent faced the
   yellow door with the grey key and fell back. Card 072 got around this
   by planning under a hypothesis. Here, being made is the act's success.

1003160 is won by the first build. The held act (open the grey box)
carries the agent past the cell where the goal's plan dropped the key,
so the drop-and-pick-up cycle never starts.

## 6. Success criteria and prediction

1. **CHARTER's tiers:** tier 1 ≥ 99%; tiers 2 and 3 not worse than
   100/100 and 30/30; the 100 fresh tier 3 seeds not worse than version
   26's 98/100.
2. **No fallback loops:** in no episode does a belief state (every
   token, the placement and the hand) recur more than 50 times on steps
   whose action this fallback chose. Version 26 breaks this on seeds
   1003160 (about 877 recurrences) and 1003184.
3. **One rule:** card 072's and 095.7's fallbacks are not installed, and
   one module chooses the act.

**Prediction.**
- Tiers 1 and 2 are unchanged, since they never reach the fallback.
- Tier 3: the same 30/30, with a few episodes taking other steps because
  the order of boxes changes.
- Fresh seeds: 1003184's turning loop ends, because the target is held.
  1003160 is open: the held act may carry the agent to the box before
  the goal's plan drops the key again. If the loop persists there,
  criterion 2 fails on it, and card 102 (where to drop) comes first.
- So the success rate probably stays at 98–99/100. The gains are one rule
  in place of two and no turning in place.
- **Risk:** card 072's goal-directed ranking (ways whose success would
  open a plan to the goal) is replaced by "least known per step". Card
  070's decoy world, where trying found the right key in 95–98% of folds,
  is not in CHARTER's evaluation. It is reported as a check, one fold, not
  a criterion.

**Budget.** The gate takes about 3 minutes (three seeds). Tiers 1–3 and
the fresh seeds take about 6 minutes. One decoy fold takes about 5 minutes.

## 7. Result

`runs/101/` (`main.sh`). Version 26's flags without `WM_TRYING` and
`WM_INFORM`, plus `WM_NOPLAN=1`. The baseline is version 26 (card 100's
`runs/100/b_*`). On the fresh seeds, version 26 was re-run with
criterion 2 counted (`WM_NOPLAN=measure`); it gives the same 98/100 as
card 100.

| | Version 26 | This card | Judged |
|---|---|---|---|
| Tier 1, DoorKey-8x8 | 200/200 | 200/200, identical episodes | passes (≥ 99%) |
| Tier 2, BlockedUnlockPickup | 100/100 | 100/100, identical episodes | no different |
| Tier 3, ObstructedMaze-Full | 30/30, 128.9 steps | 30/30, 118.8 steps | no different |
| Fresh tier 3 seeds | 98/100 | **100/100** | no different (two seeds; not significant) |
| Episodes with a belief state recurring over 50 times | 2 (1003160: 1,642 steps; 1003184: 1,766) | **0** | criterion 2 met |

- **Criterion 1: met.** Tiers 1 and 2 never reach this fallback, so their
  episodes are identical. Tier 3 changes in the order the boxes are
  opened: 12 episodes are shorter, 9 longer and 9 the same.
- **Criterion 2: met.** No episode on any tier has a belief state
  recurring more than 50 times on this fallback's steps. Both seeds that
  version 26 lost to loops are won: 1003160 in 212 steps and 1003184 in
  1,856 steps. 1003184 still spends 1,735 steps in the fallback. It
  makes 80 informative acts (boxes and keys opened or picked up) before
  the goal's plan returns, plus 82 random actions when nothing uncertain
  can be planned.
- **Criterion 3: met.** `act.py` asserts that card 072's and 095.7's
  flags are off. One function chooses the act. It still imports 095.7's
  `certainty()` as its measure of how sure a result is.
- **Cost: a few more steps on the fresh seeds.** On the 98 seeds both
  versions win, this card takes 146.1 steps on average against 141.2.
  43 seeds are longer, 24 shorter and 31 the same, which a sign test
  says is not chance (p ≈ 0.03). The likely reason is the ranking:
  version 26's inform took the nearest target, while this card weighs
  how uncertain an act is against the walk to it. It is not a criterion,
  since tiers are judged on success.
- **Fallback use on the fresh seeds:** 6,867 steps (version 26: 9,727).
  824 acts were chosen, 382 of them the act held from before, and 302
  were made.

**Deviations from the card as approved.**
1. **Three changes made in the gate,** found by tracing seed 1003184:
   the held act first, no hand of the token's own appearance, and the
   act made without the prediction check. They are recorded in §5 and
   were all in place before the main run. The third goes beyond the
   card's text, which said the plan to reach the act takes no step
   through its effect. It turned out that the last step, the act itself,
   must be taken whatever is predicted.
2. **The decoy fold was not run.** Version 26's arm C builds its store
   and its network (arm A) from all of memory 066 (095.1's tokens), not
   from the memory the decoy fold filters. So arm C still knows the
   fold's held-out opening tries. The fold would show the planner
   finding a plan from that knowledge, never reaching this fallback, and
   so would not test the risk it was meant to test. A fair fold needs
   arm C's store and arm A rebuilt without the fold's tries. That is
   colour transfer, which is paused until Crafter.

## 8. Decision

**Keep (version 27).** Tier 1 passes, no tier is worse, the fresh seeds
lost to fallback loops are won, and two fallbacks become one rule. The
cost is about 5 more steps per episode on the fresh seeds, from the
ranking. The decoy world's question (does a goal-directed choice of which
key to try matter?) stays open until arm C can be held out on it.

Declared rule: keep if 1–3 hold. Revise if criterion 2 fails only on loops that the
goal's plan drives (the drop of seed 1003160): card 102 first, then
re-run. Also revise if the decoy fold falls far below card 072's. Stop if
a tier is worse.
