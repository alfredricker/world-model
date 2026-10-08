---
id: "086"
title: version 19, recall's conditions over roles and relations, in the agent
rung: 1
serves: [P3, P12, P4, P19, P18]
status: approved
verdict:
arch_version: 18
date: 2026-10-08
---

# 086: version 19, general conditions in the agent

Approved by the user (2026-10-08: "card 086 looks good"). Set with the
user on 2026-10-08: a simplicity bias in recall, judged by the CHARTER
tiers and by a lower bar on transfer ("every room right
within a few tries"), in place of the first-sight criterion of cards
079–084.4. Card 085.3 makes tier 3 buildable first.

## 1. Question

Version 18's recall names particular tiles: a key opens a door as one
fact per colour, and tier 2's hand table is wrong in 3 of 9 pick-up and
1 of 3 toggle cells (card 084). Card 084.3 found that conditions which
name no tile (what a tile does under the other actions, and cuts on
card 070's relation), admitted before identity, give a cell that holds
only openings for a pair never seen opening, and get tier 2's hand
tables right (9 of 9, 3 of 3). It was tested on tables, not in the
agent. Put into the agent, does it keep tier 1, do no worse on tiers 2
and 3, and find the right key for a door within two tries in every
colour fold? P3, P12, P4, P19, P18.

## 2. What changes

One component: recall's prediction for pick up, toggle and drop. A
level of cells sits between a query's own tries and version 18's
neighbours (card 084):

```
version 18:  P = (N_own + β P_nb) / (|N_own| + β)
version 19:  P = (N_own + β P_1) / (|N_own| + β)
             P_ℓ = (N_ℓ + γ P_ℓ+1) / (|N_ℓ| + γ),  P_k+1 = P_nb
```

- **Conditions** (card 084.3): roles (version 18's recall of what the
  front or held tile does, empty-handed, under the other actions) and
  cuts on `rel:P` admitted first, while each raises the evidence by
  more than log(candidates); identity and view tuples only after, for
  what they leave unexplained. Back-off drops the most specific first.
- **On the category** (card 084.3's diagnosis arm, which passed all
  three of its criteria): evidence and γ are computed on whether and
  how the action changes something, not on the tile it becomes, which
  card 038's ways carry over as now. Card 084.4 showed why: a class
  naming the result tile can never be predicted from other
  combinations.
- **γ by combinations left out** (card 084.4's unit), on the category.
- **Online tries** join their cells as well as their own situation
  (card 084's criterion 3), so one failed try still overrules (card
  072's trying is unchanged).
- Moves, walking, planning, orders and trying are version 18's.

**CHARTER.** Recall stays the predictor: stored tries are never merged,
cells are recall's groups re-read when admission is refitted, no
network is added. Roles come from what tiles do, relations from card
070's projection. Declared priors (C3): the candidate family, the cost
per condition, the two stages, the back-off order.

## 3. Dependencies

Card 084.3 (`tools/card084/lifted.py`: admission, cells, back-off);
card 084.4 (γ by combinations); version 18 (card 074.2); card 085.3
(tier 3's memory builds); card 072's trying; card 070's encoder and P.

## 4. Data check

Per tier and action: stored tries, admitted conditions in order, γ, and
whether identity is admitted. Per fold: the cell the left-out pair
reaches and what it holds.

## 5. Feasibility gate

- **Integration:** with nothing removed, the agent's recall gives every
  cell of card 084's three tables right (56, 28, 42), and the same
  predictions as `lifted.py`'s category arm.
- **Upper bound:** the decoy world with the key known (card 074.2:
  100% in every fold) is reached by version 18; tier 1 100%.
- **Cost:** setup time per tier against version 18's (tier 2: 283 s);
  time per step.

**Gate result** (`runs/086/gate.sh`, `gate_none.json`, `gate_tier2.json`):
**passed.** Decoy memory with nothing removed: toggle 56 of 56, pick up
28 of 28, drop 42 of 42; tier 2's hand tables 9 of 9 and 3 of 3
(version 18: 6 and 2). Identity is admitted for no action; toggle's
first conditions are card 084.3's ("held tile walked onto", cut 2.53,
"front tile walked onto", cut 6.04). γ on the category with
combinations left out: toggle 812, pick up 1,808 (card 084.4: 8,103,
the grid's top), drop 0.0001. Building the level: under 3 s per setup.

**Smoke test before the main runs** (red fold, 10 episodes,
`runs/086/smoke_decoy_red.json`; report only): recall right on 36 of 36
door and key pairs, the red pair included (version 18: 35, the red
pair wrong). But 7 of 10 episodes succeed, against version 18's 10 of
10 on the same seeds (no decoy tries), and the failures run out of
steps. Traced (seed 1001000, `trace_red_1001000.log`): the planner finds
no plan from the first step and every action is the random fallback.
The cause is not recall's prediction: toggling the red door with the red
key is predicted to open it (P = 1.0), but the tile it becomes is
imagined by card 038's ways from the other colours' openings, a vector
0.24 from the real open red door, which is no tile memory has seen and
is judged not walkable (`TK.free_of`). Every other colour's opening
becomes the real open door, walkable. So no route crosses the red door,
and card 072's trying never offers the toggle, since recall already
predicts it works. This is ARCHITECTURE's known limit for tier 2's blue
door (card 068), reached now because the prediction is right. It lies in
another component (the imagined result and its walkability), so the
main runs wait for the user's choice.

## 6. Success criteria and prediction

On card 074.2's seeds, against version 18 (McNemar, p < 0.05):

1. **CHARTER's tiers.** Tier 1 ≥ 99% (200 episodes); tier 2 (100) and
   tier 3 not worse than version 18. Tier 3 has no score yet: both
   versions run on the same 30 seeds.
2. **The lower bar on transfer.** In each of the six colour folds with
   trying (one hue's openings removed from memory, 100 episodes):
   success not worse than version 18's (93–96%), and in every episode
   that opens the fold's door, it opens within its first two toggles
   at that door.
3. **Tier 2's hand.** The tier 2 episodes version 18 failed with a full
   hand beside the key it needs (cards 074.1, 074.2) are reported one
   by one; no new loop appears.

Also reported (not judged): the hue left out of memory entirely
(`--hold hue`, card 069), the stronger transfer test, with its known
confound (the open door of that hue is not known to be walkable);
first-sight P(change) for each fold's pair; P19's curve as card 084.3.

**Prediction.** Tier 1 unchanged. Tier 2 rises (the hand table is
right) but not to 99%: 24 of its 41 failures are untraced. The folds
pass the lower bar as version 18 nearly does (564 of 600 first tries
right in card 072), and the pair's first-sight P(change) rises from
about 0 to above 0.5. Risk: a cell of few tries decides where version
18's neighbours were right, in situations the tables do not cover.

**Decision rules.** Keep (version 19) if the gate and 1–2 hold; revise
if 1 holds and 2 fails; stop if 1 fails.

**Budget.** Gate about 15 minutes. Tiers 1–2 and the folds about 2
hours (card 074.2's), handed to the user as commands. Tier 3: 30
episodes per version of up to 3,600 steps, timed on the first few
episodes before the rest are run; handed to the user.

## 7. Result

## 8. Decision
