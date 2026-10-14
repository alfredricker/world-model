---
id: "094.2"
title: recall's prior reads only the conditions learned to matter
rung: 1
serves: [P6, P10, P19, P17]
status: done
verdict: uninterpretable
arch_version: 20
date: 2026-10-09
---

# 094.2: the router on admitted conditions

Card 094's second revision, after [094.1](../094.1-relative-position-recall/card.md)
stopped. The direction (the user, 2026-10-09): the agent attends to the
few tokens it has learned to matter, not to everything in view, and
memory keeps a token's place only once a condition on it is learned.

## 1. Question

Recall already learns which conditions matter for each action. A
condition (a part of the front or held token, the front–held relation,
or "a token with this appearance is in view") is admitted only while it
raises the held-out likelihood of the stored outcomes by more than
log(number of candidates). Recall's evidence reads only those. Its prior,
card 091's router, does not: it compares stored tries on every token in
view (at most 6). Tier 2's last failures come from that prior: 19
successful box pickups with an empty hand are voted down because other
tokens in their views differ. If the router reads the same admitted
conditions as the evidence, does it predict held-out tries at least as
well, do those failures go, and does no tier get worse? Rung 1; P6, P10,
P19, P17.

## 2. What changes

One component: the router's input (recall's prior).

```
version 20:  router key = (front, held, every token in the believed view, at most 6)
this card:   router key = (front, held, the view tokens an admitted view condition reads, at most 6)
             per action, from that world's own admission (fitted at setup from its memory, as now)
```

- With no admitted view condition (pick up on tiers 1 and 3), the key is
  front and held alone, and the vote within card 091.1's identity group
  becomes every stored try with the same front and held tokens, any view:
  the back-off's "same tokens, any view" level.
- **Retrained, not filtered:** card 094 found that feeding the old router
  filtered views breaks it. Its stored keys are rebuilt with the same
  filter, and it is trained as card 091's (tiers 1, 2 and the decoy
  world, never tier 3; 4,000 steps).
- **Unchanged:** admission itself, recall's evidence (own situation
  first), the vote and its learned τ and α, identity groups, forward's
  prior (card 087), the planner, walking.

**Admission's weak conditions.** On tier 2, pick up admits seven view
conditions (a yellow, blue, green or grey ball; a green, red or yellow
key in view), with gains of 4 to 28 against 18,363 for the front–held
relation, just above the threshold (about 4). They may be chance
correlations of random play. This card leaves admission as it is, and
the data check measures whether these conditions separate the 19 stored
box pickups from the failing situations. If they do, admission's
threshold is the next question, not this card's.

## 3. Dependencies

Card 049 (admitted conditions; passed, in every version since 9), 091
(router, training), 091.1 (identity groups), 087 (forward's prior), 094
(lesson: retrain, do not filter, the router's input). Literature
(LITERATURE.md, cards 037 and 049): Kruschke 1992 (ALCOVE), attention
per dimension learned from error, near zero on dimensions that do not
tell outcomes apart; Rescorla–Wagner and Kruschke 2001 (EXIT), a cue
earns weight only by predicting what the others leave unexplained;
`1110_2211` (Pasula et al. 2007), rules name only the objects they need.

## 4. Data check

Per tier (1, 2, 3) and action:
- the admitted view conditions, and the stored keys before and after
  the filter (fewer keys means a cheaper vote);
- tier 2's five failures (card 091.1's trace; seed 1002029's believed
  views): the admitted view values in each failing situation against
  those of the 19 stored box pickups with an empty hand.

## 5. Feasibility gate

- **Upper bound (fit):** held-out likelihood, card 091's evaluation
  (combination and front hidden), tiers 1 and 2 and tier 3 (never trained
  on), against card 091's router and the action frequencies.
- **Targeted:** in the five failing situations, P(the box comes into
  the hand | pick up, empty hand) above 0.5. Version 20: 0.13–0.87.
- **Trivial baseline:** version 20's router; and a router on front and
  held alone (no view), to show whether admitted view conditions add
  anything over none.

## 6. Success criteria and prediction

1. **Prediction:** held-out likelihood per action at least card 091's
   router's on tiers 1, 2 and 3.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than 95% (sign
   test on differing seeds); tier 3 not worse than 14/30.
3. **Cost:** time per step on tiers 2 and 3 not above version 20's under
   equal load.

**Prediction.** Tier 2's four box failures go (094.1's first arm won them
by voting on front and held alone); 1002017 is untraced. Tier 2 97–99%.
Held-out likelihood similar on tier 1 (pick up already admits no view),
better on tier 2. Tier 3 no different, and a little faster, since fewer
distinct keys means smaller votes. The risk: the router's view input
may carry information admission misses (tier 3 toggle admits doors in
view; tier 2 pick up admits balls and keys), and the no-view baseline
will show it.

**Budget.** Data check and key dumps: about 10 minutes. Training and the
gate: about 15 minutes. Tiers 1–2: about 10 minutes. Tier 3: about 2
hours, handed to the user as a command.

## Data check result

Router keys, distinct (front code, held code, view as read) against
version 20's (every key distinct), from `runs/094.2/keys_*_admitted.log`:

| Pick up / drop / toggle | Version 20 | Admitted cut | Admitted view conditions |
|---|---|---|---|
| Tier 1 | 56 / 54 / 55 | 13 / 12 / 19 | toggle: a yellow door |
| Tier 2 | 1,480 / 1,533 / 1,479 | 401 / 185 / 268 | pick up: 4 balls, 3 keys; toggle: 3 keys |
| Decoy | 1,315 / 1,321 / 1,285 | 117 / 123 / 130 | none |
| Tier 3 | 54,119 / 54,001 / 51,817 | 237 / 238 / 2,168 | toggle: 4 doors |

## Gate result

**Fit** (`runs/094.2/*/krouter_*_eval.json`; log-likelihood per held-out
try, the same tries for all three routers, since the keys and their
order are the same):

| Combination hidden | Pick up: 091 / admitted / no view | Drop | Toggle |
|---|---|---|---|
| Tier 1 | −0.165 / **−0.105** / −0.156 | −0.222 / −0.265 / −0.210 | −0.0015 / −0.0040 / −0.0047 |
| Tier 2 | −0.0072 / **−0.0003** / −0.0012 | −0.0003 / −0.073 / −0.0001 | −0.0007 / −0.0011 / −0.0007 |
| Tier 3 (never trained on) | −0.0013 / **−0.0002** / −0.0013 | −0.047 / −0.074 / −0.101 | −0.030 / −0.026 / −0.019 |

With the front hidden, the admitted router is better than card 091's on
every tier and action except tier 1 drop. Drop admits no view condition
on tiers 1 and 2, so the admitted and no-view routers read the same
drop keys; their difference there (up to 0.07) is training variance,
and comparisons of that size do not separate the routers.

**Targeted:** met. On seed 1002029, P(the box comes into the hand |
pick up, empty hand) is 0.68–0.96 in all 1,525 queries (version 20:
0.10–0.96, 71% above 0.5).

## Result (tiers 1–2; tier 3 handed to the user)

| Arm | Tier 1 | Tier 2 | vs version 20 | Steps when solved (tier 2) | s per step (tier 2) |
|---|---|---|---|---|---|
| Version 20 | 200 | 95 | | 119 | 0.026 |
| Admitted cut | 200 | 95 | 4 won, 4 lost (p = 1) | 86 | 0.022 |
| No view | 200 | 92 | 3 won, 6 lost (p = 0.51) | 90 | 0.021 |

- Won: 1002015, 067 and 070 (box failures), now solved in 31–37 steps
  (version 20 ran out of steps on them), and 1002017. Lost: 1002007,
  019, 090, 092.
- **Tier 2's box failures are not the router's.** Traced on 1002029
  and 1002092 (both versions): after dropping the key, recall predicts
  the pick up works (0.77–1.0) but that the hand will then hold *the red
  box*, not the yellow one. Boxes of every colour share one code (code
  id 16), and pick up's stored outcome classes hold one "a box comes into
  the hand" class, recorded as the red box's token. When the agent's own
  situation has tries, recall returns that stored token; the goal (hold
  the yellow box) then looks unreachable and the planner finds no plan.
  When the vote over similar tries decides, the vectors keep colour and
  the plan works. Version 20 escaped this on 1002092 by chance through
  its fallback actions; the admitted arm did not. The lost seeds are
  this gap, or the fallback's luck, not the router; the won seeds are
  where the cut moved the decision to the vote.
- So card 091.1's note (the box failures come from the router's vote)
  is only part of it: the remaining cause is that a stored result names
  a literal token ("the red box comes into the hand") rather than the
  relation ("the token in front comes into the hand").

## 7. Result

1. **Prediction:** pick up better on every tier; drop and toggle mixed,
   within training variance. Not met as written ("at least card 091's
   on every action").
2. **Tiers:** tier 1 met (200/200); tier 2 met (95, no different);
   tier 3 pending (`runs/094.2/tier3.sh`, about 2 hours).
3. **Cost:** met on tier 2 (0.022 against 0.026 s per step); tier 3
   pending.

## 8. Decision

**Stop** (the user, 2026-10-09), superseded by the series
[095](../095-state-change-prediction/card.md), which replaces the router
as recall's prior for pick up, toggle and drop. Tier 3 was not run, so
the verdict is uninterpretable; tiers 1–2 were met (200, 95). Kept from
this card: the router's cut to admitted conditions makes far fewer
distinct keys (tier 3: 54,119 to 237 for pick up), and tier 2's box
failures are a literal stored result ("the red box comes into the
hand"), which 095.1's relational results remove.
