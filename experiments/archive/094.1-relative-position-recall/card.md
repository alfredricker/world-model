---
id: "094.1"
title: recall reads object files with their positions relative to the agent
rung: 1
serves: [P6, P10, P19, P17]
status: done
verdict: fail
arch_version: 20
date: 2026-10-09
---

# 094.1: recall over object files with relative positions

Card 094's revision (the user, 2026-10-09: "revision as relative position
recall"). Card 094 found that the planner needs no more than object files
hold, and that recall's view, a set of tokens with positions dropped,
gains nothing from floor and walls (they are in every view). The user
and I agreed the spatial element of recall is each object's position
relative to the agent, not the layout, which is for moving.

## 1. Question

Recall's prior (card 091's router, within card 091.1's identity groups)
compares stored tries by the set of tokens anywhere in view. Tier 2's
last failures come from that: 19 stored box pickups with an empty hand,
all successful, are voted down because each had different tokens
somewhere in view (card 091.1's note). If the router instead reads the
object files in view, each with its position relative to the agent, and
learns how much each matters, does recall predict held-out tries at least
as well, do those failures go, and does no tier get worse? Rung 1; P6,
P10, P19, P17.

## 2. What changes

One component: the router's keys and input (recall's prior).

```
version 20:  router key  = (front, held, set of every token within 6 places)       positions dropped
this card:   router key  = (front, held, attended tokens within 6 places, each with (dx, dy) from the agent)
             at most 6, nearest first (version 18's cap); attended as card 094 (memory or card 087 says some
             action changes it, or it ends the episode)
             router input = each token's vector + its role + its relative position (a learned embedding per place)
```

- **Learned relevance, not a rule:** nothing says near tokens matter
  more; the router learns it from which stored tries predict which, as
  attention with relative positions does (`1803_02155`).
- **Its own keys:** the router's stored keys are rebuilt from memory's
  stored views (positions are kept per stored try), merged where front,
  held and positioned tokens are equal, with their outcome counts. Recall's
  other parts (own situation first, admitted conditions, situation index)
  keep version 20's keys. The episode's own tries are added as they
  happen.
- **Unchanged:** the vote (kernel over the embedding, τ and α learned),
  card 091.1's identity groups, own tries first, the planner, walking.
- **Trained as card 091's router:** tiers 1 and 2 and the decoy world,
  never tier 3, each stored key predicted from the rest of its world with
  single keys and whole (front, held) combinations hidden; 4,000 steps.

## 3. Dependencies

Card 091 (router, training), 091.1 (identity groups), 094 (object files:
attention; the planner reads them unchanged), 044 (tokens as what and
where), 050 (own tries first). Literature:
- `1803_02155` (Shaw et al. 2018): self-attention over relative positions
  between elements, learned per offset; here, each token's offset from
  the agent;
- `1706_01427` (relation networks, Santoro et al. 2017): objects given as
  features with their coordinates, relations learned over them;
- object-vector cells (Høydal et al. 2019) and `2112_04035`: where as a
  vector from the agent, path-integrated, apart from what.

## 4. Data check

Per tier and action, from memory's stored views:
- stored keys now (set form) against keys with positions, merged where
  equal; the router's vote cost grows with them;
- attended tokens per view, and their distances from the agent;
- tier 2's stored box pickups with an empty hand: how their nearest
  attended tokens compare with the five failing situations.

## 5. Feasibility gate

- **Upper bound (fit):** held-out likelihood, card 091's evaluation
  (combination and front hidden), on tiers 1 and 2 and on tier 3 (never
  trained on), against card 091's router and the action frequencies.
- **Targeted:** in the five tier 2 failures' situations (card 091.1's
  trace, seed 1002029's believed views), P(the box comes into the hand |
  pick up, empty hand) above 0.5 in every view. Version 20: 0.13–0.87.
- **Trivial baseline:** version 20's router (card 091).

## 6. Success criteria and prediction

1. **Prediction:** held-out likelihood per action at least card 091's
   router on tiers 1 and 2 and on tier 3.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than 95% (sign
   test on differing seeds); tier 3 not worse than 14/30.
3. **Cost:** time per step on tiers 2 and 3 not above version 20's
   under equal load; the router's stored keys reported.

**Prediction.** Tier 2's box failures go, since the stored pickups'
nearest tokens resemble the present ones even when distant tokens
differ: tier 2 rises to 97–100%. Held-out likelihood is similar on tier
1, better on tiers 2 and 3. The risk is cost: positioned keys merge less
(perhaps 2–5 times more keys), so each identity group's vote is larger.

**Budget.** Data check and key dump: about 10 minutes. Router training
and the gate: about 15 minutes. Tiers 1–2: a few minutes. Tier 3: about
2 hours, handed to the user.

## Data check result

Memory's stored windows keep places, but its believed views were stored
as sets. The play was replayed with the same seeds (`tools/card094.1/pkeys.py`)
and card 062's belief replay kept each place; the replayed actions,
windows and believed sets equal the stored ones in all four worlds.

| Router keys (pick up / drop / toggle) | Set form (version 20) | With places |
|---|---|---|
| Tier 1 | 56 / 54 / 55 | 9,130 / 9,296 / 8,794 |
| Tier 2 | 1,480 / 1,533 / 1,479 | 72,199 / 73,038 / 53,486 |
| Decoy | 1,315 / 1,321 / 1,285 | 82,194 / 81,709 / 57,690 |
| Tier 3 | 54,119 / 54,001 / 51,817 | 96,164 / 95,868 / 73,101 |

With places almost every try is its own key (the card predicted 2–5
times more keys; tiers 1 and 2 have about 50 times more).

## Gate result

**Fit: better on nearly every comparison** (`runs/094.1/prouter_eval.json`;
log-likelihood per held-out try, 4,000 tries per action, card 091's
router on the same tries):

| | Pick up | Drop | Toggle |
|---|---|---|---|
| Tier 1, combination hidden | −0.175 (091: −0.274) | −0.176 (−0.202) | −0.0005 (−0.018) |
| Tier 2, combination hidden | −0.0002 (−0.0073) | −0.0001 (−0.0003) | −0.0020 (−0.0021) |
| Tier 2, front hidden | −0.053 (−0.011, worse) | −0.238 (−0.380) | −0.0023 (−0.0033) |

Forward's keys are the same in both forms; its gain (−0.001 against
−0.35) is retraining, and the agent's forward prior is card 087's.
Tier 3's held-out evaluation was not run.

**In the agent** (`tools/card094.1/posrecall.py`, `runs/094.1/pos*_tier*.json`):

| Arm | Tier 1 | Tier 2 | vs version 20 | s per step (tier 2) |
|---|---|---|---|---|
| Version 20 | 200 | 95 | | 0.026 |
| Positional router for every query | 200 | 84 | 4 won, 15 lost (p = 0.019) | 0.35 |
| Positional router for seen and imagined views; version 20's for stored situations | – | 81 | 0 won, 14 lost | 0.061 |

- **Most of recall's queries have no places.** The planner judges the
  situations in which an action could work from *stored* situations
  (card 047), identified by their token sets. In the first arm 1.8 of
  2.0 million queries reached the positional router with no view tokens
  at all, so it voted on the front and held tokens alone.
- **That accident probably won the box pickups.** The 4 seeds won
  (1002015, 029, 067, 070; four of version 20's five failures) were won
  in the arm where most queries ignored the view and voted over every
  stored try with the same front and held tokens. When the stored
  situations went back to version 20's vote (third arm), none of the
  four was won. This points to the back-off's "same tokens, any view"
  level (card 096), not relative positions; it is inferred from the two
  arms, not tested directly.
- **Mixing the two priors is worse.** With places only for seen and
  imagined views, a stored situation judged workable by one vote is
  judged otherwise in the present by the other, and the agent loops (5
  seeds explore 374–475 of 576 steps; none of the box failures is won).

## 7. Result

1. **Prediction:** met on tiers 1–2 except pick up with the front
   hidden on tier 2; tier 3 not measured.
2. **Tiers:** tier 1 met (200/200); tier 2 **not met** (84 and 81
   against 95).
3. **Cost:** not met: 2.3–13 times slower per step on tier 2; 50 times
   more stored keys.

## 8. Decision

**Stop** (the user, 2026-10-09). Relative positions predict held-out
tries better, but the planner reasons over stored situations that have
no places, so positions reach only part of recall, and the part they
reach disagrees with the rest. Giving stored situations places would
change the planner's situations too, a second component. Raw offsets
also make almost every try its own key, so nothing groups. The direction
taken instead (the user, 2026-10-09): recall reads only the tokens
learned to matter, and a token's place is kept only once a condition on
it is learned, as a relation between tokens, not a coordinate. Next:
[094.2](../094.2-router-on-admitted-conditions/card.md); the rest goes to
[096](../096-structure-memory-by-consolidation/card.md).
