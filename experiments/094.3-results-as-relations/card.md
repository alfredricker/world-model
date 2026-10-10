---
id: "094.3"
title: recall's result as a relation to the try's tokens, at the same-tokens level too
rung: 1
serves: [P6, P10, P19, P17]
status: abandoned
verdict:
arch_version: 20
date: 2026-10-09
---

# 094.3: results as relations

**Set aside (the user, 2026-10-09)** before any run, for the series
[095](../095-state-change-prediction/card.md): recall predicts the change
in the whole state with one learned rule, not one mechanism for the
front and one for the hand. Its relational result (what a token becomes,
as keep, copy, shift or set) is carried into [095.1](../095.1-which-tokens-change/card.md).

Found in [094.2](../094.2-router-on-admitted-conditions/card.md)'s trace
(the user, 2026-10-09: "draft the relation card"). Tier 2's last box
failures are not the router's: recall predicts the pick up works, but
that the hand will then hold *the red box* when the agent faces the
yellow one.

## 1. Question

Recall predicts two things for a pick up, toggle or drop: whether the
front and hand change (the category), and what each becomes (the
result). The result has two rules today:

- **Similar tokens** (card 038's transport): per changed place and per
  part of the token's vector, a way learned from the stored tries
  (leave-one-pair-out): *keep* the part, *copy* it from the other place
  (the hand takes the front's), *shift* it by the stored change, or *set*
  it to the stored results' mean. For a pick up, the hand copies the
  front: a relation, so the colour carries.
- **Same tokens** (card 042's rule, used whenever the agent's own
  situation has tries in the category): the stored result of the
  outcome class, a literal token. Classes are keyed by the result's
  code, and boxes of every colour share one code (code id 16), so pick
  up's one "a box comes into the hand" class names the token it was
  first recorded with, the red box.

If the same-tokens level also gives its result through the learned ways
(the class decides *which* outcome; the relation decides *what the
token becomes*), do tier 2's box failures go, and does no tier get
worse? Rung 1; P6 (transfer to new instances), P10, P19, P17.

## 2. What changes

One component: recall's result at the same-tokens level
(`result()` in `tools/card051/index.py`, card 042's rule).

Example: facing the yellow box with an empty hand, pick up.

```
                    which outcome?                   what is in the hand after?
version 20:         "a box comes into the hand"      the token stored with that outcome:   box red
                    (from the exact-match tries)     (recorded when a red box was first picked up)
this card:          the same                         the learned rule, applied to this try: hand ← front = box yellow
                                                     (the rule the similar-tries route already uses)
```

The rule is learned per part of the token's vector (keep, copy from the
other place, shift by the stored change, or set), from the stored tries;
for a pick up it learns "the hand takes the front's token".

- **Where the way is set**, the result is not a relation to the try's
  own tokens (a box opened in tier 3 shows a key whose colour the box
  does not give). There the stored token is kept: the own tries' most
  frequent result, not an average of their vectors, which would name no
  real token.
- **Unchanged:** the category and its probabilities (router, own tries,
  admission), the ways and how they are learned, the planner, walking.
- Card 094.2's router (admitted cut) or version 20's: whichever 094.2
  keeps after tier 3.

## 3. Dependencies

Card 038 (transport and its learned ways; in every version since 5),
042 (the same-tokens level), 051 (index), 091.1 and 094.2 (the trace).
Literature (LITERATURE.md): `1110_2211` (Pasula, Zettlemoyer and
Kaelbling 2007), rules name objects by their relation to the action's
target; DOORMAX (Diuk et al. 2008) and Schema Networks, effects as
changes to the acted-on object's attributes, which transfer across
instances; card 069–070's relations.

## 4. Data check

Per tier (1, 2, 3) and action, over the stored tries:
- outcome classes whose result code is shared by several tokens (the
  red and yellow box), and the stored tries in them;
- for each stored try, the ways chosen for its category and place
  (keep, copy, shift, set).

## 5. Feasibility gate

- **Upper bound (fit):** for every stored try in a class from the data
  check, predict its result from the other tries of its own situation
  (leave one out): the share whose predicted token is exactly the token
  observed, by the literal rule and by the relational rule. The
  relational rule must not be lower on any tier and action.
- **Targeted:** on 1002029 and 1002092, after the key is dropped, the
  predicted hand after picking up the yellow (blue) box is that box.
- **Trivial baseline:** version 20's literal rule.

## 6. Success criteria and prediction

1. **Prediction:** the gate's exact-token share at least version 20's on
   every tier and action.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tier 2 not worse than the best
   version (95%, sign test on differing seeds); tier 3 not worse than
   the best (14/30, or 094.2's if better).
3. **Cost:** time per step on tiers 2 and 3 not above the base's under
   equal load (transport over a few own tries is cheap; the result is
   cached per query as now).

**Prediction.** Tier 2's box failures go (1002029 and the seeds lost to
this gap in 094.2: 1002090, 092 and perhaps 007 and 019): tier 2 98–100%.
Tier 1 no different. Tier 3 may gain where doors of several colours
share a code: toggling a locked door now predicts that door open, not
the first-recorded colour's. The risk is a shift way that lands between
tokens on tier 3, which the gate's exact-token share shows.

**Budget.** Data check and gate: about 10 minutes. Tiers 1–2: about 10
minutes. Tier 3: about 2 hours, handed to the user as a command.
