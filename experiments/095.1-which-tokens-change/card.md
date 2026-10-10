---
id: "095.1"
title: interactions predicted over the whole state by one rule, offline
rung: 1
serves: [P6, P3, P4, P17]
status: done
verdict: fail
arch_version: 20
date: 2026-10-09
---

# 095.1: which tokens change, and what they become

The first step of [card 095](../095-state-change-prediction/card.md)'s
series. Offline: the agent is not changed, so CHARTER's tiers are not
run; 095.2 puts the rule in the agent.

## 1. Question

Given the state before a pick up, toggle or drop (the attended tokens of
the believed view, each with its place relative to the agent, and the
hand), can one rule shared by every token predict which tokens change
and what each becomes, without being told that only the front and the
hand can change, and predict held-out tries' next state at least as well
as version 20's recall? Rung 1; P6, P3 (a result as a relation, so a new
colour carries), P4, P17.

## 2. What changes

Nothing in the agent. A new predictor, built and judged offline:

```
version 20 recall:  (front, hand, view set) → category (front changed? hand changed?) + result per slot
                    two fixed slots; the same-tokens result is a literal token ("box red")
this card:          state = {(token, place)} ∪ {(held token, "hand")}, action
                    → for every token: P(changes); if it changes, what it becomes
```

Two arms, one rule each for every token, judged on the same held-out
tries (the user, 2026-10-09: "whichever is stronger becomes the prior"
for 095.2, with recall over the episode's own tries on top either way):

- **Arm A, a network (structure as a learned function).** Attention over
  the state's tokens, each its vector, a learned embedding of its place
  and the action, as card 091's router reads tokens. Per token: the
  probability it changes, and, per part of its vector, card 038's four
  ways chosen by learned gates: keep, copy from another token (which one
  by attention over the state), shift by a learned change, or set.
  Trained by gradient descent on tiers 1 and 2 and the decoy world.
- **Arm B, exemplars (structure from recall, Hintzman 1986).** Every
  token of every stored try is an instance: the action, the token, its
  place, and whether and how it changed. A query token is predicted by a
  vote over instances, weighted by card 049's admitted conditions, here
  chosen among: the token's own parts, its place, and the parts of the
  token at each other place (all places relative to the agent; the hand
  is a place). What it becomes is stored as a relation, per part: kept,
  copied from the token at place r, shifted, or set, with the way and r
  chosen from the stored tries as card 038 chooses ways. Nothing is
  trained; a new try counts at once.
- **In both, nothing names the front or the hand:** each place, the
  hand's included, is learned (A) or admitted (B) from the tries. A
  result is read as the nearest token the encoder knows (the snapping
  recall uses). Held-out tries are whole episodes left out; tier 3 is
  never used for fitting.

## 3. Dependencies

Card 038 (the four ways; in every version since 5), 049 (admission of
conditions, used by arm B), 062 (the believed view per stored try; replayed exactly,
card 094.1's `pkeys.py`), 091 (attention over tokens with roles), 094
(attended tokens). Literature: DOORMAX (Diuk et al. 2008), Schema
Networks, `1110_2211` (Pasula et al. 2007), in LITERATURE.md; for arm A,
graph and interaction networks (Battaglia et al. 2016, 2018), to be
added: one function shared across objects predicts each object's next
state; for arm B, Hintzman 1986 (MINERVA 2) and the exemplar models
already behind recall (Nosofsky 1986; Kruschke 1992); the two arms are
the slow and fast stores of complementary learning systems (McClelland,
McNaughton and O'Reilly 1995).

## 4. Data check

Done for the series (card 095, section 3): in every stored interaction,
only the front and the hand change (pick up and drop: both, 33–51% of
tries; toggle: the front only, 1–26%). Still to count per tier: tokens
per state (attended, up to 6 nearest, plus front and hand), tries per
(action, front code, hand code), and the tries whose result is a token
whose code other tokens share (the red and yellow box; doors of several
colours).

## 5. Feasibility gate

- **Upper bound:** the predictor fit on all of tier 2's tries, scored on
  the same tries: exact next state above 99% (the outcome is close to
  deterministic given the front and the hand).
- **Trivial baselines:** "nothing changes"; the action's outcome
  frequencies; version 20's recall on the same held-out tries (category
  and result, own tries first).

## 6. Success criteria and prediction

1. **Next state:** on held-out episodes of tiers 1 and 2 and on tier 3
   (never trained on), log-likelihood of which tokens change at least
   version 20's recall's, and the share of tries whose whole next state
   is predicted exactly (every token, snapped) at least version 20's.
2. **Roles learned, not given:** predicted probability of change at any
   place other than the front and the hand below 0.01 on average, where
   none happens; and on a pick up, the hand's result copied from the
   token in front in over 90% of tries (A: by its copy attention; B: by
   the place r chosen).
3. **Relations carry:** in the situations of tier 2's box failures
   (1002029, 1002092: empty hand, a box in front, the red box in memory),
   the predicted hand after a pick up is the box in front.

Each arm is judged on all three. **The stronger arm** is the one with
the higher held-out log-likelihood of which tokens change, summed over
tiers 1–3 and actions, provided it meets criteria 2 and 3; within 1% of
each other, arm B, since it learns from one try and needs no separate
component (card 096's first question).

**Prediction.** Both arms meet criteria 2 and 3 (these are the data's
regularities, and copying carries colour). Criterion 1: both similar to
version 20 on tiers 1 and 2, better on exact next state where codes are
shared. Arm A generalises better to tier 3 (its places and relations are
smooth functions); arm B is closer on tiers 1 and 2 and may miss on tier
3 where no instance is similar. Tier 3's toggles (boxes that reveal
keys) are the risk for both, since the key's colour is not a relation to
the box and only "set" can give it.

**Budget.** Replay of the states: about 5 minutes. Arm A's training:
about 10 minutes on the GPU. Arm B's admission: about 10 minutes.
Evaluation of both: about 10 minutes.

## Arm C (added 2026-10-09, the user)

After arms A and B were first scored, the user approved a third arm, arm
B's vote with arm A's prediction as its prior, in place of B's flat
change rate ("similar (B) -> A", two steps):

    P(token changes) = (B's weighted changed count + α · A's P) / (B's weighted count + α)

α per action and world by leave-one-try-out likelihood on 20,000
training instances; a changed token's result is B's relation, or A's
where B has no changed neighbours. α came out at the grid's floor (0.01)
everywhere except tier 3's drop (1.8): where B has near neighbours they
decide; where it has none, A's prediction is the only voice.

## Data check result

- Every change in the true window lies in the state (0 missed of
  127,725 / 208,643 / 257,560 on tiers 1–3); tokens per state 12.7 /
  13.5 / 18.2; only places 71 (the front) and 169 (the hand) ever
  change.
- The drop rule needs the floor in front in the state (a dropped token
  appears there), so the state holds every token within 2 steps, not
  only attended ones (`tools/card095.1/tok.py`).
- Box pick ups with an empty hand: 25 in all of tier 2's memory, 2 in
  its held-out episodes, too few to test criterion 3; it is tested on
  counterfactual states instead (below).

## Gate result

Upper bound: not run separately; arm B on held-out episodes reaches
99.84–100% (below), above the 99% bound's purpose.

## 7. Result

Held-out episodes (every tenth; `runs/095.1/results.json`), weighted by
memory's try weights. Log-likelihood per try of which tokens changed;
exact = every token's change and result right:

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Version 20 | −0.00001, 100% | −0.00002, 100% | not run (over an hour; the user: skip) |
| Arm A (network) | −0.056, 99.49% | −0.044, 99.55% | −0.178, 97.65% |
| Arm B (exemplars) | −0.00003, 100% | −0.00036, 99.99% | −0.0014, 99.84% |
| **Arm C (B, A as prior)** | **−0.00001, 100%** | **−0.00006, 99.99%** | **−0.00018, 99.84%** |

Counterfactual states (`runs/095.1/cf_*.json`): 300 held-out tier 2
pick ups with an empty hand, the token in front replaced by a box of
each colour; the share where the predicted hand holds that box:

| Box | Version 20 | Arm A | Arm B | Arm C |
|---|---|---|---|---|
| blue, yellow, purple | 24% (76%: the red box) | 92–93% | 96–100% | **100%** |
| red | 100% | 93% | 96% | **100%** |
| grey, green | 93–96% | 92–94% | 61–66% | **100%** |

Roles, learned and not given (all arms): mean predicted change at other
places 0.000001–0.000068; on a pick up the hand's result is copied from
the token in front in every case (arm A's copy attention above 0.5 in
100% of tries; arm B's chosen place the front in 1,268 / 8,904 / 16,908
of 1,268 / 8,904 / 16,908). Arm B admits, without being told: pick up,
the token in front and the place (plus the hand on tiers 2–3); drop,
the hand, the place and the token in front; toggle, the token in front,
its own appearance and the hand. Tier 2 and 3 add one or two weak
nearby places (as recall's weak view conditions).

1. **Next state, at least version 20's:** not met as written. Arm C
   equals version 20 on tier 1 and is 0.00004 nats per try lower on tier
   2 (99.99% against 100% exact: a few pick ups in 28,938 tries);
   version 20's router was trained on all of tier 2's keys, held-out
   ones included, so the comparison slightly favours it. Tier 3 has no
   version 20 figure. Arm A misses by more (97.65–99.55%).
2. **Roles learned:** met by every arm (below 0.01 elsewhere; the hand
   copies the front in over 90%: 100%).
3. **Relations carry:** met by arm C (100% for every box colour); arm A
   92–94%; arm B fails on grey and green (61–66%: too few near
   neighbours, so it doubts the pick up happens); version 20 24% on
   three colours.

**The stronger arm** by the card's rule (held-out log-likelihood,
provided criteria 2 and 3 hold): arm C. Cost, for 095.2 and 095.3: arm
B's vote reads 14,300 / 52,604 / 127,100 groups on tier 3's pick up,
drop and toggle (from 1.2 million distinct instances per action), and
its fit took 24 minutes there.

## 8. Decision

**Keep** (the user, 2026-10-09: "keep 095.1 with arm C, this is looking
promising"). Arm C (arm B's vote over stored token instances, arm A as
its prior) becomes the prediction 095.2 puts in the agent. It finds by
itself that only the front and the hand change, gives results as
relations (the red-box error is gone in every colour), and predicts
held-out tries within 0.01% of version 20, which knows the two slots by
construction. Verdict fail only on criterion 1's "at least version
20's", by a few tries in 28,938.
