---
id: "095.7"
title: exploring uncertain results — act where recall cannot say what a token will become
rung: 1
serves: [P15, P5, P12]
status: done
verdict: fail
arch_version: 21
date: 2026-10-10
---

# 095.7: exploring uncertain results

Written overnight (2026-10-10) under the user's overnight authority, the
first new card of the night; approved under it. A step of [card 095](../095-state-change-prediction/card.md)'s
series, after [095.3](../095.3-state-change-in-the-agent/card.md) (version 21).

## 1. Question

Version 21 fails 6 of 30 tier 3 seeds, all the same way: the blue ball
lies in a room behind locked doors whose keys are inside boxes, the ball
is never seen, so the planner has no plan for 3,600 steps. Toggling a
box always changes it (2,889 of 2,889 stored tries) into a key whose
colour varies (444–515 tries per colour, six colours), but arm C returns
one result (the commonest key), card 072's guesses skip actions recall
already predicts will work, and no plan asks for a box's contents. So
the agent opens a box only by a random action. If, when there is no
plan and nothing unseen to walk to, the agent acts on the nearest token
whose *result* recall is unsure of (GOAL P15: "seek experience that
resolves the agent's own uncertainty"), do tier 3's failures fall?
Rung 1; P15, P5, P12.

## 2. What changes

One component: a step in the fallback that runs when there is no plan.

```
version 21, no plan:  explore (unseen places; doors next to them) → guesses (card 072) → random
this card, no plan:   explore → act on a token with an uncertain result → guesses → random
```

- **An uncertain result.** Arm C's result is decided per part of the
  vector by the way (keep, copy, shift, set) with the most neighbour
  weight. For shift and set, the winning value's share of that way's
  weight measures how sure the result is; keep and copy are sure.
  Certainty = the smallest share over parts. A token is a candidate when
  arm C predicts it changes (P ≥ 0.5) and certainty < 0.8.
  A grey box under toggle: about 1/6 (six key colours). A locked door
  with its key, a drop, a pick up: 1 (one value, or a relation).
- **The step:** among candidate tokens in the believed map (toggle and
  pick up with what is held now), the nearest by walking's closeness;
  walk to face it (the planner's own walk), then act.
- **Unchanged:** arm C's predictions and results, the planner, frontier
  exploration and card 060's door, card 072's guesses, moves.

## 3. Dependencies

095.3 (arm C, version 21), cards 057/060 (exploration), 067 (walking),
072 (guesses). Literature: information-seeking exploration (GOAL P15);
Bayesian surprise and information gain (Itti and Baldi 2009; Houthooft
et al. 2016, VIME); count- and uncertainty-based exploration in
model-based RL (Pathak et al. 2017; Sekar et al. 2020, Plan2Explore).

## 4. Data check

Tier 3's memory: toggling a grey box changes it in 100% of 2,889 tries,
into keys of six colours at 15–18% each; pick up of a box 96% (floor in
front). Tier 2's boxes: checked in the gate (empty boxes vanish: a sure
result).

## 5. Feasibility gate

- **Upper bound:** the six failing seeds (1003006, 009, 014, 015, 018,
  027) traced: at least 3 solved, and boxes opened by this step.
- **Trivial baseline:** version 21 (0 of the six).

## 6. Success criteria and prediction

1. **Tier 3:** better than version 21's 24/30 on the same seeds (more
   seeds won than lost), at least 27/30.
2. **CHARTER's tiers 1–2:** tier 1 ≥ 99%; tier 2 not worse than
   100/100.
3. **Cost:** tier 3 seconds per step at most 1.5 × version 21's (0.038).

**Prediction.** 4–5 of the six failures solved (27–29/30); the oscillation
in 1003018 (dropping and picking up the grey key before a ball) may
remain. Tiers 1–2 unchanged (the step runs only without a plan).

**Budget.** Gate: 10 minutes. Tiers 1–3: about 15 minutes.

## Gate result

The six seeds (`runs/095.7/gate_tier3.json`): 5 solved (1003006, 014,
015, 018, 027) in 136–280 steps, the step acting 16–45 times per
episode; 1003009 still loops in exploration. Gate passed.

## 7. Result

`tools/card095.7/inform.py` (`WM_INFORM=1`), `runs/095.7/inform_tier*.json`:

| | Tier 1 | Tier 2 | Tier 3 | s per step (tier 3) |
|---|---|---|---|---|
| Version 21 | 200/200, 24.4 steps | 100/100, 55 | 24/30, 1,677 steps | 0.038 |
| **This card** | 200/200, 24.4 | 100/100, **30.7** | 24/30, **264** | 0.016 |

Tier 3 paired by seed: 5 won (the gate's five), 5 lost (1003000, 002,
013, 020, 023): no different. Every one of the six failures spends
3,575–3,595 steps in exploration: card 060's door next to unseen places
is pursued through a ball in its way, the agent drops what it carries to
free its hand, the dropped token lands on the route, and the next step's
plan, made from the top again, picks it back up (traced on 1003000,
1003023, 1003018). With boxes opened, keys are known, so more doors are
pursued and more episodes reach this loop.

1. **Tier 3 ≥ 27/30, more won than lost:** not met (24/30, 5 and 5).
2. **Tiers 1–2:** met (200/200; 100/100 in 30.7 steps against 55).
3. **Cost:** met (0.016 s per step against 0.038).

## 8. Decision

**Keep** (overnight authority): architecture version 22. No tier is
worse, tier 2 needs half the steps, and the step does what it was for:
the five seeds whose keys were inside boxes are solved, none by random
action. The five it lost are a separate defect (re-planning from the
top every step undoes a drop's purpose), which card 097.2's held
intention addresses; with it, tier 3 is 30/30 (097.2).
