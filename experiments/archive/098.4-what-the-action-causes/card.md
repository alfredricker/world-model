---
id: "098.4"
title: Crafter — what the action causes and what the world does by itself, each against doing nothing
rung: 1
serves: [P6, P5, P2]
status: draft
verdict:
arch_version: 27
date: 2026-10-10
---

# 098.4: what the action causes

Drafted 2026-10-10 at the user's request ("draft the interactions /
cause effect learning card since effects happen in this environment
without action"). The same day it was re-scoped onto tracked tokens, so
it runs after card 098.5, and rewritten without grid terms ("We should
think very generally, as future environments are minecraft and
robotics"). It continues card 098.1, whose one revision is used.

## 1. Question

In Crafter much changes without the agent: cows wander, zombies close
in, skeletons loose arrows that fly on, and food and water run down on
hidden timers. Card 098.1 credited every change to the action, as
version 27 does. Even on the true state it was right on only 39% of
changing events.

The question is about card 098.5's tokens: what stays put, object files
that move, and carried tokens. Can the agent learn from its own play
which changes its actions cause, by comparing what follows each action
with what follows doing nothing in the same situation? And can it
predict what the world does by itself, motion included, as calibrated
probabilities? Rung 1. It serves P6 (action consequences), P5
(calibrated confidence) and P2 (a world that changes by itself).

## 2. What changes

Nothing in the agent; the card is offline. Version 27 credits any change
to the last action. This card replaces that, in the predictor only, by a
contrast:

```
version 27:  P(token changes to o | situation, action)                → all of it credited to the action
this card:   P(o | situation, action) − P(o | situation, do nothing) → the part the action adds
             P(o | situation, do nothing)                             → what the world does by itself
```

- **Tokens and outcomes.** Every token in view or carried either stays,
  becomes another token, moves (an object file's motion, after placing
  has taken out the agent's own motion) or vanishes. Births are outcomes
  too: a token appears at a position relative to the tokens near it,
  such as an arrow beside a skeleton.
- **The predictor** is arm B's vote (cards 049, 095.1). Every stored
  step's tokens are instances, weighted by agreement with the query on
  admitted conditions.
  - The candidate conditions are the token's own vector, the agent's own
    token (its heading, or asleep), tokens near it, and carried tokens.
  - "Near" is a relation between tokens, their relative position,
    admitted by leave-one-out gain. It is not a fixed neighbourhood; card
    094.1 found that raw offsets make every try its own key.
  - There is no network prior: this is arm C with α = 0.
- **The contrast** comes from Cheng's causal power, in LITERATURE. For
  each outcome, the action adds Δ(o) = max(0, q_a(o) − q_0(o)), where
  q_0 is doing nothing. Three quantities follow:
  - **Foresight:** P(the action causes a change) = Σ Δ(o).
  - **Attribution:** after acting, the share of the observed outcome y
    that the action caused is Δ(y) / q_a(y). This is Pearl's probability
    of necessity, when the action and the world act independently, as
    Cheng assumes.
  - **The world's own dynamics:** q_0 itself.
- **Every action, moves included,** is contrasted with doing nothing. A
  move's shift of the whole view is the agent's own motion, which
  placing takes out. What a move causes beyond that, such as a turned
  heading or a zombie that turns to follow, is judged like any other
  effect.
- **Chance, conditions and hidden state.**
  - A change that a condition explains sharpens once the condition is
    admitted: a table needs two wood, and drinking adds water only below
    9.
  - What no condition explains stays a probability: a sapling from grass
    is 0.1.
  - What hidden state decides looks like chance until memory holds that
    state. A cow dies on the third hit, and a skeleton shoots only when
    its reload has run out.

The evaluator's label: a token's change is **caused** when a copy of the
simulator that did nothing instead, drawing the same random numbers,
shows a different outcome (appendix A). None of this reaches the agent.

Code: `tools/card098.4/effects.py`, on card 098.5's play and tracks.
Outputs go to `runs/098.4/`.

## 3. Dependencies

- Card 098.5's object files; this card waits for its decision.
- Card 098.3's tokens, the best so far, stated as such. The asleep
  appearance shares a code with one heading, so asleep steps are
  reported apart.
- Arm B's vote and admission (cards 049, 095.1, 096.1).
- Cheng 1997, causal power (LITERATURE). Seitzer et al. 2021
  (`2106_03443`) contrast with the average over actions. This card
  contrasts with doing nothing, because the planner needs to know
  whether a change would have happened anyway.
- FlowMo-WM (`flowmo-wm`, 2026) separates action-driven dynamics from
  drift that the world adds by itself.

## 4. Data check

From card 098's 100,000 random steps, by the evaluator's state:

| | Steps | Acted-on token changes | Vital changes | Other carried changes |
|---|---|---|---|---|
| Doing nothing | 5,769 | 87 (1.5%) | 699 (12%) | 0 |
| Do | 5,864 | 281 | 889 | 573: sapling +1 (389), wood +1 (184) |
| Place plant | 5,816 | 445 | 702 | 346: sapling −1 |
| Place table | 5,698 | 125 | 703 | 28: wood −2 |
| Make wood sword | 5,918 | 87 | 735 | 5 |
| Any other place or make | about 5,800 each | about 100 each | about 750 each | 0 |

- **Most changes after an action are the world's own.** Vitals tick on
  about 12% of steps, whatever the agent does.
- **Moving things are common.** Cows are in view in 43% of frames,
  zombies in 12%, skeletons in 9.8% and arrows in 8.5% (card 098.5).
- **Random play reaches few effects.** It never made a pickaxe, placed
  stone or a furnace, or collected stone. Deeper effects need the
  agent's own play.
- **The data is card 098.5's 300,000 new steps.** Doing nothing is about
  17,000 of them.

## 5. Feasibility gate

- **Upper bound:** the same vote and contrast on the evaluator's state:
  true kinds, positions and identities, true inventory counts and the
  sleeping flag.
- **Trivial baselines:**
  - credit every change to the action (version 27): criterion 1 at 0.5;
  - "nothing changes": criteria 2 and 3 at 0 skill.
- **Label check:** the share of steps whose world streams diverge
  between the two copies (expected under 1%).

Result of the gate, before the main run:

## 6. Success criteria and prediction

Judged on workers 8–9.

1. **Attribution.** For each token change that follows an action, the
   agent calls the change caused when Δ(y) / q_a(y) ≥ 0.5. Balanced
   accuracy against the simulator's label must be at least 0.90, and at
   least 95% of the upper bound's. The baseline that credits every
   change scores 0.5.
2. **What the action causes.**
   - On caused changes, the most likely caused outcome is the observed
     one in at least 90% of cases, and at least 95% of the upper bound's
     rate.
   - Binned by predicted P(the action causes a change), predicted and
     observed frequencies differ by at most 0.05 in every bin with at
     least 50 cases.
3. **What the world does by itself, motion included.** For changes the
   action did not cause, q_0 must meet two bars:
   - calibration within 0.05 in every bin with at least 50 cases;
   - a Brier skill score at least 90% of the upper bound's.

**Reported, not criteria:** results by action and by kind; motion by
kind (arrows, zombies, cows); skeletons' shots and vitals' ticks; awake
and asleep steps, night, and keys stored 1–3 times; the spread of
predictions against stored occurrences (certain effects sharpen, chance
stays spread); and the same vote without the contrast, which isolates
this card's one change.

**Prediction.**
- **Attribution** comes out near the upper bound. What the world does
  follows doing nothing at the same rate, so Δ ≈ 0. Misses fall where
  tokens hide a condition: asleep steps and night.
- **The action's effects** (wood +1, sapling ±1, plant and table placed,
  wood −2, drink +1, food +6, a turned heading) come out above 90%.
- **The world by itself:** arrows near certain (their look shows their
  direction); zombies mostly predictable from their position relative
  to the agent; cows as chance (half the time they stay, otherwise one
  of four directions). Skeletons' shots and cow kills look like chance,
  because the reload and the cow's health are hidden; memory of the past
  would make them predictable (P2, a later card).
- **Risk:** doing nothing is 1 step in 17, so a rare situation's
  baseline borrows from similar ones and may be rough.

**Decision rules.** Keep if 1–3 hold; the contrast and the world's own
dynamics then go into arm C for Crafter. Revise if only rare situations'
baselines fail (remedy: pool every action's steps for the tokens that
action does not affect). Stop if attribution on the true state is itself
below 0.9.

**Budget.** The vote runs over every token in view on 300,000 steps,
with instances equal on the admitted conditions grouped (card 095.1).
Expect about 20 minutes, timed on one worker first. If it would take
over 30 minutes, it goes to the user as a command.

## 7. Result

## 8. Decision

## Appendix A. How the labels are made

- Before each step, a copy of the simulator steps with "noop" instead.
- The player's own chance draws (the sapling's 0.1) come from a separate
  random stream in both copies, so every other object draws the same
  numbers in both worlds:
  - a cow that walks in would have walked in either way;
  - a zombie that turns to follow the player's new position differs
    because of the action, which is correct.
- A token's change is caused when its outcome differs between the two
  copies, by kind, position and identity for objects, the agent's
  sprite, and the inventory counts.
- Every 10 steps, spawning can draw differently once an action has
  changed the map. Steps whose world streams differ after the step are
  counted and left out.
- Copying the simulator costs about 2 ms per step.
