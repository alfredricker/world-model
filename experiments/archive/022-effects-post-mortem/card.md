---
id: "022"
title: "post-mortem: from readiness to effects"
rung: 0
serves: [P6, P12, P1]
status: done
verdict:
arch_version: 3
date: 2026-09-28
---

# 022: post-mortem: from readiness to effects

A direction change (CHARTER rule 1), signed off by the user on 2026-09-28.
No rung had been reached, so it was not triggered by a failed rung; the
user decided it directly. This card records why; it runs nothing.

## 1. What was tried

Cards 013–021 made the recursion of conditions reach single moves by
adding two parts beside it:

- **Readiness:** for each way, "would its action work if I stood at this
  pose?", at every pose of the map.
- **Walking:** a learned local step, applied K times from readiness, that
  spreads "within k steps" over the map (value iteration networks).

## 2. What failed, and where

- The step itself works when every cell is supervised with exact answers
  (99.97% of moves closer, card 014; 99.8% / 97.7% egocentric, card 016).
- Learned from the agent's own experience it does not: 61–98% of moves
  closer, acting 48.4% at 30k and at 90k updates (card 016). The step
  learns its own chaining from targets it makes itself.
- Readiness trained where the agent stands is right there and at chance
  elsewhere (card 016's maps check: 5–56% of ready cells found, 6–54%
  false).
- Readiness needs information from far away (card 017's input collisions:
  the same local view with opposite labels), because one output mixes an
  action's local effect (the door opens) with its remote consequence (the
  goal becomes reachable). Cards 018–020 fitted that with evaluator labels;
  learning it from experience was never shown.

## 3. Why the framing is wrong

Neither part is knowledge of cause and effect. GOAL.md asks for it
directly: C6 and P6, "predict how actions change the abstract variables
that matter". Readiness is an effect prediction asked at imagined poses,
invented so that walking could be planned; walking is a separate module
beside the recursion, although P12 says the hierarchy extends down to
primitive actions.

## 4. The new direction

The primitive is the **effect of an action on conditions**, learned from
the agent's own observed outcomes: "if I do a here, condition n changes
like this". Every level of the recursion uses it the same way: to achieve
a condition, find an action whose effect produces it, and make that
action's own conditions the next subgoals.

- **Moves are actions with effects** on spatial conditions (where things
  are relative to me). In the egocentric view these effects are simple and
  uniform: forward brings everything one tile closer unless the way is
  blocked, a turn rotates the view. Walking is these effects chained, not
  a separate part.
- **Readiness disappears.** "Pickup works at pose p" is pickup's effect in
  the state reached by walking to p.
- **Local effect and remote consequence separate.** Toggle's effect is
  local (the door opens); that an open door makes the goal reachable is a
  relation between conditions, which the recursion already carries.
- **Chaining is computed from learned effects**, not learned from
  self-made targets.

The open tension: chaining move effects K times is simulating steps, while
P21 wants deliberation over conditions, not simulated steps. Walking is
meant to become System 1 once learned; that is a later card.

## 5. What stays

The learned condition tree (card 012), the egocentric view (card 016),
architecture 3's patch encoder and spatial map (card 017), and the exact
evaluator. Card 021 is stopped unfinished: it tested readiness, which this
direction removes.

## 8. Decision

**Stop** readiness and the separately learned walking step. The first test
of the new direction is
[card 023](../023-closer-in-view/card.md): conditions mean "approachable
by moving closer in view", checked with exact computation.
