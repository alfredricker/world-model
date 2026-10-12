---
id: "102.6"
title: actions with a where — where an action acts is a learned relation, and an action may carry a place
rung: 1
serves: [P6, P3]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 102.6: actions with a where

Drafted 2026-10-11 as the fifth card of the 102 series, at the user's
request to remove MiniGrid's assumptions in a sensible order.

## 1. Question

An action is one of MiniGrid's six ids, or Crafter's seventeen. Where it
acts is the front, a place the model is told (version 27). After card
102.1 it is a relation the model learns, but still a relation to the
agent's own token.

ARC-AGI-3 has an action that carries a place: a click at a cell. A
robot reaches to a point. In both, where the action acts is given with
the action, not read from the agent.

Suppose an action is a kind together with an optional where. A click
is then "click" at (x, y), and MiniGrid's toggle is "toggle" with no
where. The action's where enters card 102.1's pair function like a
token's: what the action changes is a learned relation between the
action and the tokens. Are clicks' effects then predicted at places and
on shapes never clicked in training? And does Crafter's "do" still find
the faced tile by itself? Rung 1. Serves P6 (effects of every action)
and P3 (a role, "the shape clicked", not a place).

## 2. What changes

One component: what an action is, as arm C reads it.

```
now (102.1): (action id, every token with its where) → each token's change
this card:   (action kind, the action's where if it has one, every token with its where) → each token's change
             the action is one more input to the pair function: its kind, and its raw where when given
```

- **Nothing about the front is declared.** For actions without a where,
  the relation to the agent's token is learned (card 102.1). For actions
  with one, the relation to the action's where is learned.
- **Clicks on objects** act through card 102.2's groups: a click on any
  tile of a shape can change the whole shape.
- **The adapter lists the actions:** each is a kind, and whether it
  carries a place and of what form (a cell of the view, for ARC). This is
  the only thing a new environment states about its actions.
- **Not in this card:** the planner choosing an action's where (an
  achiever with a place to choose; a later card), and continuous wheres.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| Actions act at the front | yes (do, place) | no (clicks) | no | removed: learned relation |
| Actions are discrete kinds | yes | yes | partly | declared by the adapter |
| An action's where is a cell of the view | none | yes | no (a point in space) | declared by the adapter; continuous later |

## 3. Dependencies

- Card 102.1's pair function and arm C; card 102.2's groups.
- Literature: Pasula et al. 2007 (`1110_2211`), rules naming objects by
  their relation to the action's target, which is this card's form when
  the target is given.

## 4. Data check

**Breaking world 5, "clicks"** (`tools/card102/worlds.py`, Appendix A):
- clicking any tile of a shape recolours the whole shape;
- clicking a lamp toggles it;
- clicking an empty cell or a wall does nothing.

Clicks are uniform over the view in random play. Clicks per kind of
target, and positions never clicked in training, are counted when it is
built. Crafter's "do" steps by what was faced come from card 098.5's
play.

## 5. Feasibility gate

- **Upper bound:** the evaluator's rule of the world (what a click at p
  changes), run through the same scoring.
- **Trivial baselines:**
  - "nothing changes";
  - card 102.1's arm C with the click's where dropped. It knows only that
    a click happened, so it cannot say what changes.
- **Check:** in Crafter, card 102.1's version and this card's give the
  same predictions for actions without a where, to within 0.01 nats.

Result of the gate, before the main run:

## 6. Success criteria and prediction

1. **Clicks at new places and on new shapes.** In breaking world 5's
   test worlds (shapes never seen, and positions never clicked in
   training), the whole predicted next state after a click is right in
   at least 95% of the upper bound's rate. This would show that where an
   action acts is a learned relation that carries.
2. **Crafter's "do" finds its tile.** On worlds 8–9, the pair function's
   attention from "do" falls most on the tile the agent faces, on at
   least 95% of awake steps. The effects of "do" are no worse than
   102.1's, within 0.01 nats per try.
3. **Nothing lost in MiniGrid.** On card 095.1's held-out tries, whole
   next states are right within 0.01 points of card 102.1's (or version
   27's, if 102.1 was not kept).

**Reported:** errors by target kind; clicks on a shape's edge against
its inside; positions by how often they were clicked in training.

**Prediction.**
- Criterion 1 holds for lamps and empty cells. Shapes depend on 102.2's
  groups, and fail where a click lands on a shape not yet grouped (before
  it has moved or changed).
- Crafter's "do" attends to the faced tile once heading is read from the
  agent's token (card 098.3: heading codes 100% awake).

**Decision rules.**
- **Keep** if 1–3 hold.
- **Revise** if only shapes clicked before they are grouped fail.
- **Stop** if the relation to a click's where does not carry to positions
  never clicked.

**Budget.**
- Breaking world 5 takes about 2 minutes to build.
- Training and scoring take about 15 minutes, timed first.
- MiniGrid's check takes about 10 minutes.

## 7. Result

Run overnight on 2026-10-12, under the user's portability objective
(`tools/card102.6/clicks.py`, 12,000 steps, about 4 minutes, capped).
- **World 5's room:** it was built in the 9 × 7 room of the other breaking
  worlds rather than 16 × 16, so that one adapter reads every world.
- **Data:** training clicks recoloured shapes 27,454 times and toggled
  lamps 4,054 times, and never landed on column 6. The tokens were made by
  the same encoder recipe: 10 codes, each 100% pure.
- **The model:** the click's cell enters 102.1's pair function as its
  locus input (every token's offset from it), and wait and click are the
  action's two kinds.

| World 5, test (held-out shapes), whole next state right | With the click's where | Where withheld | Nothing changes |
|---|---|---|---|
| All steps (49,500) | **94.6%** | 82.8% | 82.8% |
| Clicks on lamps (998) | **99.8%** | 0% | 0% |
| Clicks on shapes (7,510) | 64.6% | 0% | 0% |
| Clicks on floor or walls (36,000) | 99.8–100% | 100% | 100% |
| Changing clicks in column 6, never clicked in training (1,327) | **73.6%** | 0% | 0% |
| Changing clicks elsewhere (7,181) | 67.8% | 0% | 0% |
| Shape clicks on an edge tile / an inside tile | 64.0% / 85.9% | 0% | 0% |

- **Where an action acts is learned, and it carries.** Clicks in a column
  never clicked are predicted as well as elsewhere (73.6% against 67.8%),
  and lamps are 99.8%. Without the where nothing can be predicted.
- **Shapes fail as predicted.** Recolouring a whole shape needs to know
  which tiles form it. Half the shapes have several colours, and two
  shapes may touch, so one frame does not say; 102.2's groups do, and
  they were not attached here.
- **Not run:** criterion 2 (Crafter's "do", which would need 102.1's
  Crafter training with the action's where) and criterion 3 (MiniGrid).

## 8. Decision

**Revise** (overnight, 2026-10-12), by the card's rule: only shapes
fail, and they fail where grouping is needed. Criterion 1 is not met
(64.6% on shapes against 95%), but the relation to a click's where
carries to places never clicked, which the stop rule asked. The revision
gives arm A each token's group (102.2's, read through a pair layer within
the group), then runs criteria 2 and 3.

## Appendix A. Breaking world 5, "clicks"

- 16 × 16 grids seen whole, with no player.
- Actions: click at any cell, and "wait".
- 3–5 shapes of 2–6 tiles. A click on a shape's tile recolours every tile
  of it to the next colour in a fixed cycle. Lamps toggle when clicked.
- Test worlds draw shapes from a held-out set, and test clicks include
  positions never clicked in training.
- 500 episodes of 100 random clicks each, training and test.
