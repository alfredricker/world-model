---
id: "105"
title: one transition model over vector states — the series' plan
rung: 1
serves: [P6, P12, P21, P7, P2]
status: approved
verdict:
arch_version: 27
date: 2026-10-13
---

# 105: one transition model over vector states

Opened 2026-10-13 by the user, after reading the version 27 paper: "We need
to describe _all_ changes within a latent space that the model can use to
predict outcomes in the environment and form conditions", "INTEGRATED
LEARNING!", and "it should be as simple as latent transitions of vector
states. There is no 'choice'." The series was approved the same evening for
the overnight session. This card is the plan; each sub-card is its own
experiment.

## 1. Question

Version 27 predicts with three separate parts:
- **Interactions:** arm C, which reads only the front and the hand, has four
  outcome categories and gives four written result relations.
- **Forward:** its own model, with three categories.
- **Turns and steps:** fitted transformations.

Its planner reasons over five hand-written condition forms. Can one model,
the same for every action and for the world's own changes, map a state's
token vectors to the next state's? And can the planner's conditions come from
what that model is found to depend on? Serves P6 (every consequence), P12 and
P21 (conditions learned, chains searched both ways), P7 and P2.

## 2. The model (shared by every card)

```
state:   a set of tokens, each one vector: [what perception gives (encoder vector), position in its frame, present]
         + the action, as one more token; + history tokens (105.3)
model:   a transformer over the set. Attention logits = content + a learned function of the two tokens'
         position difference (same frame only). No absolute position is an input.
output:  every token's next vector, as a change added to its current one
loss:    distance to the next state's vectors (tokens matched by perception: the same place, or the same track)
```

- **Nothing is named.** There is no list of event types, no front, no hand,
  no carried slot and no outcome categories. Copying is attention, moving is
  a change of the position part, and appearing or vanishing is the
  presence part.
- **Moves are actions like any other.** No model is kept per action set.
- **Uncertainty** comes from recall: the changes that the nearest stored
  transitions made, in the same space (cards 105.3 onward).
- **Declared (C3):**
  - perception (tokens, positions, correspondence, the view's motion);
  - attention over relative positions within a frame;
  - the size of the network.
- **Margin (AGENTS.md, 2026-10-13).** A card may lose to version 27's
  special cases by the stated margin; the loss is recorded, not narrowed
  back.

## 3. The cards

| Card | World and tokens | What it adds | Test |
|---|---|---|---|
| [105.1](../105.1-one-model-world1/card.md) | Breaking world 1; tiles of a fixed view | The model itself: moves, blocking, pushes, lamps | Lamps at new columns; the agent's own motion |
| [105.2](../105.2-one-model-crafter-places/card.md) | Crafter; tiles placed in the world's frame (103.4) | Every one of Crafter's 17 actions in one model | Crafting, mining, placing, moving and blocked; a table at new places |
| [105.3](../105.3-objects-and-history/card.md) | Crafter; 104's SAM 2.1 tracks as tokens | Things that move by themselves; history as input | Creatures' next places against "stays" and a random walk; arrows |
| [105.4](../105.4-conditions-by-intervention/card.md) | The models of 105.1–105.3 | Conditions: what a predicted change depends on | Against the evaluator's rules, at new positions |
| [105.5](../105.5-planning-both-ways/card.md) | World 1 and Crafter | A planner over 105.4's conditions, backward and forward | Goals reached: lamps lit, a wood pickaxe made |

**Order.** 105.1 builds the model where everything is visible and exact.
105.2 moves it to Crafter's frames and actions. 105.3 brings in objects and
history once 104's tracks exist. 105.4 reads conditions from whichever model
is best, and 105.5 plans with them. MiniGrid's tiers come when the model
enters the agent (a later card), judged with the margin.

## 4. Series decision

Closed when a planner over learned conditions reaches goals in world 1 and
Crafter, and the model replaces version 27's three parts in an agent that
keeps MiniGrid within the margin.
