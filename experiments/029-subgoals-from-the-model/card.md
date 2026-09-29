---
id: "029"
title: subgoals worked out from the learned model
rung: 0
serves: [P12, P21, P6, C1]
status: done
verdict: pass
arch_version: 4
date: 2026-09-28
---

# 029: subgoals worked out from the learned model

## 1. Question

Given only the goal "reach the goal square", can the agent work out its
subgoals from the model it learned (card 028)? A closed door is in the
way, so the door must be open. The rule that opens it needs the key of
its colour, so "holding the red key" becomes a subgoal, then "facing the
red key", reached by moves. Can it choose the next subgoal from what it
sees, and reach the goal as well as card 028 does? Serves P12 (unmet
conditions become subgoals, down to actions), P21 (chains of conditions,
"A enables B, which enables the goal", not many simulated steps), P6 and
C1. Follows [card 028](../028-learned-effects-model/card.md) (keep), and
the user's priority (2026-09-28): goals and conditions based on the
agent's world model.

## 2. What changes

One component: where subgoals come from and how the next one is chosen.
The learned model (card 028) and walking (cards 024, 026) stay.

| Part | Card 028 | Card 029 |
|---|---|---|
| Where subgoals come from | Evidence over random play: the actions that preceded success, grown on demand | Worked backward through the learned rules: a condition's achievers are the actions whose learned effect makes it true; their unmet needs become subgoals |
| What a subgoal is | "A spot from which this action achieves the parent", named by an action path | A fact about things: the door's tile is open; holding the red key; a switch that is on is in view; facing the red key |
| Which thing to act on | Card 027's target rule (kinds; remove a tile, does the way still hold?) | The tile the rule is about |
| Choosing | The first way possible now, depth first, in evidence order | The first unmet need of the action that achieves the condition, depth first; among alternatives, the fewest unmet needs, then the nearest target |
| Walking | Moving closer; the tree's move ways where stuck | Same; move ways computed in the head; a tile in the way that an action can make passable becomes a subgoal |

In the key world, the chain the agent should work out at the start:

```
episode ended      <- forward, facing the goal square
facing the goal    <- walking; blocked by the closed door
door's tile open   <- toggle, facing the door, needs: holding the key of its colour
holding red key    <- pick up, facing the red key, needs: empty hand
facing red key     <- moves (moving closer)
```

It is recomputed from the top at every step, on the current facts
(teleo-reactive). Holding the wrong key makes "empty hand" unmet, so
"drop it" becomes a subgoal; in the switch world, "a switch that is on is
in view" does. Appendix A gives the procedure; appendix B the addendum on
conditions that carry to new objects (the next card, not tested here).

Arms, in each of four worlds (500 new layouts, 200-step budget, model
from 5,000 episodes of random play, as card 028): key, switch, either
(key or switch opens the door) and both (key and switch):

1. Upper bound: the shortest route (the simulator's breadth-first
   search; evaluator only). Shows every layout can be solved, and gives
   the steps to compare with.
2. **Main: subgoals worked out from the learned rules.**
3. Card 028's pipeline (evidence tree, card 027's target rule, learned
   conditions). Key and switch: card 028's run (same data and layouts);
   either and both: run here.
4. Trivial baseline: the same working backward without rules (an outcome
   ever seen counts as possible), so the door "opens" whenever toggled.

## 3. Dependencies

Card 028 (the learned model and facts; keep), 026 (move ways, depth
first), 024 (closeness), 027 (its two revision rules, reused), 010 (rule
finder). Literature: LITERATURE.md's current focus. `strips` for working
backward (means-ends analysis) and `cs_9401101` for re-evaluating from
the top at every step.

## 4. Data check

5,000 episodes of random play per world. Key and switch: card 028's
table (door openings 172–213 per colour, all holding its key; 617–676,
all with the switch on). The two new worlds, checked with card 028's
code unchanged:

| World | Openings per colour | Rules found | Held-out effects exact |
|---|---|---|---|
| Either | 601–663 with the switch on; 93–135 with the key | "switch on in view"; "held = key of its colour"; never otherwise | 99,270 of 99,270 |
| Both | 54–86, all with key and switch | "held = key of its colour" and "switch on in view", together; none of 552 weighted tries with the key alone, or 3,512 with the switch alone | 101,493 of 101,493 |

The rarest case is the both-world red door: 54 openings.

## 5. Feasibility gate

- **Facts and model:** card 028's checks in all four worlds (facts from
  pixels; held-out effects exact; rules as above).
- **Upper bound:** the shortest route solves every layout within the
  budget; its mean steps are the comparison.
- **Trivial baseline:** arm 4, predicted near 0% (nothing makes the key
  or the switch a subgoal).

Result of the gate, before the main run: passed in all four worlds.
Facts from pixels agree with the simulator on every check. Held-out
effects are exact: 100,991, 99,800, 99,270 and 101,493 transitions, every
one. The door rules are exactly as in section 4. The shortest route
solves all 500 layouts in each world within the budget. Arm 4 reaches the
goal in 0.4–12.6% of layouts. (The script computes the both world's gate
after arm 3, which was stopped; the same checks were read from its saved
numbers.)

## 6. Success criteria and prediction

1. **Subgoals.** In at least 99% of layouts in each world, the agent
   pursues every fact the world's rule requires before toggling the door
   (key: holding the key of the door's colour; switch: a switch that is
   on in view; both: both; either: at least one). It pursues none that
   the rule does not use (a key of another colour; in the key world the
   switch, in the switch world any key). Tiles in the way are reported
   separately, with whether the evaluator confirms they blocked.
2. **Acting, key and switch worlds.** Reaches the goal in at least 98%
   of layouts, random moves at most 1%, and mean steps when successful at
   most 1.1 times the shortest route's (card 028: 1.04 in the key world).
3. **Acting, either and both worlds,** with the same code (only the
   experience differs): at least 98%, random moves at most 1%, steps at
   most 1.25 times the shortest route's.

Reported with them: the chains worked out (examples per world), the
choice made in the either world against the shorter option, arm 2
against arm 3, conditions computed per move, and seconds per layout.

Prediction: 1–3 pass. The model is exact in all four worlds, and each
rule names the thing it needs. Steps in the key and switch worlds should
be near card 028's (16.6, 17.3). In the either world, choosing by the
nearest target should cost a few steps over the shortest route; in the
both world, the order of key and switch may too. Arm 3 may do worse in
the either and both worlds, where its tree has never been grown. Arm 4
fails. Risks:
- **Goal interaction:** dropping a key where it blocks the way to the
  door; the refusal rule should catch it.
- **Walking stuck:** where neither a move way nor a freed tile helps.

The card cannot show transfer to new objects (appendix B), or that the
choice between alternatives is learned. That is chosen by a declared rule
here; learning it from the agent's own costs comes later.

Budget: acting on subgoals should be fast (card 028 acted in 0.02–0.04
seconds per layout). The cost is arm 3 in the either and both worlds,
about 10 minutes each (tree growth). Estimate 20–30 minutes; if over 30,
it is handed to the user as commands.

## 7. Result

Run: `bin/prun python tools/card029/subgoals.py` (numbers in
[results.json](results.json)). A small test first (20 layouts per world,
2.2 minutes) passed everything. The full run was stopped at 30 minutes,
the limit for runs not handed to the user. Every arm had finished except
arm 3 in the both world, whose tree growth was still running (below).
Arm 3 is a comparison, not a criterion.

| 500 new layouts per world | Key | Switch | Either | Both |
|---|---|---|---|---|
| 1. Layouts pursuing exactly what the rule needs | 99.8% | 100% | 99.8% | 99.8% |
| Arm 2, main: success | 100% | 100% | 100% | 100% |
| Arm 2: mean steps (shortest route's) | 16.4 (15.9) | 17.3 (16.1) | 15.5 (14.5) | 21.8 (19.7) |
| Arm 2: steps ÷ shortest (limit) | 1.03 (1.1) | 1.07 (1.1) | 1.07 (1.25) | 1.11 (1.25) |
| Arm 2: random moves | 7 of 8,189 (0.09%) | 37 of 8,634 (0.43%) | 7 of 7,737 (0.09%) | 7 of 10,883 (0.06%) |
| Arm 3, card 028's pipeline: success, steps, random | 100%, 16.6, 0.40% (card 028) | 100%, 17.3, 0.59% (card 028) | 100%, 15.8, 0.32% | not finished |
| Arm 4, no rules: success | 4.4% | 8.6% | 12.6% | 0.4% |
| Conditions computed per move: arm 2 (arm 3) | 6.9 (379) | 6.0 (147) | 8.8 (255) | 9.3 |

**Verdicts.** Criterion 1: pass in all four worlds (at least 99%; no
layout pursued a key of another colour, the switch in the key world, or
a key in the switch world). Criterion 2: pass (key and switch worlds:
100%, random moves under 1%, 1.03 and 1.07 times the shortest route).
Criterion 3: pass (either and both worlds, same code: 100%, random moves
under 0.1%, 1.07 and 1.11 times the shortest route).

**Reported with them.**

- **The chains.** At the start of the first layouts, in the agent's words:
  - Key world: episode ended ← facing the goal (walking blocked) ← tile
    133 shows an open green door ← held = key green ← facing the green
    key ← moving closer.
  - Switch world: the same down to the door, then ← switch on in view ←
    facing the switch that is off.
  - Either world: some layouts take the key, others the switch.
  - Both world: the key first, then the switch (the declared order of
    needs).
- **Tiles in the way.** The closed door was the only one, in all 2,000
  layouts. The evaluator confirmed it blocked every time: without it open,
  the agent could not walk to where it would face the goal.
- **The either world's choice.** The agent opened the door with the key in
  260 layouts and with the switch in 240. In the 426 layouts where one
  option is shorter by the simulator's search, it took the shorter one in
  330 (77%). The declared rule, the nearest target, is not the shortest
  route; arm 2 took 1.0 step more than the shortest route on average.
- **Arm 2 against arm 3.** In the key and switch worlds, arm 2 takes about
  the same steps as card 028 (16.4 against 16.6; 17.3 against 17.3), with
  fewer random moves (7 against 33; 37 against 51). It computes 25–55
  times fewer conditions per move and needs no tree: 0.015–0.024 seconds
  per layout, against 0.02–0.04. In the either world, card 028's tree
  took 3.2 minutes to grow (24 nodes, 6 rounds), then acted as well as
  arm 2 (100%, 15.8 steps). In the both world its tree grew for 24
  minutes, 29 rounds, and was still growing when the run was stopped.
  From round 12 on, a single layout kept pausing on one new goal after
  another (forward/toggle/toggle/left, …/right/pickup, …/drop).
- **Arm 2 never dropped anything and never refused an action** (0
  refusals, 0 failed acts, 0 wrong predictions in 35,443 moves). It never
  picks up a key the rule does not name, so a wrong key never has to be
  put down. Card 028's pipeline, which picks targets by kind, used its
  "drop" way 47 times in the key world.
- **Arm 4** fails as predicted. Without rules the door "opens" whenever it
  is toggled, so the agent toggles it, sees nothing happen, marks the
  place failed, and has nothing left to try.

**What building it showed:**

- **An action uses up some of its own needs:** picking up a key ends "the
  hand is empty". The first version protected those needs too. It refused
  every pick up and reached the goal in 50% of 20 layouts, until only
  conditions met higher in the chain were protected.
- **Every random move came from walking** in two layouts:
  - Layout 487 (all four worlds, 7 moves each). The agent starts boxed in
    by the key in front and the vase beside it. The way out is a long
    detour. Neither moving closer, a move way (at most two moves deep),
    nor opening the door alone reaches the goal from there, so no tile in
    the way qualifies. A random pick up took the key, and the chain
    resumed. This is the one layout that fails criterion 1 in the key,
    either and both worlds: the key was held before it was ever a subgoal.
  - Layout 114 (switch world, 30 moves). The agent stands in the corner
    between the wall and the switch, far from the door. Even with the
    door open in its head, walking cannot find the detour. So the door
    never becomes a subgoal, although the switch it needs is next to the
    agent.

  Both would be solved by choosing a tile in the way when its change makes
  the target reachable by moves at all. The code already runs that search
  over poses in the head, but only to skip move ways.

**Prediction.** As predicted: 1–3 pass, steps near card 028's in the key
and switch worlds, a step over the shortest route in the either world,
and two in the both world. Arm 4 fails. Arm 3 was slightly worse in the
either world (15.8 against 15.5), and its tree did not finish growing in
the both world. Of the two risks, goal interaction never arose; walking
got stuck in two layouts of 2,000, without costing success. The budget
was wrong: arm 2 needs seconds per world, and the time is all arm 3's
tree growth, which exceeded 30 minutes in the both world.

**Not finished: arm 3 in the both world.** To finish it, the user runs
(about 40 minutes; tree growth stops at 40 rounds):
`bin/prun python tools/card029/subgoals.py --worlds both --arm3 both --out runs/029_arm3_both.json > runs/029arm3both.out 2>&1`

## 8. Decision

**Keep** (with the user, 2026-09-28; architecture version 5). Given only
"reach the goal square", the agent works out every subgoal from the rules
it learned by counting: the door, the key of its colour or the switch,
facing it, and the moves. It does so in four worlds with one piece of
code, and never pursues anything the rule does not name. It needs no tree
grown from evidence, and computes a few conditions per move instead of
hundreds. What stays declared: the kinds of condition, the order of
needs, the nearest-target choice, and walking. Walking is now the weakest
part. Next, with the user: the literature review for appendix B, then
the transfer card.

## Appendix A: the procedure

**Conditions** are facts about the agent's facts (card 028):
- the episode has ended;
- the hand shows X;
- X is in view;
- tile i shows X;
- facing one of the tiles T, standing on a free tile.

**Actions as the model knows them.** Each entry of card 028's outcome
table is read as an action with needs and results:
- toggle a closed red door: needs facing a tile showing a closed red door
  and holding the red key (its rule's atoms); the tile becomes an open
  red door;
- pick up a red key: needs facing a red key and an empty hand; the tile
  becomes floor and the hand the red key;
- forward onto the goal square: ends the episode.

Where a rule lists alternatives (the either world), each is a separate
way to achieve the same result. An outcome counts when its rule's rate is
at least one half (declared, as card 028).

**Working backward** (means-ends analysis):
1. For an unmet condition, its achievers are the actions whose result
   makes it true, with the tiles they act on: the tiles showing the
   needed appearance, or the one tile named by the condition.
2. An achiever's needs are checked in a fixed order: the rule's atoms
   (hand, in view) first, facing last, since facing comes right before
   acting.
3. The first unmet need becomes the subgoal, down to depth 6.
4. Among several achievers, the one with the fewest unmet needs is taken
   first, then the one whose first target is nearest by card 024's
   closeness; if it finds no action, the next is tried (backtracking).
5. When every need is met, the agent takes the achiever's action.

**Walking.** Facing tiles T is the bottom of every chain. It is reached
by moving closer (card 024) toward the poses facing T. Where moving
closer gets stuck:
- **Move ways** are tried first: poses from which one move lets moving
  closer succeed, found in the head as card 026's move ways, up to two
  moves deep.
- **A tile in the way** comes second: a tile that some action turns into
  a passable appearance (a closed door opened, a key picked up, a vase
  broken). If, with that change made in the facts, the walk succeeds, the
  change becomes a subgoal, "tile j shows the open door" (card 027's
  dependence test, with the action's result in place of floor).

**Revision rules** (card 027, generalised):
- An action or move is refused when, in the head, it makes a condition
  higher in the chain stop holding or become unreachable (for example,
  dropping a key in front of the door).
- An (action, tile, pose) is marked failed for the episode when acting
  there does not produce the predicted result.

**Declared:**
- the kinds of condition above;
- the order of needs, and the rule for choosing among alternatives;
- depth 6 and two moves deep;
- the rate threshold of one half.

Nothing is read from the simulator except by the evaluator (the shortest
route and the subgoal check).

### As built

The code is `tools/card029/subgoals.py`, on card 028's model, facts,
poses and closeness (`tools/card028/effects.py`). Where it differs from,
or fills in, the plan above:

- **A tile condition** ("tile j shows X") is achieved only by an action on
  what the tile shows now; no chains of changes to one tile. A facing need
  with no tile showing the appearance fails; there is no subgoal "bring X
  into view".
- **An action may use up its own needs.** Picking up a key ends "the hand
  is empty", which it needed. The refusal rule protects only the
  conditions met higher in the chain. The first small test refused every
  pick up until this was fixed.
- **An achiever acts only when the model predicts its action makes the
  condition true.** The refusal check runs only for actions that change
  the facts; moves change none.
- **A pure shortcut:** before computing move ways, a search over poses in
  the head checks that the target can be reached by moves at all. Where it
  cannot (a closed door in the way), moving closer and move ways cannot
  succeed either, so the result is the same.
- **Evaluator:** a tile in the way is confirmed blocking when, in the
  simulator, the agent cannot walk to any pose the agent's target names,
  but can once that tile is passable, using the target poses computed
  with the tile changed. With the goal right behind the door, the agent
  faces it from the doorway, which is standable only once the door is
  open.

## Appendix B: addendum, conditions that carry to new objects

For the transfer card; not tested here. From the discussion with the user
(2026-09-28): the agent should learn that the door's condition is about
colour, a matching relation, while the goal is about the shape of what
is fetched.

- **What an attribute is.** An attribute is any respect in which things
  can be alike or differ, found as a difference that recurs. It is
  general pattern matching, not a list of features. If the same change
  of pixels turns several appearances into several others, that change
  is an attribute. Recolouring turns the red key into the blue key, and
  the red door into the blue door: colour. The change from a key's
  outline to a door's outline is the same for every colour: shape. Both
  are found by the same comparison, and nothing marks colour or shape as
  special. Size, a part's position or a texture would be found the same
  way if they recurred. The appearances then factor into shape × colour,
  found rather than declared
  (`a-theory-of-the-discovery-and-predication-of-relational-conc`; the
  relational bottleneck,
  `the-relational-bottleneck-as-an-inductive-bias-for-efficient`, argues
  for keeping what a thing is apart from how things relate).
- **Which attribute a rule uses is chosen by the evidence.** The goal "get
  a key" varies with shape only: every key is picked up the same way
  whatever its colour, which is why card 027's counting puts the three
  keys in one kind. The door's condition is a match along one attribute
  across different shapes: "the held thing's colour equals the door's".
  A slightly different key is close along the shape attribute, so it is
  probably a key.
- **Rules over relations.** With attributes available, the rule can be
  "same colour" instead of three rules with one colour each; the same
  form covers "same shape", or "shaped like a key".
- **The incentive is compression.** Card 010's evidence already charges
  for every rule and every atom. One relational rule pooling all openings
  (563 in the key world) costs less than three rules with 172–213 each.
  So once relational atoms are offered, the evidence prefers the rule
  that applies to every colour. Applying it to a colour never seen is the
  payoff.
- **Test:** withhold one colour from experience. The relational rule
  predicts that its key opens its door; per-colour rules cannot (P3).
- **A new shape.** An appearance whose shape is close to the known keys
  is probably a key: kind membership as a likelihood from similarity
  along the shape attribute, then confirmed by what picking it up does
  (P5, P15). Card 027's network did the opposite: it placed withheld
  appearances by colour, 4 of 33.
- **Walking toward objects already generalises** once targets are named
  by kind and relation rather than by exact appearance: moving closer
  does not care what the target is.
- **Literature review first.** Before the transfer card is drafted, a
  literature review on generalisable abstractions and pattern recognition
  is in order: how attributes and relations are discovered from few
  examples (relational concept discovery, structure mapping and analogy,
  disentangled factors, abstraction learning in program induction, the
  ARC benchmark's work), and which of these work on exact pixels without
  labels. Done 2026-09-28: LITERATURE.md's current focus.
