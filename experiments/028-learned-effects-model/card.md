---
id: "028"
title: one learned model of what every action does
rung: 0
serves: [P6, P12, P19, C1]
status: approved
verdict:
arch_version: 3
date: 2026-09-28
---

# 028: one learned model of what every action does

## 1. Question

Can the agent learn from its own pixels what each of its actions does,
moves included, and compute every condition of its tree on its own
predictions, well enough to reach the goal with nothing read from the
simulator? Serves P6 (predict what each action changes), P12 (goals as
conditions, down to primitive actions), P19 (important consequences from
few examples: 172–676 door openings per colour) and C1. Follows
[card 027](../027-kinds-from-effects/card.md) (kinds by counting what
actions do in front; keep) and the direction of
[card 022](../022-effects-post-mortem/card.md): every action, moves
included, is known by its effects, and walking is move effects chained
inside the tree, not a separate part.

## 2. What changes

One component: the model the tree's conditions are computed on. The user
chose to replace it for moves and the other actions in one card
(2026-09-28), with the gate checking each part separately. The model is
learned by counting from the agent's own transitions. Its facts are the
view as a grid of tiles (appearance, and kind from card 027) plus the held
tile; the whole room is always in view. Its effects say what each action
does to the facts; its rules say which facts an effect needs when it
happens only sometimes (card 010's evidence over "holding X", "X in view").

| Part | Card 027 | Card 028 |
|---|---|---|
| Kinds | Counted from pixels | Same |
| What a move does | Simulator | Learned: where each tile goes; whether forward passes, by the appearance in front |
| What pick up, drop, toggle do, and when | Simulator | Learned: outcome per appearance in front, rules for the ones that vary |
| Whether a condition holds | Simulator walks and acts | The agent walks and acts in its head, on its facts |
| The tree's labels (growth on demand) | Simulator | The learned conditions |
| Target rule and its checks (card 027) | Simulator conditions | Learned conditions; "remove the thing" edits the facts |

Walking in the head is the learned move effects chained, moving closer by
card 024's measure read in view coordinates; where that gets stuck, the
tree's move ways take over, as now. Predictions are about things ("the key
is two ahead", "the key is in the hand"), not images. Appendix A gives the
details; appendix B the reasoning from the discussion.

Arms, in each world (500 new layouts, 200-step budget, as card 027):

1. Upper bound: card 027's final setup (conditions from the simulator).
2. **Main: everything from the learned model, tree grown with it.**
3. Learned conditions on card 027's tree (grown with the simulator):
   separates growing the tree from acting.
4. Trivial baseline: learned effects without rules; an outcome ever seen
   counts as possible (so a closed door "can open" whenever toggled).

## 3. Dependencies

Cards 027 (kinds; target rule), 026 (tree on demand, depth-first search),
024 (closeness), 016 (egocentric view), 012 (stored experience) and 010
(evidence rule finder, `src/worldmodel/logic_conditions.py`, `1511_01644`).
Literature: LITERATURE.md's current focus (`1110_2211` for effects that
happen only sometimes; OO-MDPs and schema networks for rules over objects).

## 4. Data check

Card 027's experience: 5,000 episodes of random play per world, about
500,000 stored transitions each. Every door toggle and every change is
stored.

| Event | Key world | Switch world |
|---|---|---|
| Forward: moved; turns | 32,526; 67,072 left, 66,163 right | 32,675; 65,287; 65,082 |
| Keys picked up; dropped | 45,357; 44,443 | 43,892; 43,045 |
| Switch flipped; vase broken | 27,660; 4,205 | 26,663; 4,120 |
| Closed door opened, per colour | 172–213, all holding its key | 617–676, all with the switch on |
| Door toggled without opening, per colour | 29–52 tries holding the other key, 347–386 with no key | 234–260 with no key and the switch off, 10–33 holding a key with it off |

Both door rules hold without exception in the stored experience. The
rarest case the rules must separate is "holding the other key" (29 tries
for the red door).

## 5. Feasibility gate

- **Facts from pixels:** every tile's appearance in the view matches the
  simulator's contents on all stored transitions (as card 027).
- **Upper bound:** card 027's final result, conditions from the simulator
  with counted kinds: 100% in both worlds, 16.6 and 17.3 steps.
- **Trivial baseline:** arm 4, predicted near 0% (nothing tells the agent
  to fetch a key or turn the switch on).
- **Each learned part checked before acting**, on transitions from 1,000
  fresh episodes it did not learn from: criteria 1 and 2 below are
  measured first; a failure there says which part to fix before acting is
  read.

Result of the gate, before the main run:

## 6. Success criteria and prediction

1. **Effects.** For every action, the predicted next facts (all tiles of
   the view and the held tile) equal the real ones on at least 99.9% of
   held-out transitions, and the door rules found are exactly: key world,
   "holding the key of the door's colour" for each colour; switch world,
   "a switch that is on is in view".
2. **Conditions.** On held-out situations, the learned conditions agree
   with the simulator's (card 026's exact class on the same tree) on at
   least 99% for every node acting uses.
3. **Acting.** With the tree grown from learned conditions: reaches the
   goal in at least 98% of new layouts in both worlds, random moves at
   most 1% of moves, mean steps when successful at most 1.5 × the upper
   bound's.

Reported with them: the rules found, with their evidence; the tree grown
(nodes, goals expanded) against card 027's; arm 3 against arm 2; how many
conditions acting computes per move, and seconds per layout.

Prediction: 1–3 pass. The world is deterministic, every rule has
hundreds of cases for it and none against, and moves shift or rotate the
whole view. Arm 2 should act within a step of the upper bound, and arm 3
should match it. Arm 4 fails in both worlds. The likeliest trouble is time:
conditions computed in the head are slower than the simulator's. Next,
the rarest case: the key-world rule could come out as "holding any key"
if the tries with the other key were too few to count against it; 29 to
52 per colour, none opening, should be enough. The card cannot show that
the model survives inexact pixel repeats, partial views, noisy effects or
new colours (the transfer card); chaining move effects is still simulating
steps (card 022's open point for P21).

Budget: counting the effects takes about a minute; growing the tree and
acting with learned conditions are unknown until the smoke test (the
simulator version took about a minute per world). Estimate 10–30
minutes; if a full run is over 30 minutes it is handed to the user as
commands.

## 7. Result

## 8. Decision

## Appendix A: the learned model

**Facts.** Card 016's egocentric view: 13 × 13 tiles centred on the agent,
facing up, plus the held place. Each tile's appearance is its exact pixel
tile (card 027); its kind comes from card 027's counting. With a radius of
6 and 8 × 8 rooms, the whole room is always in view. Declared, as in card
027: the view is a grid; the agent is at its centre; "in front" and "held"
are fixed places; a thing looks the same wherever it is.

**Move effects.** For forward, left and right, the declared form is: a move
either leaves the view as it was or sends every tile to one fixed place.
Which place is learned by counting: for each place before, the place after
that holds the same appearance most often, over the transitions where the
view changed. Tiles entering the view take the appearance most often seen
entering there. Whether forward changes the view is learned per appearance
in front (moved, blocked, or episode ended). Expected: forward shifts the
view one row, turns rotate it, and forward passes floor and open doors and
ends the episode on the goal.

**Other effects.** For pick up, drop and toggle: per appearance in front
(and held, for drop), the outcomes seen for the front and held places, as
in card 027. Where an (action, appearance) pair has more than one outcome,
card 010's evidence finder explains which, over atoms "the held tile is
X" and "some tile in view is X", with pairs of atoms allowed. The rule's
outcome rates come from the counts; an outcome with no rule is
predicted at its rate, and a condition treats an outcome as reachable when
a rule gives it a rate of at least one half (declared).

**Conditions on predictions.** A new conditions class answers card 026's
two questions ("does node n hold here?", "does action a achieve goal g from
here?") on facts:

- The goal square (node 0) holds once forward's predicted effect has ended
  the episode.
- A way (parent p, action a) holds when p holds, or when moving closer in
  the head reaches a spot where a's predicted effect makes p hold.
- Spots: every free tile (forward onto its appearance moves the agent) and
  direction in view. The view from a spot is the current view moved by the
  learned move effects; moves change nothing else, by those same effects.
- Closer: card 024's measure (tiles away, then turns), read in view
  coordinates.

Everything that asked the simulator in card 027 asks this class: labels
for growing the tree (evidence test and self-check), the depth-first
search, walking toward targets, move ways' targets, and card 027's target
rule (the dependence test sets the tile to floor in the facts; refused
moves and "did not work" marks use learned conditions). The environment
itself still runs the agent's real actions; only the agent's thinking
stops reading it.

## Appendix B: notes from the discussion (2026-09-28)

- **Not pixel imagination.** In card 027 the user rejected predicting what
  a tile would look like, in favour of the conditional framework. Here
  predictions are about facts (which thing is where, what is held) and
  whether an action helps is judged by the condition above it.
- **Walking is not separate.** The user: walking is an action satisfying a
  goal at the bottom of the tree; it should not need separate machinery
  when conditions are learned, and the agent should not read the simulator
  to learn it. Card 027 kept walking on the simulator as scaffolding; this
  card removes it together with the rest.
- **One card, not two.** Proposed: moves first, then pick up, drop and
  toggle, so a failure could be located. The user preferred one card; the
  gate measures effects and conditions separately to keep that property.
- **Counting decides, networks recognise.** Card 027's lesson: a network
  trained on average error merges rare look-alikes, counting does not. The
  model here is counts and rules; a network enters when things stop
  repeating exactly (the transfer card, where withheld appearances were
  placed by colour, 4 of 33).
- **Rules and relations.** In distribution, the key-world rule is three
  rules, one per colour. One comparison ("the held key's colour equals the
  door's") would say the same and carry to new colours; that is the
  transfer card's question, and the rule found here says whether the
  counting picks the per-colour form.
- **Literature.** OO-MDPs, schema networks and Pasula et al. learn rules
  of this shape over supplied objects; James, Rosman and Konidaris learn
  symbols in the agent's egocentric space. Ours differ in where the units
  come from: appearances and kinds counted from pixels.
