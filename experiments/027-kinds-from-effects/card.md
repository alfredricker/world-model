---
id: "027"
title: kinds from what actions do in front
rung: 0
serves: [P7, P1, P12, P4]
status: draft
verdict:
arch_version: 3
date: 2026-09-28
---

# 027: kinds from what actions do in front

## 1. Question

Can the agent discover the kinds of things in its world from what its own
actions do to the thing in front of it, name the kind that each way of its
condition tree acts on, and then find that kind anywhere in view well
enough to walk to it and act? A theory check with exact computation over
pixel tiles. Serves P7 (the units worth tracking are found, not supplied),
P1 (a thing means what actions do to it), P12 (conditions are about
things) and P4 (only distinctions that change outcomes are kept). First
card after the theory sessions that followed
[card 026](../026-depth-first-subgoals/card.md); the appendix records that
discussion. Card 026 reached the goal in every layout, but the simulator
told each way where its target was ("the tile holding the matching key").

## 2. What changes

One component: what a way walks to and when it acts.

```
card 026: a way's target = the poses where its action achieves the parent,
          computed by the simulator, which knows which tile is which
card 027: a way's target = poses facing a tile of the way's kind;
          at a target, take the way's action
            kind of a tile: its appearance, grouped with the appearances that
              the agent's actions affect in the same way (from its own frames)
            kind of a way: the kind in front when the way's action succeeded
```

- **Effects, from pixels.** In card 016's egocentric view (the room
  centred on the agent, facing up, plus a place showing what it holds),
  "in front" and "held" are fixed places. Forward shifts the view (moved),
  leaves it (blocked) or reaches the goal (the supplied success signal).
  Pick up, drop and toggle never move the agent; their effect is what the
  front and held tiles became, from the frames before and after.
- **Kinds.** Two appearances are one kind when every action has the same
  effect on them and turns them into things of the same kind: stochastic
  bisimulation (Givan, Dean & Greig 2003), found by splitting groups while
  card 010's evidence supports a split. The number of kinds is not given.
- **A way's kind** is the kind in front at the way's successes in
  experience (for drop, also the kind held). **Recognition:** every tile in
  view (the whole room, card 016) gets its appearance's kind; an
  appearance never seen in front has none.
- **Unchanged from card 026:** the tree, grown on demand; exact conditions
  choosing the way (the depth-first search); moving closer; the
  experience; the 500 new layouts; the 200-step budget. Move ways ("turn
  beside the key") act on no thing and keep exact targets (declared; their
  steps are reported).
- **Two worlds, same layouts:** the key world (the door opens for the key
  of its colour; a key of another colour lies in the room) and the switch
  world (the door opens when the switch is on; keys do nothing).

Tiles are compared exactly, pixel for pixel; a learned encoder (DeepSym's
binary code) is a later card. Declared priors: the tile grid as units; the
fixed "in front" and "held" places; a thing looks the same wherever it is;
the supplied goal's success signal.

## 3. Dependencies

Card 026's pipeline (`tools/card026/dfs.py`, `tools/card023/closer.py`),
its key-world tree (`runs/026_dfs.json`) and result; card 012's exact
discovery and data; card 010's evidence; card 016's egocentric view
(`tools/card016/bench_ego.py`). Literature (papi): `deepsym` (kinds as what
actions do to a thing),
`equivalence-notions-and-model-minimization-in-markov-decisio` (the
coarsest grouping with equal effects, by splitting), `1412_2309` (the
smallest change that flips an outcome locates its cause). Nothing learned.

## 4. Data check

Weighted counts from 5,000 episodes of random play per world (evaluator
labels). 15 appearances occur in front, each faced with each action at
least 93 times (the rarest are the open doors and the goal). Within each
expected kind, colours differ by at most 2 points.

| In front | Effect (share of tries) |
|---|---|
| Key, 3 colours | pick up takes it 90–91% (otherwise the hands are full) |
| Closed door, 3 colours | toggle opens it 5–6% in the key world (holding its key), 21–23% in the switch world (switch on) |
| Open door, 3 colours | forward passes, toggle closes: always |
| Switch, off and on | toggle flips it: always |
| Vase | toggle leaves floor: always |
| Floor; goal; wall | forward passes, drop leaves a key 17%; forward ends the episode; nothing |

In the 500 new layouts, the other key is nearer the start than the
matching key (in tiles) in 41.8%.

## 5. Feasibility gate

- **Effects from pixels** agree with the simulator's events on every
  stored transition (100% expected; the renderer is deterministic).
- **Upper bound:** card 026's acting with the simulator's targets, in each
  world. Key world: 100%, 16.4 steps. The switch world has not been run;
  it must reach at least 98% for criterion 3 to be passable.
- **Trivial baselines:** *contrast only*, where a way's target is any tile
  that stands out (neither floor nor wall), predicted far below: contrast
  gives candidate things, not the relevant one. *Kinds without grouping*,
  where each appearance is its own kind, predicted to work only in the
  third of layouts whose key or door has the way's most common colour.

Result of the gate, before the main run:

## 6. Success criteria and prediction

Same 500 new layouts in both worlds:

1. **Kinds:** the discovered grouping equals the expected one in both
   worlds.
   - In front: floor; wall; goal; closed door (3 colours); open door (3);
     key (3); switch (off and on); vase.
   - Held: nothing; key (3 colours).
2. **Ways name things:** every non-move way that acting uses has one kind
   for at least 99% of its successes, and it is the expected one.
   - Step onto the goal square: goal.
   - Toggle: closed door, vase or switch.
   - Pick up: key.
   - Drop: floor in front and a key held.
3. **Acting, switch world:** reaches the goal in at least 98% of layouts,
   with random moves at most 1% of all moves, and mean steps when
   successful at most 1.5 × the upper bound's.
4. **The relation gap, key world:**
   - Where the first key picked up is the matching one, at least 98% reach
     the goal.
   - At least 90% of the failing layouts picked up the other key.

Reported with them: the key world's overall success; each kind split's
evidence; appearances in view without a kind; steps taken on move ways
(still exact); and each baseline.

Prediction: criteria 1–3 pass, since the data check shows colours behave
alike within each kind. Key world overall about 45–70%, roughly the
layouts where the matching key is reached first. The failures should be a
loop: pick up the other key, drop it in front (the drop way), pick it up
again. That loop is the point. "Key" is a kind; "the right key" is a
relation between the held key and the door, which a next card would add.
Risk: card 026's pipeline has not been run in the switch world (the gate
checks). Budget: the switch world's tree grown on demand (the key world's
is rebuilt from card 026), kinds in seconds, and four acting arms per
world. About 6–10 minutes, under 30.

## 7. Result

## 8. Decision

## Appendix: notes from the theory discussion (2026-09-28)

With the user, after card 026. What follows is the reasoning behind this
card and the ones planned after it.

### A. The starting point

Card 026's tree reaches the goal in every layout, but two things are still
supplied by exact computation:

- the conditions, as sets of simulator states;
- the target of each way ("the matching key is at this tile").

Every condition in the tree is secretly about a thing: step onto the goal
square, open the door, pick up the key, drop the other key. The question
for these sessions: where do those things come from, and how do they
relate to conditions?

### B. Three signals, one question

The user proposed three pillars of abstraction discovery: contrast,
relations and causes. Each answers the same question differently: *when do
two things count as the same?* Literature added to papi on 2026-09-28; the
numbers below were checked in the papers' text.

- **Contrast: they look alike and stand out from their surroundings, or
  they move together.**
  - Contrast within a single image gives candidate pieces, but not their
    size (a car, its door, its handle) or whether they matter.
  - The strong methods (DINO, CutLER, deep spectral methods) take their
    idea of an object from networks pretrained on photo collections, which
    C1 rules out.
  - Moving together settles size. Motion-trained discovery beats
    appearance on real street video: foreground grouping 47.1 against 40.9
    for a classical appearance method and 13.8 for Slot Attention
    (`2203_10159`).
  - EISEN (`2205_08515`) learns "what moves together" as pairwise
    affinities, with no fixed slots. It explains away the agent's own
    motion first.
  - C-SWM got objects that respond to actions, but was told which object
    each action moves.
  - Contrast supplies candidate things. It cannot supply the conditions.
- **Relations: they relate to other things in the same way.**
  - The infinite relational model recovers kinds from relation tables:
    agreement with human categories 0.50–0.59, against 0.38–0.47 from
    features alone.
  - DORA learns properties by comparing examples, and relations by linking
    properties; it predicts properties are learned before relations.
  - Neural relational inference finds who affects whom from motion alone
    (82–99.9% of edges, against 52–63% for correlation).
  - Every relational method starts from given units. What they discover
    sits above the units.
  - The route that fits us is predictive: a relation exists where it is
    needed to predict an effect.
- **Causes: they make the same difference to what an action achieves.**
  - Our condition is a cell of Chalupka et al.'s causal partition
    (`1412_2309`, `1512_07942`).
  - Their coarsening theorem: predicting an outcome from partial views
    gives groups that are finer than the causal ones, never coarser. A
    learned condition may be split in two, but two conditions should not be
    wrongly merged, and one tested attempt per group settles it.
  - Causal abstraction (`1812_03789`, `1707_00819`) gives a validity test:
    every way of making a condition true must have the same effect. Total
    cholesterol fails it. Our version: "holding a key" against "holding the
    matching key".
  - DeepSym found kinds such as "rollable" as on/off codes that predict
    what actions do to an object. Its categories were 99.9–100% pure,
    against 44–87% for an autoencoder, but it was given each object already
    cut out.
  - Saulus et al. (`2606_19594`): a high-level variable is a narrow point
    through which many low-level causes act, identified by "anchors". It
    needs supplied variables and a known number of them.

A wording point: our condition finding already "contrasts successes with
failures", which is the causal signal. "Perceptual contrast" means the
first pillar.

### C. How objects relate to conditions

Agreed with the user:

1. **Objects are what conditions are about**, and they are found *through*
   conditions, not before them. The smallest change to the view that flips
   an action's success shows where its cause is. For "pick up works", that
   is the key in front.
2. **An object, for now, is a pointer** a condition fills when it needs
   one: "the thing in front", "the thing held". There is no standing list
   of objects, which also respects GOAL.md's ban on fixed slots. Robotics
   calls this deictic representation (Agre & Chapman; Ballard); not yet in
   papi.
3. **A kind is the bundle of what actions do to a thing.** A key is what
   pick up takes, and, while held, what makes toggling open the matching
   door. That is GOAL.md's own example for P1.
4. **Relations should fall out of conditions**, not be a separate system.
   The condition language needs only pointers to things and one comparison
   between two pointed-at things. A relation is then a condition over two
   things whose truth depends on that comparison.
   - "The door opens" needs three rules without a comparison (red with
     red, blue with blue, green with green).
   - With a comparison it needs one rule, "held matches door". Card 010's
     evidence test, which charges for every rule, prefers the one rule, and
     only it could cover new colours.

### D. The order of operations

Perception does not come first:

1. conditions over whole views (we have these);
2. find the thing inside each condition;
3. give it a code for what actions do to it (DeepSym's step);
4. relations, where a condition depends on two things together and one
   comparison explains every case;
5. conditions rewritten over things ("can reach a key whose colour matches
   the door"), which carry over to new layouts and colours and merge the
   tree's duplicates.

Perceptual contrast is a helper at step 2. It narrows where to look, and
it keeps a thing identified as the view shifts. The second job comes from
learning move effects, already next in STATUS: forward shifts the view one
tile and a turn rotates it, so whatever does not shift as predicted is
held or has changed.

### E. How the agent would discover "key"

1. **A useful action, no object yet.** The tree finds that pick up helps:
   after some pickups, toggling can open the door.
2. **The fixed place.** Pick up, drop and toggle act on the tile in front
   and never move the agent. The before/after difference shows the effect:
   the tile in front goes from key to floor, and the held place shows a
   key. A failed pickup changes nothing.
3. **The code.** From many attempts, the agent learns what each action
   does to each thing in front. Keys come out as "blocks me, can be taken",
   whatever their colour, because colour does not change what pick up does.
   The door comes out as "blocks me, toggling can open it", the vase as
   "toggling removes it", the floor as "I can step here".
4. **The right key is a relation.** Both keys can be picked up, but only
   one makes the door open. Toggle's success depends on the held key and
   the door together, and one comparison of their codes ("held matches
   front") explains every case.
   - That comparison forces colour into the code, which step 3 alone would
     drop. The code keeps whatever any condition needs.
   - In our renderer a key and a door of one colour share the same RGB
     colour, so the comparison should carry over to new colours.
5. **Recognition anywhere.** To walk toward a key, the agent applies what
   it learned in front to every tile in view. The key detector comes from
   interaction, not from segmentation.

The "thing in front" pointer exists in all three planned worlds: MiniGrid's
tile ahead, Crafter's facing tile (every Crafter interaction acts on it),
and Minecraft's block under the crosshair.

### F. Vision contrast: now and later

- **Now: before against after an interaction.** A pixel difference; the
  agent does not move during pick up, drop or toggle. It locates the event
  and gives P19's "something other than me changed".
- **Not yet: separating a thing from its background.** The tile grid
  already gives units, and every tile stands out, so it could not even be
  tested here. It is needed for things larger than a tile or with varied
  looks (Crafter's trees and water, anything in Minecraft), and for
  following a thing across the view while walking toward it.

### G. The relational bottleneck (`the-relational-bottleneck-as-an-inductive-bias-for-efficient`)

- **The idea:** downstream processing sees only inner products between
  learned object codes, never the codes themselves.
- **The evidence:** on "first and third are the same" rules, with 95 of 100
  objects withheld from training, only the bottleneck model (ESBN)
  generalised; Transformers, LSTMs and relation networks did not. It
  learned from as few as 4 examples.
- **Where it fits:** the key-and-door colour rule is exactly a "relational
  task" in their sense. Only whether the colours match matters, not which
  colour.
- **Where it does not:**
  - It assumes objects arrive already separated.
  - It throws away what each thing is, and most of our conditions depend
    on that (a key can be picked up; a vase cannot). The authors say
    human reasoning is not purely relational either.
- **The takeaway:** one tool, used where a condition depends on two
  things. Compare their codes, and pass on only the comparison.

### H. Hierarchies: where our tree sits

- **Teleo-reactive programs** (Nilsson 1994, `cs_9401101`). Our tree is
  almost exactly Nilsson's tree: each node is "the weakest condition from
  which this action eventually achieves the node above". He suggested
  growing the tree where no node holds (card 026), but his conditions were
  hand-written.
- **From skills to symbols** (Konidaris et al. 2018). The symbols needed
  and sufficient for planning with a set of skills are the skills'
  starting conditions and effects. Our conditions are exactly such
  starting conditions, which backs the causal pillar as the source of
  conditions.
- **Duplicates.** Every formalism identifies a condition by what it picks
  out, not by where it sits. Hierarchical goal networks index methods by
  the goal they achieve, about halving domain size against a task-network
  planner. Predicate invention prunes candidates that hold in exactly the
  same places. Card 026's tree stores conditions by their path, which is
  why duplicates appear.
- **Objects in hierarchies.** Every hierarchy is parameterised over
  objects ("go to X"). None of the learned ones discovers its objects from
  pixels; that gap is ours. The "target of a way" is already the slot where
  an object enters, like Nilsson's X.
- **What differs from task and goal networks:** the user noted that we
  discover the hierarchy from the agent's own experience (and later from
  demonstrations, P16), where those systems are given their methods and
  predicates.

### I. Decisions taken with the user

- Card 026: keep.
- Objects as pointers filled by conditions, not a standing list.
- Causes first, perceptual contrast as a helper, relations falling out of
  conditions through one comparison.
- "Learn kinds where you act, recognise them everywhere", with no
  figure–ground contrast yet.
- INTerPRet (`interpret`) stays in papi only, not LITERATURE.md: this
  model does not use language feedback.

### J. Open questions and likely next cards

- **The relation** (next, if this card behaves as predicted): toggle's
  condition as one comparison between the held thing's and the front
  thing's codes. A new-colour test in this world needs new colours: purple
  and yellow are already the vase and the switch-on colour, and MiniGrid
  has six colours in all.
- **Learned codes:** a DeepSym-style encoder on front-tile pixels replacing
  exact appearance matching. It must give unseen appearances distinct
  codes.
- **Duplicates:** merge conditions that pick out the same situations, so
  the tree becomes a shared graph.
- **Places, not things:** "in the last room" does not fit the front-tile
  story and may belong to the spatial side of conditions (moving closer).
- **Hypotheses for GOAL.md:** drafted in the session (conditions are
  causal; acting is a search over conditions; objects are what conditions
  are about; relations where a condition depends on two things; a
  condition is identified by what it picks out). Waiting for the user.
