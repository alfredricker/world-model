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

Can the agent learn from its own pixels the kinds of things in its world,
from what its actions do to the thing in front of it? And can it then find
each way's target as "the thing of the way's kind that the current
condition depends on", well enough for the depth-first search to reach the
goal without the simulator saying which tile is which? Kinds are learned
from pixels; conditions and walking are still exact (section 2 says why).
Serves P7 (the units worth tracking are found, not supplied), P1 (a thing
means what actions do to it), P12 (conditions are about things) and P4
(only distinctions that change outcomes are kept). It follows
[card 026](../026-depth-first-subgoals/card.md) and the theory sessions
recorded in the appendix. Card 026 reached the goal in every layout, but
the simulator told each way where its target was.

## 2. What changes

One component: what a way walks to and when it acts.

```
card 026: a way's target = the poses where its action achieves the parent,
          computed by the simulator, which knows which tile is which
card 027: candidates = tiles in view whose learned code is one the way's
                       action succeeded on in experience (the way's kind)
          target     = the candidates the way's route depends on: swap the
                       tile for floor; if the way's action can then no longer
                       achieve its parent by moving closer, it is a target.
                       If no candidate is needed (any floor tile will do for
                       drop), every candidate is a target.
          at a target, take the way's action
```

What is learned and what is still exact:

| Part | In card 027 | Why |
|---|---|---|
| Input | Egocentric pixel frames (card 016's view) and the agent's own actions | C1 |
| Effect of an action on the thing in front | Before/after pixel difference at the front and held places | Pick up, drop and toggle never move the agent |
| **Kinds, recognition anywhere** | **Learned: a tile encoder with a 4-bit code** | The new component |
| Which way to pursue | Card 026: tree grown on demand, depth-first search, exact conditions | Learned conditions failed card 021's gate; a separate problem |
| Whether the way depends on a tile | The exact condition on the state with that tile swapped for floor | Stand-in; a learned condition would be asked the same of an edited frame |
| Walking (moving closer, free tiles) | Exact (card 024's step measure) | Move effects are a later card |

- **Kind encoder (learned).** A small network maps one tile's pixels
  (8×8×3) to 4 on/off bits (straight-through, as DeepSym). Trained only on
  the agent's own transitions: from the front tile's code and the action,
  predict the effect (forward: moved, blocked or goal reached; pick up,
  drop, toggle: did the front place change, did the held place change) and
  the codes of what the front and held tiles became (same encoder, no
  gradient through that path: the learned form of "turns into things of
  the same kind"). One encoder reads the front place, the held place and
  every tile in view (declared: a thing looks the same wherever it is);
  the agent's own place is never a candidate. Rows: card 012's stored
  transitions (every change, an eighth of the rest), sampled uniformly;
  20,000 updates of 1,024 rows (declared).
- **A way's kind:** the codes in front at its successes in experience (for
  drop, also the code held). The dependence test runs only when a way has
  more than one candidate: the two keys, or floor tiles for drop.
- **Unchanged from card 026:** the tree, experience, 500 new layouts,
  200-step budget, closeness. Move ways ("turn beside the key") act on no
  thing and keep exact targets (declared; their steps are reported).
- **Two worlds, same layouts:** the key world (the door opens for the key
  of its colour; a key of another colour lies in the room) and the switch
  world (the door opens when the switch is on; keys do nothing).

Arms, in each world:

1. Upper bound: the simulator's targets (card 026).
2. Exact kinds: the main arm, but with kinds from exact grouping of tile
   appearances, as an upper bound for the encoder. The grouping is
   stochastic bisimulation (Givan, Dean & Greig 2003): every action has the
   same effect and leads into the same kinds, with groups split while card
   010's evidence supports it.
3. **Main: learned kinds plus the dependence test.**
4. Kind alone: candidates of the way's kind, without the dependence test.
5. Contrast alone: candidates are any tile that stands out (neither floor
   nor wall), with the dependence test.

## 3. Dependencies

Card 026's pipeline (`tools/card026/dfs.py`, `tools/card023/closer.py`),
its key-world tree (`runs/026_dfs.json`) and result; card 012's exact
discovery and stored data; card 010's evidence; card 016's egocentric view
(`tools/card016/bench_ego.py`). Literature (papi): `deepsym` (a binary
code trained to predict action effects gives object kinds),
`equivalence-notions-and-model-minimization-in-markov-decisio` (the
coarsest grouping with equal effects), `1412_2309` (the smallest change
that flips an outcome locates its cause: the dependence test).

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
- **Upper bound:** the simulator's targets in each world. Key world: 100%,
  16.4 steps (card 026). The switch world has not been run; it must reach
  at least 98%.
- **Exact kinds:** the grouping must equal the expected one, and acting
  with it must reach at least 98% in both worlds. Otherwise the idea fails
  before learning is tested.
- **Trivial baseline:** contrast alone (arm 5).

Result of the gate, before the main run:

## 6. Success criteria and prediction

Same 500 new layouts in both worlds, main arm:

1. **Learned kinds are pure.** No code is shared by appearances of two
   different expected kinds. The kinds in front are floor, wall, goal,
   closed door, open door, key, switch and vase; colours and switch states
   within a kind may share a code or not. Every appearance gets one code on
   at least 99% of its tiles.
2. **Ways name things.** For each non-move way that acting uses, at least
   99% of its successes have front codes from one expected kind, and it is
   the expected one:
   - step onto the goal square: goal;
   - toggle: closed door, vase or switch;
   - pick up: key;
   - drop: floor in front, key held.
3. **Acting.** Reaches the goal in at least 98% of layouts in both worlds,
   with random moves at most 1% of all moves, and mean steps when
   successful at most 1.5 × the upper bound's.

Reported with them: whether learned codes merge colours (the question for
transfer, next); conditions evaluated per move by the dependence test;
arms 2, 4 and 5; steps on move ways.

Prediction: criteria 1–3 pass. Kind alone fails in the key world (about
45–70%, roughly the layouts where the matching key is reached first) by a
loop: pick up the other key, drop it, pick it up again. The kind says what
to look for; the condition says which one matters. Contrast alone matches
the main arm except where a way's thing is "any floor tile": you drop onto
floor, which contrast calls background. So it falls short in the key world
(card 026 used drop in a few dozen layouts) and matches in the switch
world. Colours may not merge in the learned code: nothing in training
needs colour, and nothing asks for it to be dropped.

What this card can show: that learned kinds name the right things, and
that things found through conditions can replace the simulator's targets
inside the search. What it cannot yet show: that kinds help acting beyond
that. With exact conditions, the dependence test does most of the work;
that needs learned conditions or new colours (later cards).

Risks: open doors are rare (93–176 tries per action) and pass like floor,
so the encoder may lump them with floor (criterion 1). The switch world
has not been run with card 026's pipeline (the gate checks). Budget: the
switch world's tree grown on demand, the encoder in a few minutes, five
acting arms per world. About 10–15 minutes, under 30.

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
   - Corrected in review (section K): for the colours already seen, the
     conditions carry the relation without any comparison. The comparison
     is for new colours and for learning from fewer examples.

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
4. **The right key, through the condition.** Both keys can be picked up,
   but only one makes the door open. The condition "I can pick up the key
   that opens the door" depends on the matching key: remove it and the
   condition fails; remove the other key and nothing changes. So the
   condition picks the right key among the things of kind "key" (this
   card). For new colours, one comparison of the two codes ("held matches
   front") would state the rule once for every colour (a later card):
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
- A way's target is the thing of its kind that its condition depends on,
  not any thing of its kind (section K).
- The new part is learned from pixels; conditions and walking stay exact
  until their own cards pass (section K).
- Transfer and generalisation are the next question once this card
  succeeds.
- INTerPRet (`interpret`) stays in papi only, not LITERATURE.md: this
  model does not use language feedback.

### J. Open questions and likely next cards

- **Transfer: the relation and new colours** (next, if this card
  succeeds). Toggle's condition as one comparison between the held thing's
  and the front thing's codes, and codes that give unseen appearances
  distinct, sensible values. A new-colour test in this world needs new
  colours: purple and yellow are already the vase and the switch-on
  colour, and MiniGrid has six colours in all.
- **Duplicates:** merge conditions that pick out the same situations, so
  the tree becomes a shared graph.
- **Places, not things:** "in the last room" does not fit the front-tile
  story and may belong to the spatial side of conditions (moving closer).
- **Hypotheses for GOAL.md:** drafted in the session (conditions are
  causal; acting is a search over conditions; objects are what conditions
  are about; relations where a condition depends on two things; a
  condition is identified by what it picks out). Waiting for the user.

### K. Review of the first draft (2026-09-28)

- **The colour relation.** The user asked why the first draft expected
  the conditions to fail at colour relations. They do not.
  - Exact conditions carry the relation by construction.
  - Learned ones did too: card 005's wrong-key cases were at least 99.7%
    right, and card 012's learned conditions reached the goal in 95% of
    key-world layouts with exact walking.
  - What cannot carry it is the kind, which groups keys because pick up
    does the same to each. The first draft's target rule ("any thing of the
    way's kind") skipped step 2 of section D. The revision finds the target
    through the condition, and keeps "kind alone" as a comparison arm.
- **End to end.** The user asked whether this card should be learned end
  to end. The finished system must be (C1, C2), and this card learns its
  new part, the kinds, from pixels. Conditions and walking stay exact
  because their learned versions have not passed their own tests:
  - architecture 3's conditions failed card 021's gate, with held-out
    false positives up to 38%;
  - learned walking reached 48.4% (card 016).

  Stacking them now would make a failure impossible to attribute (CHARTER
  rules 3 and 7; LESSONS: interpret a capability only after its
  prerequisite is learned). The depth-first search runs as in card 026.
- **The way to end to end,** one exact part replaced per card:
  1. kinds (this card);
  2. the relation and new colours;
  3. move effects, so that walking comes from learned effects;
  4. conditions learned from pixels, stated over things. That may also
     address card 021's failure to generalise, which happened with
     conditions over whole views; this is a hypothesis;
  5. everything learned, with one shared encoder (C2).
