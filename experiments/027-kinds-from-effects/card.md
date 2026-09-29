---
id: "027"
title: kinds from what actions do in front
rung: 0
serves: [P7, P1, P12, P4]
status: done
verdict: pass
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

One component: what a way walks to and when it acts. Revised with the
user after the first run failed the gate (section 7): the two rules marked
"revision" were added, judging moves and actions by the agent's own measure
of a positive effect, the one its tree was built from (an action succeeds
when its parent condition turns true).

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
revision: walking toward a target, never take a move after which the
          way's own condition would stop holding
revision: after acting, if the way's parent condition has not turned true,
          that tile, from that spot, is marked as not working for this way
          for the rest of the episode; the agent tries the next candidate
```

What is learned and what is still exact:

| Part | In card 027 | Why |
|---|---|---|
| Input | Egocentric pixel frames (card 016's view) and the agent's own actions | C1 |
| Effect of an action on the thing in front | Before/after pixel difference at the front and held places | Pick up, drop and toggle never move the agent |
| **Kinds, recognition anywhere** | **Learned: a tile encoder with a 4-bit code** | The new component |
| Which way to pursue | Card 026: tree grown on demand, depth-first search, exact conditions | Learned conditions failed card 021's gate; a separate problem |
| Whether the way depends on a tile | The exact condition on the state with that tile swapped for floor | Stand-in; a learned condition would be asked the same of an edited frame |
| Whether acting worked (revision) | The exact parent condition after the action | The agent's own success measure; nothing is learned from it in this card (card 028) |
| Whether a move keeps the way open (revision) | The exact way condition after the move | Walking is exact in this card |
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
  20,000 updates of 1,024 rows (declared). Second revision, with the user,
  after the second run (section 7): rows are sampled so that every outcome
  is equally likely, an outcome being the action and what it did (forward:
  moved, blocked or reached the goal; pick up, drop, toggle: which of the
  front and held places changed). A rare effect then weighs as much as a
  common one.
- **Third revision, with the user, after the third run (section 7):
  discovery and recognition are split.** Counting (arm 2's grouping, learned
  from the agent's own pixels with no labels) decides what the kinds are,
  where the agent acts. The network only learns to recognise them: the same
  body (8×8×3 pixels → 128 → 128), now ending in one output per counted
  kind, trained to give each appearance the kind counting assigned it. The
  training set is the distinct appearances that were in front, each
  weighted equally, since each is one fact ("this tile is of kind k"); 2,000
  full-batch updates (declared). The network's kinds replace the 4-bit
  codes everywhere in acting. This is the order of Chalupka et al.
  (`1412_2309`, Algorithm 1): classes from experiments first, then a network
  trained on them. Reported, not a criterion: **recognising something never
  seen** (leave one out). For each appearance whose counted kind has
  another member (the three keys, three closed doors, three open doors, two
  switch states), the network is trained without it and asked its kind;
  three seeds. The full test, with correction by acting, is the transfer
  card.
- **A way's kind:** the codes in front at its successes in experience (for
  drop, also the code held). The dependence test runs only when a way has
  more than one candidate: the two keys, or floor tiles for drop.
- **Unchanged from card 026:** the tree, experience, 500 new layouts,
  200-step budget, closeness. Move ways ("turn beside the key") act on no
  thing and keep exact targets (declared; their steps are reported).
- **Two worlds, same layouts:** the key world (the door opens for the key
  of its colour; a key of another colour lies in the room) and the switch
  world (the door opens when the switch is on; keys do nothing).

Arms, in each world. Arms 2–5 use the two revision rules; the marks are
episode memory only, and no weights change while acting. With no spot
left, or every closer move refused, the move is random (counted).

1. Upper bound: the simulator's targets (card 026).
2. Exact kinds: the main arm, but with kinds from exact grouping of tile
   appearances, as an upper bound for the encoder. The grouping is
   stochastic bisimulation (Givan, Dean & Greig 2003): every action has the
   same effect and leads into the same kinds, with groups split while card
   010's evidence supports it.
3. **Main: learned kinds plus the dependence test.** From the third
   revision on, the learned kinds are the network's recognition of the
   counted kinds.
4. Kind alone: candidates of the way's kind, without the dependence test.
5. Contrast alone: candidates are any tile that stands out (neither floor
   nor wall), with the dependence test.
6. Exact kinds without the revision rules: the first run's rule, reported
   in the gate as a check that the rerun reproduces it.

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

Result of the gate, first run (before the revision):

Failed in both worlds on one check: acting with exact kinds. The main run
(learned kinds) was not started. Numbers in [results.json](results.json).

| Check | Key world | Switch world |
|---|---|---|
| Effects from pixels agree with the simulator | 100% | 100% |
| Upper bound (simulator's targets) | 100%, 16.4 steps | 100%, 17.1 steps (tree of 20, grown in 5 rounds) |
| Exact kinds equal the expected grouping | yes (8 kinds) | yes (8 kinds) |
| Every way's kind is the expected one | yes, each way on one kind | yes by majority; two toggle ways act on both switch and vase (98.8% and 76% switch), both correct |
| **Acting with exact kinds (at least 98%)** | **96.6%** (17 of 500 fail), 16.1 steps | **95.2%** (24 of 500 fail), 17.0 steps |

Result of the gate, second run (with the revision rules): passed in both
worlds. The first run's rule, rerun alongside, reproduced its numbers
exactly (96.6% and 95.2%).

| Acting with exact kinds | Key world | Switch world |
|---|---|---|
| Reaches the goal | **100%** | **100%** |
| Mean steps when successful (upper bound) | 16.6 (16.4) | 17.3 (17.1) |
| Random moves, share of all moves | 0.4% | 0.6% |
| Actions that did not work (layouts with any) | 45 (28) | 60 (56) |
| Moves refused | 6 | 10 |

## 6. Success criteria and prediction

Same 500 new layouts in both worlds, main arm:

1. **Learned kinds are pure.** No code is shared by appearances of two
   different expected kinds. The kinds in front are floor, wall, goal,
   closed door, open door, key, switch and vase; colours and switch states
   within a kind may share a code or not. Every appearance gets one code on
   at least 99% of its tiles.
2. **Ways name things.** For each non-move way that acting uses, at least
   99% of its successes have front codes that belong only to expected kinds
   for its action. A way may have more than one kind (revised: "toggle
   something so the door can be reached" is rightly done to a switch or a
   vase). Expected kinds:
   - step onto the goal square: goal;
   - toggle: closed door, vase or switch;
   - pick up: key;
   - drop: floor in front, key held.
3. **Acting.** Reaches the goal in at least 98% of layouts in both worlds,
   with random moves at most 1% of all moves, and mean steps when
   successful at most 1.5 × the upper bound's.

Reported with them: whether learned codes merge colours (the question for
transfer, next); conditions evaluated per move by the dependence test;
arms 2, 4, 5 and 6; steps on move ways; actions that did not work and
moves refused (revision), per layout.

Prediction: criteria 1–3 pass. Kind alone fails in the key world (about
45–70%, roughly the layouts where the matching key is reached first) by a
loop: pick up the other key, drop it, pick it up again. The kind says what
to look for; the condition says which one matters. Contrast alone matches
the main arm except where a way's thing is "any floor tile": you drop onto
floor, which contrast calls background. So it falls short in the key world
(card 026 used drop in a few dozen layouts) and matches in the switch
world. Colours may not merge in the learned code: nothing in training
needs colour, and nothing asks for it to be dropped.

Prediction for the revision (added before the second run): exact kinds
now pass the gate in both worlds, with a few actions per failing layout of
the first run that did not work (the switch turned on from the wrong side,
say) and a small rise in steps. Kind alone now mostly succeeds, by trial
and error: more actions that did not work and more steps than the main
arm; the step criterion is where it should fall short. Contrast alone
still fails where drop is needed: it never offers floor, so there is
nothing to try.

Prediction for the second revision (added before the third run): the goal
gets its own code in both worlds (its forward outcome, reaching the goal,
now carries one outcome's share of the rows instead of 0.04%), and
criteria 1–3 pass. Random moves fall to near the exact-kinds arm (under
1%). Risk: balancing gives rare outcomes a large share, so some common
distinction could now be merged instead; criterion 1 checks every code.

Prediction for the third revision (added before the fourth run): the
network recognises every appearance as its counted kind, so criteria 1–3
pass and the main arm acts exactly as arm 2 (same kinds, same choices).
That part is close to a check that it can learn 15 labelled tiles. Leave
one out: keys, closed doors and open doors are placed by shape, most of
them correctly; the risk is colour, since the network sees raw pixels (a
withheld green closed door may be called the goal, the nearest tile in
pixels). The withheld switch state is the least certain: its only other
member is a ball of another colour.

What this card can show: that learned kinds name the right things, and
that things found through conditions can replace the simulator's targets
inside the search. What it cannot yet show: that kinds help acting beyond
that. With exact conditions, the dependence test does most of the work;
that needs learned conditions or new colours (later cards).

Added before the run, from a smoke test (1,000 episodes, 40 layouts) and
the implementation:

- **Drop needs a place.** In 2 of 40 key-world layouts the agent picked up
  the other key to clear its way to the matching key, then dropped it back
  where it blocked, and looped. Drop's target is a place, not a thing, and
  swapping floor for floor cannot find it. The gate's exact-kinds acting
  (key world, at least 98%) may fail on this.
- **Contrast alone fails more widely.** The dependence test also marks
  things further up the chain (the door, the goal square) as targets,
  because removing them ends the way's route too. So contrast alone should
  fail wherever such a thing is nearer than the way's own thing.
- **A grouping bug was fixed:** groups whose members all had one outcome
  never merged.

Risks: open doors are rare (93–176 tries per action) and pass like floor,
so the encoder may lump them with floor (criterion 1). The switch world
has not been run with card 026's pipeline (the gate checks). Budget: the
switch world's tree grown on demand, the encoder in a few minutes, five
acting arms per world. About 10–15 minutes, under 30.

## 7. Result

### First run (before the revision)

The learned part of the card was not tested. What was tested worked:
effects read from pixels matched the simulator on every transition, and
exact grouping by effects found the eight expected kinds in both worlds
(colours merged; the two switch states merged), with every way's kind the
expected one. The failure is in the target rule, and it would fail the
same way with learned kinds.

Every failing layout was replayed step by step. All 41 are loops, with
three patterns and one cause:

- **Dropping back where it blocked** (key world). The other key blocks the
  matching key; the agent picks it up, then drops it on the tile it came
  from, because every floor tile passes the test and it is already facing
  one (layout 12). This is the pre-run note's failure.
- **Turning the switch off again** (switch world, 21 of 24). With the switch
  on, the agent still cannot walk to the door (the vase is in the way), so
  the tree's next way, "toggle something so the door can be reached", still
  applies. Breaking the vase would do it, but the switch also passes the
  test (removing it ends the route), and the agent is facing it, so it
  toggles it off; then on; and so on (layout 56).
- **Turning back and forth** (switch world 3 of 24, and key-world layouts
  such as 363). The test names the right tile but not the side to act from,
  or names both keys. Walking toward the nearest candidate spot turns one
  way; the exact condition, which knows the one spot that works, turns the
  other; the agent alternates (layouts 71, 363).

The cause: the rule asks whether the way's route *depends on* a thing
(remove it and see), when acting needs to know whether *acting on it*, from
that spot, would achieve the parent. The two agree for a key that must be
picked up, and differ for a place to drop onto, for a switch that is
already on, and for which side of a thing to stand on. The two-kind toggle
ways also show that criterion 2 (one kind per way) is too strict: "toggle
something so the door can be reached" is rightly done to a switch or a
vase. 16 of the 17 key-world failures began by picking up the other key; when the matching key
came first, acting always succeeded (483 layouts).

The first run is a fail at the feasibility gate. Revised with the user
(section 2); nothing is learned from the outcomes in this card, which
leaves learning from the agent's own attempts to card 028.

### Second run

The gate passed (section 5), so the learned stage ran: 196 seconds in all,
the encoder 33 seconds per world. Main arm against its criteria:

| Criterion | Key world | Switch world |
|---|---|---|
| 1. Learned kinds pure | **fail**: goal shares the green closed door's code | **fail**: the same pair |
| 2. Ways name things (99%) | **fail**: step onto the goal 0%, toggle the door 69% | **fail**: 0% and 65% |
| 3. Reaches the goal (98%) | 99.4% | 99.4% |
| 3. Random moves (at most 1%) | **17.0%** | **13.9%** |
| 3. Steps (at most 1.5 × upper bound) | 20.2 (1.23 ×) | 19.8 (1.16 ×) |

Apart from that one pair, the learned code is the kinds: every other code
belongs to one expected kind, and every other way passes criterion 2 at
100%. Keys of all three colours share one code in both worlds, and the
two switch states share one. Closed doors keep one code per colour in the
key world; in the switch world red and blue share one.

**One cause for all three failures.** The goal square is the rarest thing
in front in the training rows (154 of 372,302 in the key world, 477 of
366,854 in the switch world; stepping onto it ends the episode), and its
nearest neighbour in pixels is the green closed door (solid green against
dark green; mean pixel difference 39, against 92 between door colours).
Rows are sampled uniformly, so keeping the goal apart would lower the loss
by very little, and the encoder gave it the door's code. That puts the goal
square, present in every layout, into the kind of the way "toggle the
door". The dependence test keeps it, because removing the goal does end the
way's route. It lies behind the closed door, so no move brings the agent
closer and the move is random. Replaying the arm layout by layout: of 1,806
random moves in the key world, 1,525 are this way heading for the goal and
248 more are its moves refused (switch world: 1,264 and 138 of 1,453). The
random moves are spread over all door colours, since the goal is always
there.

Comparison arms, all with the revision rules:

| Arm | Key: reaches goal, steps, random | Switch: reaches goal, steps, random | Actions that did not work (key, switch) |
|---|---|---|---|
| Upper bound (simulator's targets) | 100%, 16.4, 0% | 100%, 17.1, 0% | – |
| Exact kinds | 100%, 16.6, 0.4% | 100%, 17.3, 0.6% | 45, 60 |
| Main (learned kinds) | 99.4%, 20.2, 17.0% | 99.4%, 19.8, 13.9% | 45, 61 |
| Kind alone (no dependence test) | 98.6%, 23.1, 18.5% | 98.6%, 20.4, 16.6% | 390, 292 |
| Contrast alone | 96.4%, 24.2, 34.1% | 90.8%, 33.5, 36.2% | 202, 1,020 |

Against the predictions: exact kinds passed the gate as predicted, with a
small rise in steps. Kind alone mostly succeeded by trial and error, with
five to nine times the main arm's failed actions, as predicted; but it did
not fall short on steps (1.41 × and 1.19 ×), as predicted it would.
Contrast alone did worse in the switch world than the key world, the
opposite of the prediction (drop was not the problem). The first
prediction, criteria 1–3 pass, was wrong on one pair of appearances.

The dependence test also has a weakness this run exposed: anything further
up the chain passes it (removing the goal ends every route), and the
revision rules only catch a wrong target after acting on it, which never
happens if the target cannot be reached.

The second run is a fail: criteria 1 and 2 fail on one pair, and
criterion 3 on random moves only. Revised with the user: the encoder's
rows are balanced by outcome (section 2).

### Third run

Rows balanced by outcome; everything else as the second run (198 seconds).
The gate passed again with the same numbers.

| | Key world | Switch world |
|---|---|---|
| Codes shared by different kinds | goal with green closed door (as before) | blue and red closed doors with vase; green closed door with keys |
| 2. Ways failing (share on pure codes) | step onto the goal 0%, toggle the door 69% | toggle the door 65%, pick up so the switch can be reached 0% |
| 3. Reaches the goal | 99.4% | 98.8% |
| 3. Random moves | 17.0% | 13.9% |
| 3. Steps (× upper bound) | 20.2 (1.23 ×) | 20.2 (1.18 ×) |

In the key world the learned grouping came out identical to the second
run's (different code numbers), so acting was identical too. In the switch
world the goal got its own code, as predicted, but two new merges
appeared, as the risk note warned. Of 1,548 random moves there, 1,068 are
the way "toggle something so the door can be reached" finding no move
closer to its targets, which now include a closed door (it shares the
vase's code), and 287 more are that way's moves refused.

**Why balancing did not help.** When two appearances share a code, the
network pays only for the rows of the one it predicts worse, so what
matters is the smaller one's share of the rows. Uniform rows made the goal
that minority (154 of its code's rows against the door's 1,938). Balanced
by outcome, the goal's single outcome, reaching the goal, carries a ninth
of all rows, and the green door's forward rows are now 449 of 33,680
"blocked" rows, so the door is the minority and the merge costs as little
as before. Whatever the weighting, some look-alike is a small share of its
code and can be merged cheaply. The grouping by counts has no such
weakness: about a hundred tries with a different effect are decisive
evidence however rare they are among all rows.

**The "exact kinds" arm is itself learned from pixels.** It uses no
labels: appearances are distinct pixel tiles, effects are read from
before/after pixels and from the episode ending, and groups are split by
card 010's evidence. The card treated it as an upper bound for the
network, but in these worlds, where one thing always looks pixel-identical,
it is a learner, and it passed every check: the eight expected kinds in
both worlds and acting at 100%. What the network adds is placing an
appearance never seen before, which this card does not test (transfer).

The third run is a fail: criteria 1 and 2 fail, and criterion 3 on random
moves only. Revised with the user: counting decides the kinds, the network
learns to recognise them (section 2).

### Fourth run

Counting decides the kinds; the network learns to recognise them (168
seconds in all; the network trains in under a second per world). The gate
passed with the same numbers as the second and third runs.

| Criterion | Key world | Switch world |
|---|---|---|
| 1. Learned kinds pure | pass: the 8 expected kinds | pass: the 8 expected kinds |
| 2. Ways name things (99%) | pass: every way used, 100% | pass: every way used, 100% |
| 3. Reaches the goal (98%) | 100% | 100% |
| 3. Random moves (at most 1%) | 0.4% | 0.6% |
| 3. Steps (at most 1.5 × upper bound) | 16.6 against 16.4 (1.01 ×) | 17.3 against 17.1 (1.01 ×) |

The network gives every appearance its counted kind, so the main arm acts
exactly as the counting arm, move for move, as predicted. This part is
little more than a check that a network can learn 15 labelled tiles; the
substance is in what counting found and in the target rule (sections 5
and 7 above). Comparison arms: kind alone 98.8% and 98.6%, with 12% and
11% random moves and several times the failed actions; contrast alone
96.4% and 90.8% (unchanged: it uses no kinds).

**Recognising something never seen (leave one out, reported): 4 of 33
right.** Withheld, each of the 11 appearances was given a kind by three
networks:

| Withheld | Given (three seeds) |
|---|---|
| Keys: blue; green; red | vase ×3; switch ×3; switch ×3 |
| Closed doors: blue; green; red | open door, vase, vase; goal ×3; open door ×3 |
| Open doors: blue; green; red | floor ×3 each |
| Switch off; switch on | key, switch, key; switch ×3 |

Every key and door was misplaced. The network goes by colour and overall
brightness, not shape: an open door is mostly dark with a thin coloured
edge, like floor; the green closed door is called the goal, the nearest
tile in pixels (the one risk the prediction named). Only the switch, whose
two states are the same shape in two colours, was placed by shape, and
not reliably. Against the prediction (most placed correctly by shape):
wrong. Fourteen labelled tiles, most kinds with two members that differ
in colour, give the network no reason to separate shape from colour.
Transfer to new appearances will need that separation from somewhere;
the transfer card starts here.

The fourth run is a pass on criteria 1–3.

## 8. Decision

**Keep** (with the user, 2026-09-28). Later cards build on kinds decided
by counting what actions do in front, and on the target rule with its two
revision rules (refuse moves that end the way's condition; mark an action
that does not turn its parent true). The network returns in the transfer
card, trained on the counted kinds, where recognising appearances never
seen (4 of 33 here, by colour rather than shape) is the problem to solve.

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
