---
id: "051"
title: the planner keeps its chain of conditions, checks rules compiled from recall, and costs per step what is in play
rung: 0
serves: [P12, P21, P17, P14, P8, P5, P15]
status: done
verdict: pass
arch_version: 9
date: 2026-10-04
---

# 051: the planner keeps its chain of conditions, checks rules compiled from recall, and costs per step what is in play

Drafted 2026-10-04 at the user's request: "a detailed card about your
proposed efficiency, generalization, and understanding improvements to
the planner ... I want to know clearly the before and after of each
change". The user allowed it to exceed the template's length. It is a
design card: seven changes, each with its before and after, what it buys,
its risk and its test. Section 9 proposes the order and how to split them
into runs, because CHARTER asks for one component per experiment.

It builds on cards 049 and 050 (recall through admitted conditions, own
tries first), kept as architecture version 9. Renumbered from 052 to 051
on 2026-10-04, so that it comes before the staged encoder (card 052),
which needs its near-linear cost.

## 1. Question

Version 8's planner works backward from the goal over conditions, as
GOAL.md's P12 asks, and it is right in the familiar worlds. Three
problems stop it from scaling:
- **Cost.** Each step rebuilds the whole chain and asks recall, which
  scans all of memory, hundreds of times. A lifetime of T steps costs
  about T² (card 049; ARCHITECTURE.md, "Known limits"). The user asked
  for near-linear per-step cost before BabyAI or Crafter (2026-10-03).
- **Commitment.** The chain is forgotten between steps, so the agent can
  undo at one step what it did at the previous one: card 049's two doors
  (key A picked up and dropped in turn, 0 of 150) and card 050's
  cluttered layout 66 (the purple key the same way).
- **Understanding.** The planner's reasons exist only during one call.
  Nothing records "door A is open because the agent holds key A, which
  it picked up for that", so a failure cannot be attributed to the link
  that broke. Card 052's encoder needs exactly such events (conditions
  met or broken).

Can the planner keep the backward hierarchy of conditions and gain all
three? Before rung 1. Serves:
- P12, the hierarchy made explicit and kept: conditions, the actions that
  achieve them, and what each serves;
- P21, System 2 deliberates over chains of conditions without simulating
  steps, and simple decisions become lookups (System 1);
- P17, per-step cost independent of memory size;
- P14, a broken link is repaired where it broke;
- P8, rules composed into new chains in new layouts;
- P5 and P15, "unknown" kept apart from "fails", and tried when worth it.

## 2. Are we leaving the subgoal hierarchy and its depth-first search?

No. The user asked this directly (2026-10-04), so here is the answer
first.

**What stays.** Working backward from the goal: a condition's achievers
are the actions whose predicted effect makes it true, an achiever's needs
are its conditions, and needs become subgoals down to actions (card 029,
means-ends analysis, as STRIPS). The search still descends depth-first
into one achiever's needs, with the same depth limit (6). Walking stays
card 045's: a learned how-soon network (System 1) and routes whose tokens
are conditions.

**What changes in the search.** Three things around the depth-first
descent:
1. *Which achiever is expanded first.* Today the planner tries achievers
   in a fixed order (fewest unmet needs, then nearest) and expands each
   one fully until one works. After the change, one cheap backward pass
   gives every condition an estimated cost first. The descent then takes
   the cheapest achiever, and the others remain fallbacks (change 4).
   Depth-first, guided by estimates, instead of depth-first in a fixed
   order.
2. *What a step starts from.* Today the descent restarts from "episode
   ended" at every step. After the change, the previous step's chain is
   kept, and a step re-examines only the parts the last observation
   touched (change 5).
3. *What the descent asks at each node.* Today each node asks recall.
   After the change, it checks a rule compiled from recall, and asks
   recall only where the rule does not apply (changes 2 and 3).

The hierarchy is the same, and so is the order of reasoning (goal, then
conditions, then subgoals, then actions). The search is still
depth-first. What changes is how the planner chooses what to expand,
what it remembers between steps, and what it asks at each node.

## 3. How the planner works today

The code: `tools/card029/subgoals.py` (`solve`, `pursue`, `choose`),
with walking from `tools/card045/movement.py` (`Walk.walk`), conditions
from `tools/card043/consistent.py` and situations from
`tools/card047/situations.py`.

At every step, from the top:
1. `choose` calls `solve(("end",))`, retrying up to 12 times when a
   refusal changes what is allowed.
2. `solve(c)` lists c's achievers (actions whose recall-predicted effect
   makes c true) with their needs, sorts them by the number of unmet
   needs, and expands each group in turn with `pursue`, keeping the
   nearest that works.
3. `pursue` takes the achiever's first unmet need and recurses into it
   (`solve`, or `walk` for facing). When every need is met, it imagines
   the action and refuses it if the imagined step undoes a met need
   higher in this chain, or makes a facing in this chain unreachable.
4. A need on the hand or the view (card 043) is met in a situation where
   recall predicts the action works with that situation's own hand and
   view. The situations come from stored tries on the same tile and on
   similar tiles (card 047), each judged by recall.

Each node asks recall at least once, and recall compares the query with
every stored try on every admitted condition (card 049). Card 049's
shakedown worked out 65 conditions per move in the key world and 120 in
the either world.

**Layout 66, read against this** (card 050's trial, seed 403; traced
2026-10-03). The agent stands at (2,2), facing a locked red door. It
holds a purple key, which it has just learned does not open the door.
- Step 9's chain: door open ← a different hand ← (an achiever through
  the blue key) ← pick up the blue key ← an empty hand ← **drop**. The
  imagined drop puts the purple key on the cell in front, (2,3). That is
  the only open cell next to the agent, so it walls the agent in. The
  red key is not in this chain, so the refusal check does not protect
  the route to it.
- Step 10, hand empty: the descent restarts from the top and now
  chooses the red key. Its route is blocked by the purple key, so
  "make (2,3) walkable" becomes a need, achieved by **picking up the
  purple key**.
- Step 11, holding the purple key: the chain of step 9 again, so
  **drop**. Then the pair repeats for the rest of the episode.

Each step's chain is reasonable on its own. Two causes combine: the
choice of achiever changes between steps (blue key, then red key), and
protection covers only the chain of the current step. Why step 9's
chain went through the blue key is not traced; the trace shows a need
for a drop facing the vase, which is not yet understood.

## 4. The seven changes

Each change below has the same five parts: before, after, what it buys
(efficiency, generalization, understanding), its risk, and its test.

### Change 1: memory indexed by situation (recall)

**Before.**
- A recall query compares the query with every stored try on every
  admitted condition: about 0.1 ms per thousand tries, linear in memory
  (card 049's timing).
- The query's own situation (card 050) is found by the same scan.
- Admission of conditions builds an n × n distance matrix per candidate,
  which is 1–5 s at 771 tries and about 20 GB per candidate at 50,000.

**After.**
- **Own situations by hash.** The key is (front code tuple, held code
  tuple, the values of the admitted view conditions), and each entry
  holds the outcome counts of its tries. A lookup takes constant time.
- **Neighbours by approximate nearest-neighbour search,** over the
  admitted-condition features, weighted by recall's λ. The nearest k
  (for example 32) replace the full scan, at about log n per query
  (Neural Episodic Control does this; `1703_01988`). They are asked
  only when the own situation has no tries, or for the prior α.
- **Admission on groups.** Tries with equal codes on every candidate
  are one group with summed outcomes, so the fit is over S distinct
  situations, not n tries. In the key world that is 50 groups against
  771 tries per action. It is refitted only on surprise: a failed
  prediction, or a new situation.

**Buys.**
- *Efficiency:* query cost independent of memory size; admission
  quadratic in distinct situations, which grow far more slowly than
  tries.
- *Generalization:* unchanged. The neighbours are the same, only found
  faster.
- *Understanding:* each situation has an entry that can be read: "this
  door with this key in hand, 412 tries, opened 412".

**Risk.** The approximate search can miss a neighbour that a full scan
would weight. The groups merge tries that differ only in tokens no
candidate reads. That is not CHARTER's forbidden merging into hard
kinds: the tries themselves stay stored, and groups are rebuilt from
them whenever the candidates change.

**Test.** Predictions identical to card 050's on all held-out tries (at
least 99.9% the same class). Query time flat (at most 2 times) from 1,000
to 50,000 stored tries (synthetic growth, as card 049's timing).

### Change 2: rules compiled from recall (System 1 for conditions)

**Before.** Every node of the descent asks recall what an action does to
a tile in a situation. The same questions repeat across steps and
layouts.

**After.** For each action and outcome, one or more **rules**, compiled
from recall:
- *Conditions:* the admitted conditions (card 049), each with its
  region. The front tile's region is the code tuples whose own situations
  give the outcome. A relation is a distance below a radius. A view
  condition is "a token with this code tuple is in view".
- *Effect:* the outcome class (what the front and the hand become).
- *Support:* the own situations and the tries behind the rule.

A rule is a cache. It is rebuilt from the kept tries, never edited, and
never counted as evidence of its own:
- recall decides whenever no rule covers the situation;
- recall decides whenever a rule's prediction fails (the own situation
  then contradicts it, card 050), and the rule is rebuilt for that
  situation;
- recall decides when a token has no code, a "new" piece.

This is CHARTER's "weights take over from recall" applied to conditions.
Rules answer only where they agree with recall on every stored case, rare
ones included, and novelty is judged outside the rule.

**Buys.**
- *Efficiency:* a node checks a rule against the current tokens in
  constant time (change 3) instead of querying memory. Recall is asked
  once per new situation, not once per node per step.
- *Generalization:* a rule's regions say how far a condition extends. A
  purple key inside the region of "keys that fit" meets the condition
  until a try says otherwise; outside it, recall decides. This is card
  049's transfer, made explicit.
- *Understanding:* the rules can be printed: "toggle at a closed door
  works when the held tile's part 1 is near the door's part 1
  (relation)". Card 049 already admitted conditions of this shape.

**Risk.**
- *Sliding back to card 028's counted tables.* That is what rule 8
  forbids. The guard: rules hold no counts of their own and are
  regenerated from tries, and recall is the authority wherever they
  disagree.
- *Rare outcomes absorbed by a dominant one.* Rules are compiled per
  outcome class, and a class with any own situation keeps its own rule.

**Test.** On held-out tries in the four familiar worlds, the rule-backed
prediction equals recall's in at least 99.9%. The share of nodes answered
by a rule, and of nodes that fall back to recall, is reported.

### Change 3: conditions checked by lookup in the current tokens

**Before.**
- "Is X held?" reads the hand token.
- "Is X in view?" and "which tokens show X?" scan the 638 token
  places.
- Situations for a hand or view need are gathered from memory (card
  047's templates) and judged one by one by recall.

**After.** The current tokens are indexed by code tuple (tuple to the
tokens showing it), updated by the step's changes only. Checking a rule's
condition becomes:
- a hash lookup for a tile condition;
- a scan of the few candidate pairs for a relation;
- an index entry for "in view".

A need on the hand ("a hand that makes toggle work here") becomes "a
held tile inside the rule's region", answered by listing the tokens in
view whose code tuple lies in it. These are the candidates to pick up.

**Buys.**
- *Efficiency:* a node's check is constant time.
- *Generalization:* candidates for a need come from the region, so a
  tile never seen in that role can be one.
- *Understanding:* "which keys in view would work" is a list the planner
  can report.

**Risk.** Small. The index must follow every change to the tokens, so
it is rebuilt from the tokens if a check fails.

**Test.** For every node in card 050's runs, the same answers as the
scan.

### Change 4: one backward pass of cost estimates guides the descent

**Before.** `solve` sorts achievers by their number of unmet needs, then
expands them group by group, fully, until one works. It keeps the
nearest by walking cost (card 029, then card 045's how-soon network).
The cost of an unmet need is not estimated before it is expanded, so
the search can descend into an expensive or impossible branch first. A
step's work grows with how many branches it tries.

**After.** Before the descent, one pass over the conditions in play
gives each an estimated cost:
- cost(c) = 0 if c holds;
- otherwise, the minimum over c's achievers of (the action, plus
  walking to face its tile from the how-soon network, plus the summed
  cost of its needs).

The pass ignores what actions undo. It is the additive cost estimate of
HSP (Bonet and Geffner 2001) and close to FF's relaxed plan (Hoffmann
and Nebel 2001), and it is linear in the number of rules and conditions
in play. The descent then expands achievers in order of estimated cost
and keeps the others as fallbacks. The choice between the key and the
switch, now made by comparing walking chains (card 045), becomes a
comparison of two estimates.

**Buys.**
- *Efficiency:* the descent usually expands one branch, not all of
  them.
- *Generalization:* new combinations (two doors, three keys) get
  estimates from the same rules, so longer chains cost the pass more
  conditions, not more search (P8).
- *Understanding:* every condition has a number ("open door B: about 31
  steps away"). That is P12's "how soon", measured against outcomes
  (rung 1's test) rather than only used.

**Risk.** Estimates that ignore undoing are optimistic. That is safe for
ordering, because the descent still checks every need, but it can still
prefer a branch that conflicts with another. Change 5 handles the
conflict.

**Test.**
- Same decisions as card 050 in the familiar worlds (at least 99% of
  actions identical).
- Fewer nodes expanded per step (reported against card 050).
- The estimates' rank correlation with the evaluator's true steps over
  test states (reported; rung 1's measure).

### Change 5: the chain kept between steps, as links, and repaired

**Before.** Teleo-reactive (card 029): rebuilt from the top at every
step. Protection covers only the needs met higher in the chain of the
current step and the facings in it. A choice can flip between steps:
layout 66's blue key at step 9 and red key at step 10, or two doors' key
A.

**After.** The chain is a set of **links**. A link is (producer,
condition, consumer): "pick up key A gives *holding key A* for toggling
door A". Each step:
1. **Check the links the observation touched.** For each link whose
   condition changed: if it now holds as planned, advance; if it was
   broken, mark it.
2. **Repair the broken part only.** Re-derive the subtree under a broken
   link with changes 2–4, and keep the rest.
3. **Protect every open link.** An action whose predicted effect breaks
   an open link is a **threat**, and the planner then has three choices:
   - order it after the link's consumer (open door A first, then drop
     key A);
   - choose another way to do it (drop the purple key on a cell no route
     link crosses);
   - choose another achiever.

   The kinds of link include holding an object, a door open, and a
   route's tokens walkable (card 045's walk conditions).
4. **Rebuild from the top** only when the goal's own link is broken, or
   when the cost estimate of the kept chain rises above an alternative's
   by more than a margin. That margin is the one new declared parameter.

This is partial-order planning's causal links and threats (SNLP,
McAllester and Rosenblitt 1991; UCPOP, Penberthy and Weld 1992),
repaired incrementally as in LPA* and D* Lite (Koenig and Likhachev
2002).

**Buys.**
- *Efficiency:* a step's work is proportional to what changed, not to
  the chain's size.
- *Generalization:* two doors and layout 66 are the same failure (a
  later need undoes an earlier one), and links fix it in any layout
  without a rule about keys (the user's warning against engineered
  patches, 2026-09-29).
- *Understanding:* the chain is an object that persists and can be
  printed: "goal ← door B open ← holding key B ← drop key A after door A
  is open". When an act fails, the broken link names the expected
  condition that did not hold: the event card 052's encoder term needs.

**Risk.**
- *Commitment against new information:* a kept chain can ignore a
  better option that appears (a door opened by someone else, later). The
  margin in step 4 bounds this.
- *Stale links:* a link whose condition changed outside view. P2's
  belief is not modelled yet, so such links are checked when seen.

**Test.**
- Card 049's two doors: at least 80% of the 30 layouts in 4 of 5 seeds
  (card 049: 0 of 150; card 048 with version 8: 2 of 150).
- Card 050's cluttered layouts with walls of objects: at least 99%.
- Familiar worlds unchanged.

### Change 6: needs met in states the learned effects produce, not spliced

**Before.** A need on the hand or the view is checked by writing the
needed hand or view into the present state (card 043's declared
exception for conjunctions, extended by card 048 to hands needed inside
another action's conditions). In card 048 this made key A vanish from
the imagined room.

**After.** A need is met through the actions that produce it, imagined
by their predicted effects:
- "an empty hand" is the drop, with the dropped object on the drop
  cell;
- "holding key B" is the pick up, with key B gone from the floor;
- each produced state is then checked by the rules.

Where the drop goes is a choice. The drop cells are ranked by change 4's
cost of the links they would threaten, using change 5's threats.

**Buys.**
- *Efficiency:* neutral (one imagined action per need, as now).
- *Generalization:* the imagined state is one that can occur, so recall
  and the rules are asked about situations like the ones they were
  learned from.
- *Understanding:* the planner's imagined chain is a sequence of real
  effects, which can be checked against what happens.

**Risk.** A need with no producing action becomes unmeetable, where a
splice would have pretended it was met. That is correct, but it may
expose missing effects.

**Test.** No spliced states left (counted). The two tests of change 5.

### Change 7: "unknown" kept apart from "fails", and tried when worth it

**Before.** A prediction under one half reads as "does not work". A new
colour door with no matching tries reads as closed for good, and the
agent either acts at random (card 050's seed 401, random in 97% of steps)
or never tries.

**After.** Recall reports its support: the summed weight of the own and
neighbouring tries behind a prediction. With low support, the outcome is
**unknown**, not false. In change 4's pass, an unknown achiever costs its
walking cost plus a price for a try that may fail. If it is still the
cheapest option, the planner tries it. The try's outcome is then the own
situation's first try (card 050), which settles it.

**Buys.**
- *Efficiency:* fewer random steps.
- *Generalization:* new objects get tried rather than ignored, the
  behaviour card 047 and CHARTER ask of transfer ("judged by tries").
- *Understanding:* the planner can say "I don't know whether the purple
  key fits; trying costs 6 steps".

**Risk.** The user deferred chance on 2026-10-01 ("later"). This change
does not add chance: it separates "never tried here" from "tried and
failed". It is still a new decision rule with a price to declare, and it
should come last.

**Test.** Seed 401's new-colour switch door, 100 layouts (card 049: 14%,
mostly random walks); the share of random steps there.

## 5. Before and after, in one table

| | Before (version 8 with cards 049 and 050) | After |
|---|---|---|
| Search | Depth-first over achievers in a fixed order, from the top every step | Depth-first over achievers in order of estimated cost, from the kept chain |
| What a node asks | Recall over all of memory | A rule checked against indexed tokens; recall when the rule does not apply |
| Memory lookup | Linear scan per query | Own situation by hash; neighbours by approximate search |
| Admission | n × n per candidate, once per world | Over distinct situations, again on surprise |
| Between steps | Nothing kept | Links kept, threats refused, broken links repaired |
| Hand and view needs | Spliced into the present state | Produced by imagined actions; the drop's cell chosen |
| Low support | Read as "fails" | Read as "unknown", tried at a price |
| Per-step cost | Grows with memory and with the chain | Grows with what changed and the rules in play |
| What can be printed | The current step's trace | The chain with its links, the rules, each condition's estimated cost |

## 6. What this does not change

- The goal (the episode's end) and the hierarchy of conditions.
- Recall as the source of every prediction (rules are its caches).
- The encoder, tokens and walking (card 045).
- Chance: an outcome is still predicted at one half or more.
- Belief out of view (P2, rung 5).

## 7. Dependencies

- Cards 049 and 050 (recall through admitted conditions, own tries
  first): passed, kept as version 9.
- Card 029's means-ends planner, card 043's conditions in produced
  situations, and card 045's walking: all passed.
- Literature, to add to LITERATURE.md when this card becomes the focus:
  - Bonet and Geffner 2001 (HSP); Hoffmann and Nebel 2001 (FF);
  - McAllester and Rosenblitt 1991 (SNLP); Penberthy and Weld 1992
    (UCPOP);
  - Koenig and Likhachev 2002 (D* Lite);
  - Nilsson 1994 (teleo-reactive programs, the planner's present form);
  - `1703_01988` (Neural Episodic Control, approximate search);
  - `1110_2211` (Pasula et al., rules over learned outcomes).

## 8. How this differs from classical planning

Asked by the user on 2026-10-03. The search parts (goal regression,
relaxed cost estimates, causal links) are classical and borrowed on
purpose (CHARTER rule 7). The difference is the model they run on:
- conditions are regions of a learned latent space, admitted from
  contrasts in the agent's own tries, not predicates written by a
  designer (C1);
- rules are caches over recall, rebuilt from kept tries and overruled
  per situation by a single failed try. They are not operators fitted
  once;
- matching is graded, so a new object can meet a condition by distance,
  and low support reads as unknown;
- walking is System 1, and only conditions are deliberated (P21).

The planners that also learn their rules (Pasula et al. 2007, Silver et
al. 2021, Chitnis et al. 2022, Konidaris et al. 2018) start from given
objects and predicates, or features to invent them from. The tests here
should therefore aim where a hand-written domain could not do as well:
new objects, partly known situations, conditions nobody named.

## 9. Order and runs

CHARTER asks for one component per experiment. Two of the seven changes
are in recall, and five in the planner:
- **recall:** change 1 (index), and change 2's compilation;
- **planner:** changes 3–7.

Proposed order, each a run with its own gate, numbered as cards when the
user approves each:
1. **Changes 1 and 3 (efficiency, no change in behaviour).** Test:
   identical decisions to card 050, and per-step time flat from 1,000 to
   50,000 stored tries. This is the near-linear requirement on its own,
   with nothing else changed, so any difference in behaviour is a bug.
2. **Change 2 (rules).** Test: rule predictions equal recall's (at least
   99.9%), and decisions unchanged.
3. **Change 4 (cost pass).** Test: decisions unchanged in the familiar
   worlds; fewer nodes per step; the estimates' rank correlation with
   true steps reported.
4. **Changes 5 and 6 together (links, produced states).** They share the
   threat check, which needs the drop's real effect. Test: two doors at
   least 80% in 4 of 5 seeds; cluttered at least 99%; familiar worlds
   unchanged.
5. **Change 7 (unknown).** Last, with the user's agreement, since it
   touches chance. Test: seed 401's new-colour door.

**Budget.** Steps 1–3 should change no decisions, so each can be checked
on spare seed 399 in under 10 minutes. Step 4 needs card 049's full test
set, about 40 minutes, handed to the user.

## 10. Prediction

- Step 1 removes most of the slowdown between card 047 (0.42–0.72 s per
  layout) and card 049 (5.8–11.2 s). Per-step time stays within 2 times
  from 1,000 to 50,000 stored tries.
- Step 4 lifts two doors from 0 to above 80%. Some two-door layouts may
  still fail where key A must be put down on a cell that blocks no link,
  and none exists without a detour the cost pass underrates.
- Step 5 lifts seed 401's new-colour door well above 14%, because the
  agent tries the new door once with the switch on instead of walking at
  random.

## 11. Overnight work (2026-10-04)

The user delegated card 051 and card 052 overnight (2026-10-04: "proceed
with reasonable keep / throwaway / revisions"). Each step's criteria are
written here before it runs; decisions taken overnight are marked as
Claude's and can be reversed. The running record is [log.md](log.md).

### Step 1: memory indexed by situation (changes 1 and 3), declared before the run

**What is built** (`tools/card051/index.py`, on card 050's recall):
- **Groups of stored keys.** Recall's prediction for a pick up, toggle
  or drop depends on a stored key only through its front tile, its held
  tile and the admitted view conditions. Keys equal on those form one
  group with summed outcome counts. A query is compared with each group
  once, not with each key: the same numbers, at a cost that grows with
  the distinct situations, not with memory. A key's own situation is a
  hash lookup.
- **Admission on groups.** The same leave-one-out likelihood, computed
  over groups of keys equal on every candidate condition.
- **Situations for the planner by class.** Card 047's templates list
  every distinct (held tile, view) of the stored tries on a tile, which
  grows with memory. Situations equal on the held tile and the admitted
  view conditions get the same prediction, so one per class is kept.
  The same for the situations that made a tile walkable.

**Gate.** Profile of version 9 on 10 cluttered layouts, seed 399, alone
on the machine: 1.08 s per layout. Recall's per-query scan takes 7.5 s
of the 10.8 s of acting; setup 61 s.

**Criteria** (seed 399, the spare seed):
1. **Same decisions.** Cluttered world, 30 layouts, and the key and
   switch worlds, 30 layouts each: at least 99% of actions identical to
   card 050's, the same successes.
2. **Flat cost.** Memory grown by real tries from random play in
   cluttered layouts to about 5,000, 20,000 and 50,000 stored pick up,
   toggle and drop keys: time per step within 2 times the base memory's
   (about 800 keys per action). Card 050's recall measured alongside, to
   20,000 keys.

Keep if both pass; otherwise revise once, then stop and record.

### Step 4a: threats between the needs of one achiever (part of change 5), declared before the run

Taken before steps 2 and 3, because step 1 already removed most of the
cost those two were for, and the two-door failure is the largest gap.

**The failure, traced** (seed 399, two doors, layout 0;
`runs/051/trace_two_doors_399_0.txt`). "Pick up key red" has two
unmet needs: an empty hand, and facing key red. Facing key red needs
door blue open, which needs key blue in hand. Card 029's pursue() takes
the first unmet need, the hand, so the agent drops key blue; the next
step it picks key blue up again to reach key red, and so on for 200
steps. Protection covered only conditions above in the chain, not what
the other need's plan relies on.

**What is built** (`tools/card051/threats.py`). When an achiever has two
or more unmet needs, each is planned from the present. A plan records the
conditions it relies on that hold now: the needs met along its chain,
and, for an achiever with no needs because it works as things are,
"this action works with the present hand and view". A need threatens
another when the act its plan works toward (the drop, the toggle),
imagined on the present, breaks a condition the other's plan relies on.
The first need that threatens none is pursued; if all do, card 029's
order. Recorded: how often needs were weighed and reordered.

**Shakedown** (seed 399, two doors, layouts 0–4): 5 of 5 reach the goal
(36–49 steps), against 0 of 30 under card 050.

**Criteria** (seed 399 first, then seeds 400–404):
1. Two doors (30 layouts): at least 80% in 4 of 5 seeds (card 049 and
   card 050's recall: 0 of 150).
2. Nothing lost: chained rooms with one door and the cluttered world (100
   layouts each) no more than one layout worse than card 050 in any
   seed; the four familiar worlds (30 layouts) pass card 049's
   criterion 1.

Keep if both pass.

### Step 4b: the chain's choices kept between steps (the rest of change 5), declared before the run

**The failure, traced** (seed 403, cluttered layout 66;
`runs/051/trace_clutter_403_66.txt`). Recall on seed 403 lets the purple
key read as fitting the red door (parts mix kind and colour); one failed
try corrects that (card 050). Then step 9 chooses the blue key (drop the
purple key first), step 10 the red key, whose route the dropped purple
key now blocks (pick it up), and the two alternate for the rest of the
episode. Step 4a does not help: the two keys are alternative achievers,
not needs of one achiever.

**What is built** (`tools/card051/commit.py`). Each step records, for
every condition in the chain it acts on, the achiever it chose; the next
step tries that achiever first for the same condition and searches as
before only when it no longer gives a plan. No cost margin yet (change
5's step 4).

**Shakedown** (seed 403 layout 66: goal at step 32; layout 47 still
fails, as under every recall; seed 399 two doors layouts 0–2: the same
steps as step 4a).

**Criteria** (seeds 400–404, with step 4a's tests):
1. Cluttered world: at least 99% in every seed (card 050: 98% in seed
   403, 100% elsewhere).
2. Two doors at least 80% in 4 of 5 seeds; one door and the familiar
   worlds no worse than step 4a (one layout of slack per test).

### Step 4c: routes that need two tokens made walkable (walking, card 045's component), declared before the run

**The failure, traced** (seed 400, two doors, layout 11; it fails in
every seed). The cell before door green is reachable only through the
cell where key green lies (the switch and the vase wall off the rest), so
reaching key red needs key green picked up and door green opened. Card
045's walking counts one token at a time walkable when no route exists;
no single token gives a route, so no chain was found and the agent acted
at random for 200 steps.

**What is built** (`tools/card051/walk2.py`): when no single token gives
a route, pairs of tokens that recall says can be made walkable are tried
(each approach of a chain may cross one of them, chains through waypoints
as card 045); the token the cheapest such route steps onto first becomes
the condition ("walk", j). It acts only where card 045's walking returns
nothing, so decisions elsewhere should not change.

**Shakedown** (seed 400): layout 11 reaches the goal at step 45; layout
0 at step 46, as before.

**Criteria** (seeds 400–404, on step 4b's agent): two doors at least
29/30 in every seed and layout 11 solved in at least 4 of 5; one door,
cluttered and familiar worlds no worse than step 4b (one layout of
slack per test).

### Step 4c result: revise (both world lost 4 layouts in seeds 403 and 404); fixed by step 4d

| Seed | Two doors | One door | Cluttered | Familiar worlds |
|---|---|---|---|---|
| 400–402 | **30/30**, 1.06 | **100%**, 1.00 | 100%, 1.00–1.08 | pass |
| 403 | 30/30 | 100% | 99% (layout 93) | **both world 26/30: fail** |
| 404 | 30/30 | 100% | 100% | **both world 26/30: fail** |

Large gains beyond two doors: one door's layouts 78 and 88, which failed
under every recall and the upper bound, now succeed, and cluttered routes
fell from 1.29–1.46 to 1.00–1.16 times the shortest route. The both
world's regression, traced (`runs/051/trace_both_403.txt`, layout 13):
holding key blue, the toggle at door blue still needs the switch on in
view; card 043 asks only for the view because the hand works as it is
now, so holding key blue is no condition and nothing protects it. With
routes through two tokens, a cheap achiever for the view need appears
that starts by dropping key blue, and the agent picks it up and drops it
in turn.

### Step 4d: the hand kept when only the view is asked for, declared before the run

**What is built** (`tools/card051/threats.py`, `HAND_LINK`): when an
achiever's need is on the view because the hand works as it is now (card
043's single-part need), the present hand is added to the conditions
protected in that branch and to the plan's links. An action that changes
the hand there is refused, as card 029 refuses one that breaks a met need.

**Shakedown:** both world seed 403, 30/30 (step 4c: 26/30); two doors
seed 400 layouts 0 and 11, as step 4c.

**Criteria:** step 4c's (seeds 400–404, everything no worse than step 4b,
one layout of slack; two doors at least 29/30 in every seed), with every
familiar world passing card 049's criterion 1.

### Step 4d result (seeds 400–404): pass; steps 4c and 4d kept together (Claude, overnight)

| Seed | Two doors (version 9: 0/30) | One door (version 9: 98%) | Cluttered (version 9: 98–100%) | Familiar worlds |
|---|---|---|---|---|
| 400–402, 404 | 30/30, 1.06 | 100%, 1.00 | 100%, 1.00–1.08 | 100% in all four, card 029's steps |
| 403 | 30/30, 1.06 | 100%, 1.00 | 99% (layout 93), 1.16 | 100% in all four, card 029's steps |

Ratios are steps against the evaluator's shortest route, which may not
move objects other than the door's key; the agent sometimes beats it by
picking up a key in its way (one door 0.997). Time per layout 0.10–0.16 s
(one door, cluttered), 0.43 s (two doors), 0.20–0.64 s (familiar worlds,
five runs sharing the machine). The only failure left in 1,950 test
episodes is seed 403's cluttered layout 93 (see step 4b).

## 12. Result

### Step 1 (seed 399): pass, kept (Claude, overnight)

| | Card 050 | Step 1 |
|---|---|---|
| Decisions, 30 cluttered + 30 chained layouts | | identical, action for action |
| Familiar worlds (moves, step kinds) | | identical |
| Time per layout, key world | 2.7 s | 0.24 s |
| Time per step at +0 / +5,000 / +20,000 stored keys | 43 / 579 / 4,640 ms | 9.9 / 6.3 / 5.7 ms |
| Recall groups per action at the same sizes | (keys: 725 / 2,450 / 7,500) | 50 / 82 / 95 |
| Admission at base memory | 725–771 keys | 50–52 groups, same conditions, gains and α |

- One revision: the first version still grew (328 ms per step at
  +20,000) because card 038's ways were refitted over every pair of a
  class's keys after every stored try. They are now fitted once, like
  recall's weights; decisions stayed identical.
- **Left:** re-admission on grown memory took 50 s at +20,000 keys
  (1.8 s at +5,000); it runs only on surprise, but it is not yet linear.
  +50,000 keys was not reached: new distinct keys get rare as memory
  fills (+5,000 keys took 47,000 random tries, the next +5,000 took
  169,000), and the test's list of tries filled 22 GB; the agent's own
  arrays at +10,000 keys are about 25 MB per action (log.md). Imagining a result for a new
  situation (card 038's transport) still reads every key of the outcome
  class, linear but cheap at these sizes.

### Step 4a (seeds 399–404): pass, kept (Claude, overnight)

| Seed | Two doors (card 050: 0/30) | One door | Cluttered | Familiar worlds |
|---|---|---|---|---|
| 399 | 29/30, 1.05 | 98%, as card 050 | 99%, as card 050 | pass, same moves |
| 400–404 | 29/30, 1.05 in each | 98%, as card 050 | 98–100%, as card 050 | pass in 5/5, card 050's steps |

Layout 11 of two doors fails in every seed: the planner finds no chain
from the start and acts at random (not traced further). Cluttered layout
66 of seed 403 still fails (step 4b).

### Change 2, readable half only (no change in acting)

`tools/card051/rules.py` prints every own situation recall holds, per
action, with its tries and outcome, in the evaluator's names:
[rules_seed400.md](rules_seed400.md). For example, toggle reads the front
tile's parts and one relation (part 2, front against held), and the
table shows a closed door opening only with the key of its colour. The
compiled rules that would answer the planner's questions (change 2
proper) are not built.

### Step 4b (seeds 400–404): pass, kept (Claude, overnight)

| Seed | Two doors | One door | Cluttered (step 4a) | Familiar worlds |
|---|---|---|---|---|
| 400–402, 404 | 29/30, 1.05 | 98%, as card 050 | 100% (100%) | pass, card 050's steps |
| 403 | 29/30, 1.05 | 98%, as card 050 | **99%** (98%) | pass, card 050's steps |

- Seed 403: layouts 47 and 66 now reach the goal; layout 93, which step
  4a solved, now fails (`runs/051/trace_clutter_403_93.txt`). On this
  seed recall lets the purple key read as fitting the red door; after
  that try fails, the planner looks for "a different hand" through
  dropping the purple key in front of the vase, and alternates between
  that achiever's two needs, which take different forms on alternate
  steps, so neither step 4a's threat check nor step 4b's commitment
  (keyed by condition) catches it. One layout in 500 cluttered episodes.
- Time per layout rose slightly with step 4a's sibling planning (key
  world about 0.25 s, against step 1's 0.24 s; two doors 0.51 s).


## 13. Decision

**Keep** steps 1 and 4a–4d (the user, 2026-10-04, confirming Claude's
overnight keeps). Keeping them makes this card architecture version 10:
memory indexed by situation, threats between the needs of one achiever,
commitment between steps, routes through two tokens, and the hand kept
when only the view is asked for. On seeds 400–404: two doors 30/30
(version 9: 0/30), one door 100% (98%), cluttered 99–100% (98–100%),
the four familiar worlds 100% with card 029's steps; time per step
5.7–9.9 ms from base memory to +20,000 stored keys (version 9: 43–4,640
ms).

Not done, left for a later card when BabyAI's scale calls for them:
change 2's compiled rules (only the readable table was built), change
4's cost pass, and change 7 ("unknown" kept apart from "fails"; it
touches chance and waits for the user). Change 6 was not needed: the
spliced check came up 0 times in 1,041 planning steps. Open: cluttered
seed 403 layout 93 (an achiever whose needs alternate in form);
re-admission on grown memory is not yet linear (50 s at +20,000 keys,
run only on surprise); the +50,000-key point was not reached.
