---
id: "096"
title: structure memory, extracted from stored events offline (a sleep phase)
rung: 1
serves: [P3, P10, P12, P19, P17]
status: draft
verdict:
arch_version: 20
date: 2026-10-09
---

# 096: structure memory by consolidation (an outline)

Drafted 2026-10-09 with the user as an outline, not yet an experiment: a
literature review comes first, and the card will be split into tested
steps after it. ARCHITECTURE.md's "Memory systems" names three memories;
this card is the slow one.

## 1. Question

Structure memory today is pieces added one at a time: the encoder, card
070's key–door relation, card 087's property network, recall's admitted
conditions. None is learned from the agent's stored events after it
acts, and none is used as recall's prior except forward's. If the agent
has an offline phase between episodes (a sleep) that replays stored
events and:
1. **extracts structure:** per kind of token, what each action does to it
   (pick up works on boxes with an empty hand), what it does on its own
   over time (stays; moves toward the agent), and how kinds relate (this
   key opens doors like this one);
2. **keeps goal trees** from solved episodes: the chain of conditions
   that reached the goal, with the situation it started from (card 092's
   closing note);
3. **refines the latent space:** the encoder moved so that tokens that
   behave alike lie close, with stored events re-encoded afterwards (card
   052's staging: memory written only from a settled encoder); its
   transition head is card 095.6's whole-state model, so refinement and
   prediction share one network;

does recall, with structure as its prior, predict better on held-out
tries, transfer to new kinds, and plan from fewer tries in a new world?

## 2. What changes (sketch)

**First question: does structure need a component at all?** In
exemplar models (Hintzman 1986, MINERVA 2) knowledge of categories
emerges from recall over stored instances: every trace answers in
proportion to its similarity, and what traces share comes out as the
generalisation, with no separate store (the user, 2026-10-09). So the
first step tests whether recall over stored tries, with the right
similarity and back-off, already gives what structure is asked for
(pick up works on boxes with an empty hand; a key opens doors like it).
A separate component is added only where recall alone measurably fails.

**Episodic memory as sequences of conditions** (the user, 2026-10-09).
Per episode, the sequence of condition changes that mattered (changes
in the environment that affected the agent or that it caused), each
with its situation and the steps between, not frames. "Mattered" is
GOAL P19's test (surprising, lasting, changing later predictions, or
changing how likely or how soon a goal is). Goal trees, and P12's "how
likely and how soon" (C6), are read from these sequences: from where
they started and how long they took. This replaces "goal trees from
solved episodes" below with something stored for every episode, solved
or not.

**Remember what was learned to matter** (the user, 2026-10-09, after
[094.1](../094.1-relative-position-recall/card.md)). Long-term memory
keeps, in time order, the conditions the agent believed important, not
the whole view. A token's place is kept only once a condition on place
is learned. Such a condition is flexible: a relation between tokens
("next to the agent", "between the agent and the goal", "adjacent to
the door"), learned as card 070's key–door relation was, not a
coordinate. In 094.1, raw offsets made almost every stored try its own
key, so nothing grouped. Three points from the discussion:
- **A condition never stored cannot be learned.** If memory drops a
  token's place until place is known to matter, the agent can never find
  that it matters. So raw detail is kept for a while, and consolidation
  tests candidate conditions on it (admission's rule: kept while it
  predicts held-out tries better) before the detail is dropped. A
  surprise (an outcome the admitted conditions did not predict) marks
  detail to keep, since that is where a missing condition lies. This is
  the fast-store, slow-extract split of complementary learning systems.
- **Two kinds of place condition.** "Near the agent" is a condition on
  one action. In MiniGrid it is built in, since every interaction acts
  on the token in front, so MiniGrid cannot test it. Crafter can
  ("craft only near a table"). "Path unobstructed" is a condition for
  getting somewhere, which the planner works out when walking (closeness
  by propagation), so it is not stored as an action's condition.
- **The encoder later, not first.** Admission already refines what
  matters as data arrives. Retraining the encoder (card 052's staging)
  comes in only where a needed condition is not separable in the present
  latent space; card 052 grew too large by starting there.

```
awake:   act; every try is stored in event memory (as now)
asleep:  replay stored events → fit structure (properties, own dynamics, relations, goal trees)
         → optionally update the encoder → re-encode stored events → refit recall
recall:  same tokens → similar tokens → structure (the prior, replacing "all tries")
```

**One structure for two questions.** "Can a box be picked up?" and "where
is the enemy now that it is out of view?" are the same kind of
knowledge, a token kind's transition, but different uses:
- both predict a kind's next state from the present one, given an
  action; the enemy's case is the "no action of mine" transition (time
  passing), applied repeatedly while it is unseen;
- the first is read by recall when planning; the second by working
  memory (card 094's object files), which carries an unseen token's
  expected where forward each step, its uncertainty growing since last
  seen.
In MiniGrid the own-dynamics part learns "stays where it was" for every
kind, which card 094 currently assumes.

## 3. Literature review (before any step is drafted)

Candidates, each to be read for what to take and what not:
- exemplar memory: Hintzman 1986 (MINERVA 2), Nosofsky's GCM (already
  behind recall); event segmentation (Zacks et al. 2007) for what makes
  a change an event;
- the closest designs: Soar's working, procedural, semantic and
  episodic memories (Laird 2012); ACT-R; CoALA (Sumers et al. 2023);
- complementary learning systems and replay (McClelland, McNaughton and
  O'Reilly 1995; Kumaran, Hassabis and McClelland 2016); hippocampal
  replay in sleep (Wilson and McNaughton 1994); schemas speeding
  consolidation (Tse et al. 2007);
- `dreamcoder-2` (Ellis et al. 2021): wake–sleep library learning, a
  sleep that abstracts reusable pieces from solved problems (goal trees);
- wake–sleep (Hinton et al. 1995); generative replay (Shin et al. 2017);
  Dyna (Sutton 1991); world models learned from replay (Ha and
  Schmidhuber 2018; Dreamer);
- schema networks and object-oriented models (DOORMAX), already in
  LITERATURE.md, for kinds and their effects;
- forgetting by use: which stored events structure makes redundant
  (STATUS's note on smart forgetting; the first form, pruning instances
  that change no vote, is [095.2](../095.2-recall-at-scale/card.md)).

## 4–6. Data check, gate, criteria

To be written per step after the review. The likely first step: whether
recall alone, with the four-level back-off (same tokens, similar views;
same tokens, any view; similar tokens; all tries), gives what structure
is asked for, judged on tier 2's five failures and held-out tries.
