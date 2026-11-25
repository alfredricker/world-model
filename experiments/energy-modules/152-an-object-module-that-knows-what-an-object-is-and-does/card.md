---
id: "152"
title: an object module that knows what an object is, where it is and what it does
rung: 2
serves: [P7]
status: done
verdict: pass
arch_version: og4
date: 2026-11-20
---

# 152: an object module that knows what an object is and does

Roadmap step 2, rung 2, on the direction under trial. The user,
2026-11-20, after cards
[150](../150-three-feed-forward-layers-of-energy-modules/card.md) and
[151](../151-layers-of-attractor-modules-in-a-world-of-parts/card.md): "I
also don't want to sidestep learning abstractions of objects. An object's
position is important information but we should be able to understand an
object by its properties to generalize to unique positions and settings",
and "we should still apply things that provide us utility from modern
hopfield, tolman eichenbaum, akorn, and cram". In card 150 a "slot" was
only a label and in card 151 a stored object was tied to grid positions.
Runs take about a minute per world. Biological counterparts: the split
between what an object is and where it is (ventral and dorsal streams,
and the Tolman-Eichenbaum account of the hippocampal formation); knowing
from sight how an object can be handled (Gibson's affordances; premotor
neurons that respond to graspable objects); general knowledge of a kind
beside the memory of one individual (complementary learning systems).

## 1. Question

Can one object module hold where an object is, what it is in its own
frame, and what it does, all learned from the agent's own episodes, so
that it recognises a known object at a new place among new neighbours, and
predicts and uses the behaviour of an object it has never met from that
object's looks? P7.

## 2. What changes

Nothing in og4. The sprite world gets three behaviours
(`tools/card152/world.py`); the module is `tools/card152/modules.py`.

```
world     takeable: taken when touched at the side with the grip closed, let go when it opens
          fixed:    never taken, never moves
          sticky:   taken when touched with the grip open, shaken off when it closes
          An object has a shape (5), a colour family (5), a width and its own texture.  In a world "by shape" the
          behaviour goes with the shape, in a world "by colour" with the colour; which goes with which is drawn per
          world.  40 individual objects come back in later episodes; 8 of them do not behave as their looks say.
module    one per object; all share their memories.  Given: which pixels are the object's.
  where   the place of the object's own frame in the image
  what    three memories of stored patterns, each storing an input that nothing stored fits (card 151's rule):
            shape   where its points lie in its frame, scaled to the frame's extent
            colour  the mean colour at its points
            itself  the full pattern in its frame, one per individual, stored at first clear sight (a modern
                    Hopfield memory).  What and where settle together: the energy of (stored pattern, shift) is how
                    badly the pattern, placed at that shift, explains what is seen; a pattern pixel hidden by another
                    object costs nothing, one seen as background costs as much as a wrong one
  does    a belief over kinds of behaviour.  A kind is a set of pairwise couplings between the tokens of a step
          (card 148's energy; a kind's set is a shared set plus its own sparse difference).  An object's kind is the
          set that predicts its records.  A kind is opened for an object that no kind predicts, and kept if others join
  looks to does   pairwise couplings between shape code, colour code and kind, by pseudolikelihood with L1 (CRAM's form)
  memory of the individual   its count of records each kind mispredicted, kept with its stored pattern
use       a new scene, first frame: settle what and where; belief over kinds = looks, plus the individual's own
          records if it is recognised; effects and the action are settled under each kind and weighted by the belief
```

A kind selects a set of couplings because one pairwise energy with a
"kind" token cannot say that the grip which takes one kind releases
another: that is a three-way dependence.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Which pixels are the object's is given | no | no | no | no | declared: true masks. Card 147's binding finds only what has moved, so a fixed object needs binding by a known pattern, which is feedback |
| An object's frame is a place and an extent: no rotation, change of scale, shape or light | no | yes | partly | no | declared |
| An individual differs from every other in its pixels | partly | no | no | partly | declared: look-alikes need where they were (LESSONS, perception) |
| Contact and "held" are read from the simulator | - | - | - | - | declared, as in cards 148–149 |
| Behaviour is deterministic and goes with one attribute of the looks | partly | partly | partly | partly | declared: a conjunction of attributes needs more than pairwise couplings |
| Three thresholds for "new" are set by hand | - | - | - | - | declared, as in card 151 |

## 3. Dependencies

Card 148 (couplings by pseudolikelihood; the action as an output token),
card 149 (propose, then verify forward), card 151 (a memory stores what
nothing fits and merges what comes to fit), card 147's sprites. LITERATURE,
"Energy modules and binding": Tolman-Eichenbaum Machine, modern Hopfield
networks, CRAM. AKOrN's binding from appearance is not used here.

## 4. Data check

Development world 0, 100 episodes (6,000 steps): an object is in contact
or held in 2,879 object-steps; each behaviour is taken 125 to 195 times
and touched without being taken with each grip. In still scenes of 4 to 6
objects dropped anywhere, 24% of views are partly hidden and 9% are a
quarter or more hidden. A view counts if at least 8 pixels and a tenth of
the object are seen.

## 5. Feasibility gate

- **Upper bound:** the true behaviour of every object given, in training
  and test, to the same per-kind couplings.
- **Trivial baseline:** one kind for all objects (card 148's form); for
  recognition, the same memory without an object frame.

Result of the gate on development world 0 (by shape), before the main
run: passed. Effects of an action on an object in contact or held, exactly
right: 92.2% with the true behaviour (the rest are moves stopped by the
edge of view, which the tokens do not show), 71.3% with one kind. The
action that takes an object, run in the simulator: 100.0% against 40.7%.
Recognition without an object frame: 0.1% of known objects.

## 6. Success criteria and prediction

Five test worlds by shape and five by colour (seeds 1 to 5), 300 training
episodes each, the same code and settings for all ten.

1. **A known object at a new place, in a new setting.** Still scenes of 4
   to 6 objects covering each other, half of them stored individuals and
   half new: the module names the right individual or says "new" for at
   least 98.0% of views, and for at least 90.0% of views a quarter or more
   hidden. Mean of ten worlds.
2. **A new object with a known property** (worlds by shape). Objects never
   met, some of a width never met, kind taken from the first frame alone:
   effects within 3.0 points of the true-behaviour bound and at least 10.0
   above one kind for all; and the take action right for at least 90.0%.
3. **Which looks matter is learned** (worlds by colour): the same bars as
   2; and in each of the ten worlds at least 90% of the coupling weight
   between looks and kind lies on the attribute that matters.

Reported without a bar: known individuals that are exceptions (with their
own records against looks alone); the kinds found and how they line up
with the true behaviours; recognition with where not settled; the belief
used as a weighted vote, as the winner, and as averaged couplings; 30 and
100 training episodes.

Development (world 0, both variants) took six versions of the kind
learner; the first five failed in one setting or another (opening a kind
for every unexplained object at once keeps a mix of two behaviours;
opening it for one object fails unless a kind starts from shared
couplings). On world 0 the final version gives: recognition 100.0 and
100.0; effects 92.2 (bound 92.2, one kind 71.3) by shape and 92.5 (93.5,
75.1) by colour; take 100.0 both; coupling weight 99.5% and 98.5% on the
right attribute. The bars were written after seeing these.

Prediction: 1 passes, though it is the easy case (every individual's
texture is unique). 2 and 3 pass in most worlds; the risk is the kind
learner, which was fragile in development, failing in a world it has not
seen.

## 7. Result

Test worlds 1 to 5 of each variant, 300 training episodes; numbers in
[results.json](results.json). "Effects" is the share of object-steps in
contact or held for which the settled displacement and held-after are
exactly right; "take" is the share of objects for which the proposed
action leaves the object held in the simulator (for a fixed object: the
answer "cannot").

| Recognition in still scenes (14,790 views; 1,132 a quarter or more hidden) | All views | A quarter or more hidden | Half or more hidden |
|---|---|---|---|
| The module: what and where settled together | **100.0** | **100.0** | 100.0 |
| Where not settled (frame left at the seen box) | 97.7 | 81.9 | 74.2 |
| No object frame (pattern left where it was stored), known objects | 0.0 | 0.0 | 0.0 |

| Objects never met, kind from the first frame | By shape: effects | take | By colour: effects | take |
|---|---|---|---|---|
| True behaviour given (bound) | 93.0 | 100.0 | 92.3 | 100.0 |
| The module: looks, kinds' answers weighted by belief | **91.2** | **92.5** | **92.3** | **100.0** |
| The same, couplings averaged by belief | 90.4 | - | 92.0 | - |
| One kind for all | 74.9 | 39.4 | 73.0 | 38.9 |
| The module, per world | 93.0, 93.4, 92.0, 94.3, **83.5** | 100, 100, 100, 100, **62.3** | 91.5, 94.5, 92.6, 92.5, 90.4 | 100 in all |

| Known individuals that are exceptions | By shape: effects | take | By colour: effects | take |
|---|---|---|---|---|
| Looks and the individual's own records | 93.5 | 100.0 | 94.1 | 100.0 |
| Looks only | 14.9 | 0.0 | 20.6 | 0.0 |

| Criterion | Verdict |
|---|---|
| 1. Recognition ≥ 98.0% of views, ≥ 90.0% of views a quarter or more hidden | pass (100.0, 100.0) |
| 2. By shape: effects within 3.0 of the bound and ≥ 10.0 above one kind; take ≥ 90.0% | pass on the mean, as the bar was written (91.2 against 93.0 and 74.9; take 92.5); **world 5 alone fails** (83.5 against 92.7; take 62.3) |
| 3. By colour: the same bars; ≥ 90% of the looks-to-kind coupling weight on the attribute that matters, in every world | pass (92.3 against 92.3 and 73.0; take 100.0; weight 97.2 to 99.4% on shape in shape worlds, 98.1 to 98.9% on colour in colour worlds) |

What the card shows.

- **What and where.** A pattern stored once in the object's own frame is
  found again at any place, among new neighbours, with half of it hidden,
  and its frame is put back exactly (the place is right for 100.0% of
  known objects). Without the frame nothing is recognised; without
  settling where, a fifth of the hidden views are lost. This is the easy
  case and I do not want to oversell it: every individual here differs in
  every pixel, and nothing rotates, scales or changes light. In both
  variants the looks are the same, so the ten worlds are five tests.
- **What it does, from its looks.** In nine worlds of ten the module finds
  the three behaviours as three kinds from the records alone (97 to 100%
  of objects with decisive records in the right kind), learns that shape
  matters in one variant and colour in the other with 97 to 99% of the
  coupling weight, and then treats a never-met object, of a width never
  met, as well as if it were told the behaviour, and picks the grip that
  takes it. One kind for all takes 39%.
- **The individual over the kind.** An individual whose behaviour
  contradicts its looks is handled right when recognised (93.5 and 94.1%
  of effects, the take in 100.0%), because its own records are kept with
  its stored pattern and outweigh the looks; from looks alone the take
  fails every time.
- **A belief over kinds is a mixture.** Weighting each kind's answer by
  the belief equals passing the winner here, because looks give a
  confident kind; averaging the kinds' couplings costs up to 2 points.
- **Thirty episodes are nearly enough** (by shape: effects 91.6, take
  95.2, against 91.2 and 92.5 at 300), with one world of five failing at
  each of 30, 100 and 300.

**Where it fails, and the weak part.** In world 5 by shape the kind
learner kept a fourth kind of four objects. Checked: that kind predicts
the fixed objects as well as the fixed kind does (172 of 174 tie), so only
2 fixed objects counted as having a settled kind, the couplings from looks
to kind were learned without fixed examples, and the kind read from looks
was wrong on 17% of the scored steps. The same happened at 100 episodes in world 3; at 30
episodes world 5 found two kinds. The learner took six versions and
carries five hand-set numbers (two surprises, a twentieth of the records,
three members, three failed openings, four moves). By this project's own
lesson, that many patches mark a weak principle: grouping objects by which
energy predicts them needs a better rule than mine.

Other limits: the masks, contact and "held" are given; behaviour depends
on one attribute, not a conjunction; the three behaviours differ only in
how the grip acts.

## 8. Decision

**Keep** the object module in this form: where as the place of the
object's frame, what as memories in that frame settled together with it,
does as a belief over kinds that select couplings, looks coupled to kinds,
and the individual's own records kept with its pattern. The rule that
finds the kinds is not kept as it stands. Next, with the user's go-ahead,
one of: a principled rule for kinds (for example each object's own sparse
difference from the shared couplings, grouped by a memory like shape and
colour are); binding by a known pattern, which is the feedback card and
would let a fixed object be found without true masks; or looks that
rotate and scale.
