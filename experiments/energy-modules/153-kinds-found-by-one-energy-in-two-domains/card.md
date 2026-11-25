---
id: "153"
title: kinds of behaviour found by one energy, in sprites and in a stream of words
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-20
---

# 153: kinds found by one energy, in two domains

Roadmap step 2, rung 2, on the direction under trial. Card
[152](../152-an-object-module-that-knows-what-an-object-is-and-does/card.md)
kept the object module and rejected the rule that finds its kinds of
behaviour (six versions, five hand-set numbers, one world of five
failing). The user, 2026-11-20, on the proposal to replace it and to test
it outside sprites: "This all sounds good", and before that "we shouldn't
get *too* caught up on object recognition from these small experiments.
We should stress test this model in various forms such as learning
language, grid puzzles, and these robo environments". Runs take about ten
seconds per world. Biological counterparts: latent-cause inference, in
which an animal opens a new cause when the old ones predict too badly
(Gershman and Niv 2010, after Anderson 1991); event segmentation by
prediction error (Zacks), here across objects instead of across time;
word classes learned from the frames words occur in (Mintz 2003), and
children giving a never-heard word the behaviour its form suggests (Berko
1958).

## 1. Question

Can the kinds of behaviour be found by lowering one energy, with no
hand-set thresholds, so that the same code and settings find the three
behaviours in every sprite world and the word classes in a stream of
words? P7.

## 2. What changes

One component: how kinds are found and tied to looks
(`tools/card153/kinds.py`). The object module of card 152, its tokens and
its tests are unchanged. A second world is added
(`tools/card153/lang.py`).

```
before (152)  fit, give each object the kind with fewest mispredicted records, then open / merge / dissolve kinds by
              rules: two surprises, a twentieth of the records, three members, three failed openings, four moves.
              Looks-to-kind couplings taught only by objects whose kind is ahead by one misprediction or more.
after         one energy, lowered throughout:
                L = sum over items of E(item, its kind) - log P(the grouping)
                E      minus the log probability of what the item did (its effect tokens together, given the
                       record's other tokens) under the kind's couplings, summed over its records, in nats
                P      the Chinese-restaurant prior of latent-cause models, concentration 1
              1. every item learns its own sparse difference from its kind's couplings (all at once);
              2. each difference is a candidate kind; the one that lowers L most, counting only what it explains of
                 other items' records, is stored; repeated until none lowers L;
              3. kinds are fitted again on their members; items take the kind that lowers L; two kinds merge if that
                 lowers L; a kind left with one member is given up; kept only if L came out lower; then 1 again.
              Looks-to-kind couplings taught by every item: first by its records' belief over kinds, weighted by what
              the records say, then three rounds by its belief from looks and records together; their common
              strength is refitted without the sparsity penalty.
```

An item is one object or one word; a record is one step it took part in.
Numbers set by hand: the prior's concentration (1). The sparsity of an
item's own difference is the kinds' own (0.001). Compute caps: 4 rounds,
4 fits a round, 3 rounds for the looks.

The second world. Twelve function words and content words of four classes
(singular noun, plural noun, verb, adjective) in sentences from a small
grammar ("these (A) P are A .", "a N will V some P ."). A word's form has
a suffix and an onset, six values each; the class goes with the suffix in
one family of worlds and with the onset in the other. 100 words per
class with Zipf frequencies; five frequent words of each class have the
form of another class. A record is one occurrence of a word: the token
two back, and as effect tokens the token before and the token after (a
function word, or "a content word").

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Which records belong to one item is given (masks and recognition in sprites; word boundaries in the stream) | no | no | no | no | declared: binding and segmentation are other cards |
| A record is a few discrete tokens with a small set of joint effect values (162 in sprites, 169 in the stream) | partly | yes | yes | no | declared: continuous effects need a population code; more effect tokens need settling in place of enumeration |
| Which words are function words is given | - | - | - | - | declared: in a real stream, the most frequent words (Mintz 2003) |
| Behaviour goes with one attribute of the looks | partly | partly | partly | partly | declared, as in card 152 |
| Some item has enough records of its own to seed a kind | yes | yes | partly | yes | declared: tested at 30 episodes and at 100 to 400 sentences |

## 3. Dependencies

Card 152 (the object module, its worlds and tests), card 148 (couplings
by pseudolikelihood), card 151 (store what nothing fits, merge what fits
each other). LITERATURE, "Energy modules and binding": CRAM. The
Chinese-restaurant prior is the standard one of latent-cause models.

## 4. Data check

Sprites: as card 152 (about 480 to 510 individuals in 300 episodes, 65 to
75 in 30). Word stream, development world 0, 1,000 sentences: 3,055
occurrences of 368 of the 400 words; 132 words are met at most twice; the
20 words that break the rule are all met (5.4% of known words). The test
has 600 sentences of words never met (about 1,850 occurrences), each a
new word whose form follows the rule.

## 5. Feasibility gate

- **Upper bound.** Sprites: the true behaviour of every object given
  (card 152). Word stream: the true rule from form to class with the share
  of known words that break it, read through couplings fitted with the
  true classes; no reader of forms can do better, because nothing in a
  form says a word is not an exception. "True class given" is reported as
  a bound no reader of forms can reach.
- **Trivial baseline:** one kind for all items.

Result of the gate, before the main run: passed. Sprites, development
world 0: effects 92.2% with the true behaviour against 71.3% with one
kind, take 100.0% against 40.7% (card 152); at 30 episodes the bound is
91.5% and 100.0%. Word stream, development worlds (by suffix, seed 0; by
onset, seed 100): the token after a never-met word costs 0.900 and 0.917
nats for the best reader of forms (0.864 and 0.882 with the true class
given) against 1.200 and 1.205 with one kind.

## 6. Success criteria and prediction

The same code and settings in every run.

1. **Every sprite world.** Seeds 1 to 10, by shape and by colour, 300
   episodes (seeds 1 to 5 are card 152's test worlds, 6 to 10 are new):
   in each of the 20 worlds, effects on never-met objects within 3.0
   points of that world's true-behaviour bound, and the take action right
   for at least 90.0%. Card 152's learner failed world 5 by shape (83.5
   against 92.7; take 62.3).
2. **Little experience.** Seeds 1 to 5, by shape and by colour, at 30 and
   at 100 episodes: take at least 90.0% in each of the 20 runs. Card
   152's learner, by shape only: 78.8 in one world at 30, 60.9 in one at
   100.
3. **A second domain.** Word stream, 1,000 sentences, by suffix (seeds 1
   to 5) and by onset (seeds 6 to 10): in each of the 10 worlds the
   module closes at least 90.0% of the gap, in nats on the token after a
   never-met word, between one kind for all and the best reader of forms;
   and at least 90% of the looks-to-kind coupling weight lies on the
   attribute that matters.

Reported without a bar: the stream at 100, 200 and 400 sentences; card
152's learner with this card's looks couplings (shape, seeds 1 to 5, at
300 and at 100), to tell the two changes apart; known items that break
the rule; in the stream, no kinds at all (the form's attributes coupled
straight to the context); kinds found and their purity.

Development (sprite world 0, both variants, at 30, 100 and 300; stream
seeds 0 and 100). What was tried and dropped: an energy that scored each
effect token given the others (the effect tokens explain each other, so
no kind paid); charging each kind's couplings at the fit's own L1 rate
(no candidate paid); teaching the looks by the records' belief alone
(belief too unsure: 84% and 81% of the gap to the true class, against 88%
and 85% after three rounds with looks and records together). On the
development worlds the final form gives: sprites, effects equal to the
bound within 0.1 at 100 and 300 episodes (92.2, 93.5, 92.1, 93.5) and
89.8 against 91.5 and 93.7 against 93.6 at 30, take 96.9 to 100.0;
stream, 98.3% and 95.0% of the gap, 98.6% and 98.0% of the weight on the
right attribute. The bars were written after seeing these. **A known
limit, seen in development:** at 100 sentences the stream's words have
one or two records each and the rule finds no classes (about a tenth
of the gap); at 400 it leaves a third of the words in one mixed kind.

Prediction: 1 and 2 pass, with the risk that a new world shows a failure
development did not. 3 passes, with some worlds near the bar.
Budget: about 15 minutes of runs, none over 5 minutes.

## 7. Result

Numbers in [results.json](results.json). Sprites: "effects" is the share
of object-steps in contact or held whose displacement and held-after are
exactly right, "take" the share of objects for which the proposed action
is right in the simulator; both for objects never met, kind from the
first frame. Stream: the cost in nats of the token after a never-met
word, lower is better.

| Sprites, 300 episodes, seeds 1 to 10 | By shape: effects | take | By colour: effects | take |
|---|---|---|---|---|
| True behaviour given (bound) | 93.2 | 100.0 | 92.1 | 100.0 |
| The module, kinds by this card's rule | **93.2** | **100.0** | **92.1** | **100.0** |
| One kind for all | 74.5 | 39.8 | 72.7 | 39.7 |
| Largest shortfall from the bound in any world | 0.0 | 0.0 | 0.0 | 0.0 |

Three kinds in all 20 worlds; 98.3% or more of the looks-to-kind weight
on the attribute that matters.

| Sprites, little experience: take in each of seeds 1 to 5 | 30 episodes | 100 episodes |
|---|---|---|
| This card's rule, by shape | 99.4, 98.7, **55.0**, 100.0, 98.3 | 100.0, 100.0, 100.0, 97.9, 100.0 |
| This card's rule, by colour | 100.0, 100.0, 94.6, 100.0, 100.0 | 100.0 in all |
| Card 152's learner with this card's looks couplings, by shape | 99.4, 98.7, 98.6, 100.0, **80.8** | 100.0, 100.0, 100.0, 97.9, 100.0 |
| Card 152 as it was, by shape | 100, 98.7, 98.6, 100, 78.8 | 100, 100, 60.9, 100, 100 |

| Word stream, 10 worlds; nats | 1,000 sentences | 400 | 200 | 100 |
|---|---|---|---|---|
| True class given (no reader of forms can reach it) | 0.866 | 0.876 | 0.907 | 0.981 |
| Best reader of forms (bound) | 0.901 | 0.921 | 0.967 | 1.060 |
| The module: kind from the form | **0.903** | 0.927 | 1.010 | 1.234 |
| No kinds: the form coupled straight to the context | 0.965 | 0.993 | 1.064 | 1.260 |
| One kind for all | 1.203 | 1.215 | 1.254 | 1.342 |
| Gap closed, mean (lowest to highest world), % | **99.2** (96.5 to 100.9) | 98.0 (94.1 to 100.7) | 84.9 (79.1 to 89.1) | 38.2 (-13.7 to 92.0) |
| Worlds in which four kinds are found | 10 | 10 | 4 | 1 |

At 1,000 sentences: 97.7 to 99.2% of the words whose records tell are in
the right class; 96.4% or more of the looks-to-kind weight is on the
attribute that matters; the token after is the likeliest one for 60.5%
of new words against 47.1% with one kind. A world can pass 100% because
the bound allows for rule-breakers and no new word in the test is one.

| Criterion | Verdict |
|---|---|
| 1. Every sprite world at 300 episodes: effects within 3.0 of its bound, take at least 90.0% | **pass**, 20 of 20 (largest shortfall 0.0; take 100.0 in all) |
| 2. Take at least 90.0% in each of 20 runs at 30 and 100 episodes | **fail**, 19 of 20 (world 3 by shape at 30 episodes: 55.0) |
| 3. Word stream, 1,000 sentences: at least 90.0% of the gap closed and 90% of the weight on the right attribute, in each of 10 worlds | **pass**, 10 of 10 (lowest 96.5 and 96.4) |

What the card shows.

- **One energy does the work of the thresholds.** With one hand-set
  number the rule finds exactly three kinds in all 30 sprite runs at 100
  and 300 episodes, and exactly four word classes in all 20 stream runs
  at 400 and 1,000 sentences. The same file runs both; nothing in it
  knows about sprites or words.
- **What cured card 152's failure was the looks change, not the rule.**
  Card 152's learner with this card's looks couplings passes world 5 by
  shape at 300 episodes (92.7 against the bound's 92.7, take 100.0) while
  still keeping its spare fourth kind, as it does in 3 of its 10 runs at
  100 and 300 episodes. Two kinds that predict the same do no harm once
  every item teaches the looks in proportion to what its records say and
  each kind's answer is weighted by the belief. On sprites at 100 and
  300 episodes the new rule adds the right number of kinds and nothing in
  the scores.
- **At 30 episodes the two rules are equally unreliable.** Each loses one
  world of five by shape, a different one.
- **Kinds earn their place in the stream.** Coupling the form straight to
  the context, with no kinds, closes 79% of the gap at 1,000 sentences;
  the module closes 99%.
- **The individual over the kind, again.** For known words that break the
  rule, the word's own records give 0.867 nats against 0.866 with the
  true class given; the form alone gives 3.345. In sprites, known
  exceptions: take 100.0% with their own records, 0.0% from looks.

**Where it fails, and why.** Both failures are the same weakness: a kind
can only be proposed by one item's own difference, and a proposal is
credited only with what it explains.

- *Sprites, world 3 by shape, 30 episodes* (checked by replaying the
  run): the four seeds stored were two sticky and two takeable objects.
  No fixed object's difference lowered the energy, so fixed and takeable
  stayed one kind (take 0% on takeable objects). Splitting them would
  also let the takeable kind predict its own members better, and the
  proposal is not credited with that.
- *Stream, 100 and 200 sentences* (reasoned from the energy, not
  separately tested): most words have one or two records. One record
  favours the right class by about one nat, and joining a small new kind
  costs more than that under the prior, so words do not move one at a
  time; and no single word has enough records to make a good candidate.
  The true classes are still worth 0.36 nats at 100 sentences, so the
  information is there.

Other limits: which records belong to an item is given in both worlds
(masks and recognition; word boundaries), and so is which words are
function words; the grammar is a toy; the energy enumerates the joint
values of the effect tokens. Card 152's learner was not run on the
stream: it counts wrong predictions of a determined outcome, and the
next word is not determined.

## 8. Decision

**Keep.** The rule replaces card 152's learner: one energy and one
hand-set number in place of five thresholds, the right number of kinds
wherever there is enough experience, and the same code in a second
domain. The looks-to-kind couplings taught by every item are kept with
it. The verdict is a fail because criterion 2 was written for every run
and one of twenty failed. Next, with the user's go-ahead, the two
changes the failures point to: judge a proposed kind after both it and
the kind it leaves are fitted again; and let items that share a looks
code propose together, which is the looks acting as the prior over the
grouping (the user, 2026-11-20: "you make inference based on the sensory
data available to you"). Then the grid-puzzle world.
