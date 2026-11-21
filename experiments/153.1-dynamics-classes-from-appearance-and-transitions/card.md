---
id: "153.1"
title: dynamics classes inferred from appearance and transitions together
rung: 2
serves: [P7]
status: done
verdict: pass
arch_version: og4
date: 2026-11-20
---

# 153.1: dynamics classes from appearance and transitions together

Roadmap step 2, rung 2, on the direction under trial. Card
[153](../153-kinds-found-by-one-energy-in-two-domains/card.md) failed
where no single object had enough transitions to propose a class (one
sprite run of twenty at 30 episodes; the word stream at 100 and 200
sentences). The user, 2026-11-20: "you make inference based on the
sensory data available to you", and, on this proposal, "Proceed". Runs
take 40 to 100 seconds per world, and 80 to 340 seconds at 300 episodes
(card 153: about 10). Biological counterpart: the same as
card 153 (latent-cause inference), with the cause inferred from how a
thing looks before anything is done to it (affordances).

**Terms** (cards 152 and 153 used everyday words for these; the user
asked for the standard ones).

| Here | Symbol | Cards 152–153 |
|---|---|---|
| object (a sprite object, or a word type) | $i$ | item, individual |
| appearance features: the attractor indices of the object's two appearance memories (shape, colour; suffix, onset) | $a_i$ | looks |
| observed transitions of the object | $D_i$ | records |
| dynamics class, a latent variable; class $k$ is an energy function over a transition's tokens with couplings $J_k = J_0 + \Delta_k$ | $z_i \in \{1..K\}$ | kind |
| appearance prior, a pairwise energy between $a$ and $z$ | $p(z \mid a)$ | looks-to-kind couplings |

## 1. Question

Does inferring $z_i$ from $p(z_i \mid a_i)\,p(D_i \mid z_i)$ during
learning, with classes proposed from pooled transitions and accepted on
held-out transitions, find the dynamics classes when each object has few
transitions, without losing anything where it has many? P7.

## 2. What changes

One component, the inference of the dynamics classes
(`tools/card153.1/classes.py`). The couplings, the appearance prior's
form, the worlds and the tests are card 153's.

```
before (153)  minimise  sum_i -log p(D_i | z_i) - log CRP(z);  a class is proposed from one object's residual
              couplings and credited with what it explains of other objects, before any refit; a class of one
              object is dropped
after         posterior   p(z_i | a_i, D_i)  proportional to  p(z_i | a_i) p(D_i | z_i),  in learning as at test
              proposals   from one object's residual couplings, or from the pooled transitions of all objects that
                          share an appearance attractor
              acceptance  the proposal is fitted (couplings, prior, assignments alternating) and kept only if it
                          lowers  S = - sum_i log sum_k p(z_i = k | a_i, D_i^fit) p(D_i^held | k),
                          couplings and prior fitted on one half of the transitions, scored on the other, both ways
```

No prior on the number of classes, no threshold, and no hand-set number
in the rule. Compute caps: 3 proposals tried a round, 4 alternations a
fit, 8 rounds.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Which transitions belong to one object is given | no | no | no | no | declared, as card 153 |
| A transition is a few discrete tokens with a small set of joint effect values | partly | yes | yes | no | declared, as card 153 |
| Appearance comes as two discrete attractor indices | partly | yes | yes | partly | declared: more memories, or beliefs over them, need the same prior over more inputs |
| The dynamics class depends on one appearance feature | partly | partly | partly | partly | declared, as card 152 |

## 3. Dependencies

Card 153 (the energy $-\log p(D_i \mid k)$ over a transition's effect
tokens taken together, residual couplings, the appearance prior fitted by
expectation-maximisation), card 152 (the object module and its tests).

## 4. Data check

As card 153. At 100 sentences 105 to 122 of about 140 words are met at
most twice; at 30 episodes there are 65 to 82 objects.

## 5. Feasibility gate

Bounds and baselines are card 153's and were measured there on the same
test worlds: sprites, true dynamics class given against one class; word
stream, the best reader of appearance against one class (at 100
sentences 1.060 against 1.342 nats, at 200 sentences 0.967 against
1.254). Passed.

## 6. Success criteria and prediction

The same code and settings in every run; the test worlds and seeds are
card 153's.

1. **Sprites with little experience.** 30 and 100 episodes, seeds 1 to 5,
   by shape and by colour: the take action right for at least 90.0% in
   each of the 20 runs (card 153: 19 of 20).
2. **Word stream with little text.** At 200 sentences at least 90.0% of
   the gap closed in each of the 10 worlds (card 153: 79.1 to 89.1); at
   100 sentences at least 70.0% on the mean of the 10 worlds (card 153:
   38.2).
3. **Nothing lost with enough experience.** Sprites at 300 episodes,
   seeds 1 to 10, both variants: effects within 3.0 points of each
   world's bound and take at least 90.0% in each of the 20 worlds. Word
   stream at 1,000 sentences: at least 90.0% of the gap in each of the
   10 worlds.

Development (sprite seeds 0 and 11 to 15; stream seeds 0 and 100), one
version, no setting tuned: sprites at 30 episodes, three classes and
take 96.9 to 100.0% in all 9 runs; at 300 episodes effects equal to the
bound; stream, gap closed 92.1 and 71.1% at 100 sentences (three and
four classes), 93.3 and 96.8% at 200, 98.5 and 95.7% at 1,000. The bars
were written after seeing these.

Prediction: 1 and 3 pass. 2 passes at 200 sentences; at 100 the mean is
near its bar. Budget: about 40 minutes of runs, none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). The measures are card 153's.
Sprites: "effects" is the share of object-steps in contact or held whose
displacement and held-after are exactly right, "take" the share of
objects for which the proposed action is right in the simulator; both
for objects never met, $z$ from $p(z \mid a)$ at the first frame.
Stream: the cost in nats of the token after a never-met word; "gap
closed" is the share of the distance from one class to the best reader
of appearance.

| Sprites, little experience: take in each of seeds 1 to 5 | 30 episodes | 100 episodes |
|---|---|---|
| This card, by shape | 99.4, 98.7, **100.0**, 100.0, 98.0 | 100.0, 100.0, 100.0, 97.9, 100.0 |
| This card, by colour | 100.0, 100.0, 94.6, 100.0, 100.0 | 100.0 in all |
| Card 153, by shape | 99.4, 98.7, 55.0, 100.0, 98.3 | 100.0, 100.0, 100.0, 97.9, 100.0 |

Three classes in all 20 of these runs.

| Word stream, 10 worlds; nats | 1,000 sentences | 200 | 100 |
|---|---|---|---|
| Best reader of appearance (bound) | 0.901 | 0.967 | 1.060 |
| This card | **0.904** | **0.972** | **1.104** |
| Card 153 | 0.903 | 1.010 | 1.234 |
| One class for all | 1.203 | 1.254 | 1.342 |
| Gap closed, this card: mean (lowest to highest world), % | 99.0 (95.3 to 101.4) | **98.0** (93.1 to 102.8) | **84.3** (73.0 to 100.7) |
| Gap closed, card 153 | 99.2 (96.5 to 100.9) | 84.9 (79.1 to 89.1) | 38.2 (-13.7 to 92.0) |
| Worlds with exactly four classes (most in any world) | 9 (5) | 7 (8) | 6 (8) |

| Sprites, 300 episodes, seeds 1 to 10 | By shape: effects | take | By colour: effects | take |
|---|---|---|---|---|
| True dynamics class given (bound) | 93.2 | 100.0 | 92.1 | 100.0 |
| This card | **93.2** | **100.0** | **92.2** | **100.0** |
| One class for all | 74.5 | 39.8 | 72.7 | 39.7 |

Largest shortfall from the bound in any of the 20 worlds: 0.0.

| Criterion | Verdict |
|---|---|
| 1. Sprites at 30 and 100 episodes: take at least 90.0% in each of 20 runs | **pass**, 20 of 20 (lowest 94.6; card 153: 19 of 20) |
| 2. Stream: 200 sentences at least 90.0% in each of 10 worlds; 100 sentences at least 70.0% on the mean | **pass** (lowest 93.1; mean 84.3) |
| 3. Nothing lost: sprites at 300 episodes within 3.0 of the bound and take at least 90.0% in 20 worlds; stream at 1,000 sentences at least 90.0% in 10 worlds | **pass**, 20 of 20 and 10 of 10 (lowest 95.3) |

What the card shows.

- **The classes are proposed by appearance.** Of the accepted proposals,
  134 of 139 in sprites and 72 of 110 in the stream came from the pooled
  transitions of objects sharing an appearance attractor, not from one
  object. Where each object has one or two transitions (105 to 122 of
  about 140 words at 100 sentences) this is the only source that works.
- **Held-out prediction replaces the prior on the grouping.** No
  concentration, no threshold; the rule keeps a class if it predicts
  transitions it was not fitted on.
- **Nothing was lost with enough experience**, at 4 to 34 times card 153's
  run time.

Where it is weak.

- **Extra small classes.** Held-out acceptance admits classes that help
  a little: up to 8 classes for 4 true ones at 100 and 200 sentences
  (sizes such as 32, 32, 27, 12, 7, 10, 9, 2), one fifth class of 10
  words at 1,000 sentences, and a class of one object in 2 of 20 sprite
  worlds at 300 episodes. They cost nothing measured here, because each
  class's prediction is weighted by the posterior, but the number of
  classes is no longer exactly right (card 153: exactly 3 and 4 with
  enough data). The four worlds with more than four classes at 100
  sentences are the four lowest (73.0 to 76.0).
- **Rule-breaking words with few transitions.** For known words whose
  appearance misleads, at 100 sentences the posterior from appearance
  and their own transitions costs 1.413 nats, worse than one class for
  all (1.312; card 153: 1.308): a stronger appearance prior outweighs
  one or two transitions. At 200 sentences 1.034 (card 153: 1.064), at
  1,000 sentences 0.867 against 0.866 with the true class given. In
  sprites, take for known exceptions is 61.5% at 30 episodes (card 153:
  58.3), 93.9% at 100 and 100.0% at 300.
- **Given, as in card 153:** which transitions belong to one object
  (masks, word boundaries), which words are function words, a toy
  grammar, effect tokens enumerated jointly. The proposals assume the
  appearance comes as two discrete attractor indices.

## 8. Decision

**Keep.** Inferring the dynamics class from $p(z \mid a)\,p(D \mid z)$
during learning, with proposals pooled over an appearance attractor and
accepted on held-out transitions, replaces card 153's rule: it passes
where card 153 failed and loses nothing where card 153 passed. Carried
forward as known limits: extra small classes, and the run time. Next,
with the user's go-ahead: the grid-puzzle world.
