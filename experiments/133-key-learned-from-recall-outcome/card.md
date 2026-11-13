---
id: "133"
title: a recall key learned from whether recall paid off — which remembered items matter for this touch
rung: 3
serves: [P4, P12, P21]
status: done
verdict: fail
arch_version: og4
date: 2026-11-12
---

# 133: a key learned from recall's outcome

Roadmap step 3, rung 3. Second card of the memory week (the user,
2026-11-12: one component, perfected in isolation, on true visible
state). [Card 132](../132-memory-test-bed/card.md) found that keeping each object's
last seen value stores the right content (the remembered lock was
right in 1,955 of 2,226 pulls), and that recall then reads it badly:
70.2% on pulls against 94.8% with the true lock. The key gave every
remembered item a fixed weight, and the irrelevant ones outvoted the
lock.

Biological counterpart: learned attention over the dimensions of
remembered cases. People shift weight to the dimensions that predict
the outcome, and exemplar models of human category learning do this
by error (ALCOVE, Kruschke 1992). In the brain, prefrontal control
biases which features cue hippocampal retrieval.

## 1. Question

Can the weighting of the recall key be learned from the error of
what recall returned, with no label naming what matters, so that
recall over last-seen contents comes close to recall with the true
hidden values? P4 (memory).

## 2. What changes

One component: the measure of similarity that recall uses.

```
132:  key = [touch, touched object, last seen value of each variable], fixed weights;
      5 nearest stored events, majority outcome
133:  the same contents, passed through a learned map before the distance is taken
      arm 1  one learned weight per item of the key (which items matter, the same for
             every touch)
      arm 2  a small network over the key (which items matter may depend on the touch)
      trained on the store only: each stored event is recalled from the other episodes,
      and the map is adjusted to reduce the error of the recalled effect
      effect = how much the touch changed each variable (the cube's place, each slide,
               each lock); no label for "moved" or "cube inside" is given
```

The effect replaces card 132's two hand-picked outcomes as what is
recalled. The scored questions are read from it: a pull moved if the
recalled slide change is over 1 cm; the cube was inside if the
recalled cube change on a reopening is over 5 cm (the drawer carries
it). One more rule of the narrow view follows: a variable the touch
changes is seen after it. Card 132's fixed key and ceiling are run
again in this form, so the comparison is like for like.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Perception is perfect for what is in view | no | near | yes | no | declared for the memory week |
| Memory contents are a list of named values | no | no | no | no | declared; arm 2 takes any vector, arm 1 does not |
| A touch's effect is observed right after it | mostly (delayed effects exist) | holds | holds | holds with contact sensing | card 120 |
| The store holds events like the present one | needs much play | holds | few (new games) | needs practice | 400 episodes; 100 also scored |

## 3. Dependencies

- Card 132: the test bed, the last-seen contents, its bounds.
- A key trained through the recall it feeds: Neural Episodic Control
  (graph, `paper/neural-episodic-control`); learned weights per
  dimension for nearest-neighbour recall: neighbourhood components
  analysis (Goldberger et al. 2004), in the database as its Bayesian
  form.
- Recall from other episodes only while training (leave the event's
  own episode out), as in versions 20–27.

## 4. Data check

From card 132, same store and test episodes: 2,226 test pulls, 222
of which do not move; 198 reopenings, 40 with the cube inside, 30 of
those with the cube seen earlier in the episode. The store holds 245
reopenings (40 with the cube) and about 2,900 pulls to learn from.
To be confirmed at the gate: that the cube's change on a reopening
with the cube inside is over 5 cm in nearly every case.

## 5. Feasibility gate

- **Upper bound:** the key with the true hidden values, recalling
  effects: expected near card 132's 94.8 on pulls and 100 on
  reopenings. Must reach 93 and 95.
- **Trivial baseline:** card 132's fixed weights, recalling effects:
  expected near 70 and 80.

Result of the gate (2026-11-12): pulls pass, truth 96.0 against
fixed weights 76.7. Reopenings do not pass as written: the truth key
scores 88.1, below the fixed weights' 96.5. The new rule of the view
(a carried cube is seen) already lets last-seen contents answer
reopenings, so the bound is shown by the fixed key and the open gap
is on pulls only. The cube moves at least 9.1 cm on every reopening
with the cube inside (bar 5 cm). The main run went ahead on that
basis.

## 6. Success criteria and prediction

Balanced accuracy on the test episodes, narrow view. Proposed, to be
fixed with the user.

1. **Recall reads what memory holds.** The better learned arm
   reaches at least 88% on pulls (fixed weights 70.2, truth 94.8)
   and at least 95% on reopenings where the cube was seen earlier in
   the episode (fixed weights 89.5).
2. **From 100 episodes.** Trained and stored on 100 episodes, at
   least 85% on pulls.
3. **For the right reason.** With the remembered lock shuffled
   between test events, pulls fall to at most 65%; with the
   remembered cube position shuffled, pulls change by at most 2
   points.

Reported without a bar: reopenings where the cube was not seen yet
(the prior's job), and arm 1 against arm 2.

**Prediction.** 1 passes on pulls for both arms. On reopenings arm 1
may fail: one weighting for every touch cannot attend to the lock
for a pull and to the cube for a reopening, when both are touches of
the same drawer; arm 2 can. 2 is uncertain. 3 passes if 1 does.

**Budget.** Training on 7,800 stored events: under 5 minutes per
arm, capped. No rendering.

## 7. Result

Run 2026-11-12 (`tools/card133/key.py`, about a minute per setting,
capped; one seed; numbers in [results.json](results.json)). Balanced
accuracy on the test episodes.

| Key | Pulls, 400 episodes | Reopenings (cube seen earlier), 400 | Pulls, 100 episodes | Reopenings, 100 |
|---|---|---|---|---|
| Fixed weights (card 132) | 76.7 | 96.5 | 71.5 | 93.1 |
| One learned weight per item | 64.1 | 36.9 | 80.8 | 93.1 |
| Learned network | 87.5 | 93.2 | 73.3 | 82.8 |
| True hidden values | 96.0 | 88.2 | 93.7 | 92.7 |

| Criterion | Bar | Result | Verdict |
|---|---|---|---|
| 1. Recall reads what memory holds | pulls at least 88, reopenings at least 95 | network 87.5 and 93.2 | fail (by 0.5 and 1.8) |
| 2. From 100 episodes | pulls at least 85 | weights 80.8, network 73.3 | fail |
| 3. For the right reason | lock shuffled: pulls at most 65; cube shuffled: change at most 2 | network 54.5; 87.5 to 88.0 | pass |

- **The network learned what to attend to, from the effect alone.**
  Pulls rise from 76.7 to 87.5 (truth 96.0); shuffling the
  remembered lock takes them to 54.5, shuffling the remembered cube
  changes nothing. No label said that locks matter for pulls.
- **It paid for that on reopenings** (96.5 to 93.2), and it needs
  data: at 100 episodes it is below the fixed weights on reopenings
  (82.8 against 93.1) and barely above on pulls.
- **One weight per item is the better form at 100 episodes** (pulls
  80.8) and broke at 400: its weights on the touched object's
  identity ran to the thousands and the rest stopped counting. That
  is a fault of the training (nothing bounds the scale), not a
  finding about the form.
- All 40 reopenings with the cube inside had the cube seen earlier,
  so the "not seen yet" split has no cases under the new rule of the
  view.

## 8. Decision

**Revise.** The principle holds up: a key trained on the error of the
recalled effect finds the lock by itself. Neither form is good
enough yet, for two separate reasons to fix one at a time: the
per-item weights need a bounded scale, and the network needs the
low-sample behaviour that the fixed key has. The natural next form is
version 27's rule applied to the key: the fixed weighting as the
prior, the learned one taking over as its stored evidence grows.
