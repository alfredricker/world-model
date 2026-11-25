---
id: "154"
title: feedback from object modules to binding modules
rung: 2
serves: [P7]
status: done
verdict: fail
arch_version: og4
date: 2026-11-20
---

# 154: feedback from object modules to binding modules

Roadmap step 2, rung 2, on the direction under trial. The user,
2026-11-20: "strip away the information that we feed to the model ...
end to end learning", then, on hearing that every layer so far only
passes information upward: "Maybe we should do that first", and "yes" to
this design. Cards 152 to 153.1 were fed the simulator's masks; without
them an object is bound only while it moves. Runs take about four
minutes per world. Biological counterpart: recurrent figure-ground
processing from higher visual areas back to V1 (Lamme; Roelfsema), and
infants who segment a stationary object only after earlier experience
with it (Needham).

**Terms.**

| Term | Symbol | Meaning |
|---|---|---|
| point | $p$ | a pixel-sized patch followed from frame to frame (card 147) |
| binding module | $b_p$ | one per point; its state is a belief over at most $K = 12$ objects |
| binding couplings | $J_{pq}$ | log-odds that $p$ and $q$ are parts of one object, accumulated from whether they moved together |
| object module | $k$ | one per object of a scene; its state is a stored pattern $m_k$ and a place $s_k$ |
| stored pattern | $m$ | for an object once bound by motion: where its points lie in a 16 by 16 frame, what they look like, and the mean couplings it had inside ($J^{in}_m$) and to everything else ($J^{out}_m$) |

## 1. Question

Does writing a recognised pattern's stored couplings back into the
binding layer bind a known object that has not moved in the scene, from
pixels alone, without spoiling binding by motion? P7.

## 2. What changes

One component, the path from object modules down to binding modules
(`tools/card154/feedback.py`). Motion, binding couplings and their
settling are cards 147 and 150's.

```
before   E(b) = -1/2 sum_pq J_pq b_p.b_q                 J from this scene's motion only; nothing bound at first sight
after    learning   a group of points that moved together in 3 or more steps, and that no stored pattern explains,
                    is stored as a pattern with its J_in and J_out
         first sight  every stored pattern is tried at every place; a pattern recognised at a place adds
                    J_pq += J_in  for p, q among the points it explains,  J_pq += J_out  for p explained, q not,
                    and sets its module's belief b_p for those points; then the binding layer settles as before
         recognition  score(m, s) = sum over pattern pixels not explained by a better pattern of
                    [match: -log q_mp | miss: log 1/2],  accepted if above log(patterns x places)
```

A pattern pixel matches when its error is within sensor noise (card
147's number). $q_{mp}$ is how often that pattern pixel matches by
chance, read from its matches at all places of the same frame; a pattern
pixel is taken to be as likely hidden as seen; the acceptance bar is the
prior against any one of the hypotheses. The best pattern is taken
first and the pixels it explains stop counting for the others. Hand-set
numbers: $K = 12$, the 16 by 16 frame, 3 steps of motion before storing.

**Not in this card.** A never-met object of a known shape, which I named
as a third case. A stored pattern is of one object; a never-met object
has a new texture and, in these tests, a new size, so feedback from
stored shapes would need a hand-built search over scale, which the user
ruled out on 2026-11-20. It belongs with couplings learned as a function
of appearance.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| An object looks the same, pixel for pixel, wherever it lies (no rotation, scale or lighting change) | no | yes | yes | no | declared: the match must become one over learned features |
| An object fits a 16 by 16 frame | no | yes | partly | no | declared |
| The view does not move | no | no | yes | partly | declared, as card 147 |
| Objects are met again (a pattern is of one individual) | partly | no: kinds, not individuals | partly | yes | declared: patterns of kinds are the next step |
| The memory may grow by one pattern per object ever moved | no | no | yes | no | declared: nothing is forgotten or merged yet |

## 3. Dependencies

Card 147 (motion front end, points, binding couplings: 97.9% of moved
things found), card 150 (the binding layer's settling per frame), card
152 (the sprite world with a pool of 40 objects that return).

## 4. Data check

In 100 training episodes of the three development worlds 21 to 26 of the
40 pool objects move in three or more steps; 12 to 19 never do (the
fixed ones, and those not taken). 88.5 to 100.0% of the pool objects
that moved leave a stored pattern whose mask overlaps the true one by
0.8 or more. The covered scenes hold 37 to 40 known things per world
that are a quarter or more hidden.

## 5. Feasibility gate

Development worlds (seed 0 by colour and by shape, seed 11 by shape),
known things at first sight:

- **Upper bound:** feedback with a memory stored from the true masks:
  100.0% found in each world.
- **Trivial baseline:** no feedback: 0.0% found in each world (nothing
  has moved).

Passed.

## 6. Success criteria and prediction

Test worlds: card 152's, seeds 1 to 10, by shape and by colour; 100
training episodes each. "Known" means the simulator's record says the
thing (the hand or a pool object) moved in three or more steps of
training. "Found": the label that covers most of the thing overlaps it
by 0.5 or more (intersection over union), is not the ground's label and
does not cover half of another thing.

1. **First sight.** 100 fresh scenes per world in which nothing has
   moved: known things found in at least 90.0% of cases in each of the
   20 worlds (no feedback: 0).
2. **Covered.** 300 scenes per world with 4 to 6 objects lying over each
   other: known things a quarter or more hidden, pooled over the 10
   worlds of each variant, found in at least 90.0% of cases and settled
   on a pattern stored from the same thing in at least 90.0%.
3. **No harm.** In every world: things never stored are recognised as a
   stored pattern in at most 2.0% of cases, and at most 0.05 labels per
   scene lie on the ground. Over 20 whole episodes per world, things
   that have moved are found within 3.0 points of the run without
   feedback, on the mean of the 20 worlds.

Development, one version (changes were bug fixes: a frame that may start
outside the image, a pattern used once, each pattern pixel's own chance
rate, the module's belief written with its couplings): first sight 100.0,
100.0 and 96.6%; covered and a quarter or more hidden 97.5, 100.0 and
97.4%; never-stored things recognised 0.0 to 0.7%; no label on the
ground; moved things found 98.3 against 98.4, 100.0 against 97.6 and
96.7 against 97.9 without feedback.

Prediction: all three pass; the risk is one world under 90 at first
sight because a pool object's pattern was stored badly. Budget: about 30
minutes of runs, none over 10 minutes.

## 7. Result

Numbers in [results.json](results.json). Twenty worlds, 100 training
episodes each; 69 to 105 patterns stored per world, none from a mask.

| Known things found, % | By shape | By colour |
|---|---|---|
| First sight, feedback: mean of 10 worlds (lowest world) | **98.7** (96.6) | **99.8** (98.3) |
| First sight, no feedback | 0.0 | 0.0 |
| First sight, feedback with a memory from the true masks (bound) | 100.0 | 100.0 |
| Covered, a quarter or more hidden: found | **96.3** (n 321) | **99.7** (n 298) |
| Covered, a quarter or more hidden: settled on its own pattern | 96.9 | 99.7 |
| Covered, a quarter or more hidden, bound | 99.7 | 100.0 |
| Whole episodes, known things that have not moved yet: feedback / none | 97.5 / 0.0 | 100.0 / 0.0 |
| Whole episodes, things that have moved: feedback / none | 97.8 / 97.3 | 98.3 / 98.0 |
| Pool objects that moved in fewer than 3 steps of training | 0.5 | 0.0 |

| Criterion | Verdict |
|---|---|
| 1. First sight: at least 90.0% in each of 20 worlds | **pass**, 20 of 20 (lowest 96.6) |
| 2. Covered, a quarter or more hidden: at least 90.0% found and 90.0% on its own pattern, per variant | **pass** (96.3 and 96.9; 99.7 and 99.7) |
| 3. No harm: never-stored things recognised at most 2.0% and at most 0.05 labels on the ground in every world; moved things within 3.0 points | **fail as worded**, 19 of 20 worlds (world 4 by shape: 5.4%); no label on the ground in any world; moved things 98.1 against 97.7 |

**The failure of criterion 3 is in my scoring** (checked by rerunning
world 4 by shape). All 7 cases at first sight are one pool object that
moved in 2 steps by the simulator's record, under the cut-off of 3 that
I used to call a thing "known", while the binding layer counted 3 and
stored a pattern of it whose mask is exactly right. It was recognised as
itself. Counting only recognition by a pattern stored from another
thing, that world has 0 of 129 at first sight and 1 of 1,097 in the
covered scenes. The other 19 worlds are at most 1.6% on the declared
measure.

What the card shows.

- **Feedback binds what motion cannot.** A known object that has not
  moved is bound in 96.6 to 100.0% of cases in every world, against 0
  without feedback, with every pattern learned from motion binding.
- **Covering is handled by explaining away**, with no mask: a pattern
  pixel explained by a better pattern costs nothing, an unexplained one
  costs log 2.
- **Recognition needed no threshold**: each pattern pixel's chance rate
  is read from the frame, and the bar is the prior against one
  hypothesis among patterns times places.
- **Binding by motion is not spoiled** (98.1 against 97.7).

Where it is weak.

- **The gap to the bound is pattern quality.** With patterns from true
  masks first sight is 100.0% in all 20 worlds. 82.6 to 100.0% of the
  pool objects that moved leave a pattern overlapping the true mask by
  0.8 or more; the rest are fragments or carry stray points.
- **An object that never moved is never bound** (0.0% in 19 worlds;
  the 5.4% of world 4 by shape is the object above): 10 to 24
  of the 40 pool objects per world, the fixed ones among them. In the
  worlds by shape the fixed shapes are never seen moving at all.
- **A recalled binding can decay within an episode**: known things not
  yet moved are found in 88.7% of frames in the worst world (by shape,
  seed 9). Not investigated.
- **The pattern is one object's pixels**: no change of size, turn or
  light, and a never-met object is not helped. The memory grows by a
  pattern per object ever moved (about 88 after 100 episodes) and the
  match tries every pattern at every place.

## 8. Decision

**Keep.** Feedback as stored couplings written back into the binding
layer is carried forward: it passes the two tests of what it is for, and
criterion 3 failed on a scoring cut-off, not on a false recognition. The
verdict stays a fail because the criterion was declared that way. Next,
with the user's go-ahead: the object modules of cards 152 to 153.1 fed
by this binding instead of the simulator's masks; and, for objects that
never move, binding couplings learned as a function of appearance.
