---
id: "102.2"
title: move together, act together — tokens of many tiles from common fate
rung: 1
serves: [P7, P6, P3]
status: done
verdict: pass
arch_version: 27
date: 2026-10-11
---

# 102.2: move together, act together

Drafted 2026-10-11 at the user's request; approved by the user the same
day ("move on to card 102.2") after card 102.1's first-run form was
kept. It takes up card 098.8's Appendix B (objects of many cells, as in
ARC-AGI-3's games) with the user's principle: tiles that move together
and act together are one object.

## 1. Question

Every token so far is one tile. MiniGrid's objects fill one tile each,
so nothing has tested anything else. In ARC's games a shape of many
cells moves as one, and in a robot's world every object covers many
pixels. One tile per token fails twice:
- **The aperture problem:** when a bar of one colour moves one cell,
  only its ends change, so its inner tiles look as if they stayed.
- **Scattered effects:** an effect on the object reads as separate
  effects on its tiles.

If tiles are grouped when they move or change together, is each object
one token, and is the action's effect on it predicted as a whole? Does
a known kind of shape group at first sight? Rung 1. Serves P7 (units
from continuity and interaction), P6 and P3.

## 2. What changes

One component: what a token is.

```
version 27:  token = one tile (a what at a place)
this card:   token = a group of tiles that share fate
             each step, the change from one view to the next is explained by layers (Wang and Adelson 1994):
               the background, shifted by the view's own motion (the shift that keeps most codes)
               + chains of tiles, each shifted one step relative to it
               + tiles changed in place; tiles that appear on ground or vanish into it are new or ended
             each tile is tracked by identity; two tracked tiles gain evidence of one object when they share
             a fate (a shift, or a change in place) and lose it when one moves without the other;
             a group is the tiles linked by positive evidence plus the prior (Bayes factor against independent fates)
             arm A reads a group as one token: its members' states are averaged and its fate predicted once
```

- **The aperture problem is solved by layers.** A tile that is not
  ground cannot simply vanish, so the inner tiles of a moved bar join
  its chain: what left each place was replaced by its neighbour.
- **Ground is learned:** the codes revealed and covered when exactly one
  tile moves. Of the two tiles that swap places, the mover is the code
  more often seen to change wherever it stands.
- **The prior is learned per world:** the share of pairs seen together
  that end their training episode as one object, found as a fixed point
  (empirical Bayes).
- **First sight:** arm A gets a head, "do i and j belong to one object?"
  (a dot product of their states). It is trained on hindsight (the
  groups at each training episode's end) and read only on objects that
  have not yet moved. Own evidence comes first.
- **Nothing is declared about shape, colour or adjacency.** A group
  need not be connected; a lake's tiles never move, so they stay apart.
- **Not in this card:** rotation or deformation, belief out of view
  (102.3), clicks (102.6).

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A layer moves rigidly (one shift for all its tiles) | yes | yes for shifts; some games rotate | no (3D, deformable) | declared; rotation is a later card |
| Fate is read per step | yes | yes | yes, at the frame rate | declared |
| Tiles on a grid | yes | yes | no (points) | kept from the adapter; the rule reads raw positions |
| Ground: something movers pass over | yes (grass, sand, path) | yes (the background colour) | yes (floor, table top) | learned from single-tile moves |
| How often tiles are one object | learned per world | learned | learned | the prior, from the world's own play |

## 3. Dependencies

- Card 102.1's kept form of arm A (softmax attention, raw positions).
- The placing step (cards 057, 061) and card 098.5's pair test, which
  are the one-layer and one-tile cases of these layers.
- Rows in [LITERATURE](../../LITERATURE.md): common fate (Wertheimer
  1923; Spelke 1990), layers (Wang and Adelson 1994), interactive
  perception (Bohg et al. 2017), log-likelihood ratio tests (Wald 1945).

## 4. Data check

Breaking world 2, "shapes" (Appendix A): 2,000 training and 500 test
episodes of 100 random steps.

| | Training | Test |
|---|---|---|
| Objects of two or more tiles (shapes and doors) | 6,988 | 1,756 |
| Pushes that moved a shape / door changes | 3,991 / 2,875 | 1,027 / 677 |
| Objects never moved or changed in their episode | 56% | 57% |
| Steps until an object first moves or changes (median, 90th percentile) | 23, 76 | 24, 76 |

- **Test shapes** are 7 of the 20 shapes of 2–5 tiles, never seen in
  training. Half the shapes are of one colour.
- **World 2's encoder** (card 098.3's recipe) gives 11 codes, each one
  kind.
- **Crafter's held-out worlds 8–9** are the control: 21,947 awake states.

## 5. Feasibility gate

- **Upper bound:** the true objects as tokens. Baselines: one token per
  tile, and colour and touching (the usual ARC heuristic, an engineered
  prior run only for comparison).
- **Check:** the layers give the true fate of at least 99% of moving or
  changing object tiles in world 2.
- **Result: passes.**
  - All 33,891 moving or changing object tiles get their true fate
    (100%), and every still one is called still.
  - Ground learned: floor in world 2; grass, sand and path in Crafter,
    plus rare codes seen 8–74 times.
  - The prior log-odds: −0.62 in world 2, −9.6 in Crafter.

## 6. Success criteria and prediction

1. **Objects become tokens.** In world 2's test play, once an object has
   moved or changed, its tiles are exactly one token in at least 95% of
   steps. First sight (the head alone) is reported.
2. **Effects on the whole object.** When the agent walks into a tile of
   a shape or a door that has moved or changed before, every tile of the
   object is predicted right in at least 95% of the upper bound's rate.
   One tile per token is the baseline; first interactions are reported.
3. **No false groups.** In Crafter's worlds 8–9, at most 1% of states
   hold a group of two or more tiles (every Crafter object is one tile).
   MiniGrid is left to the tiers run before adoption (the user,
   2026-10-11).

**Training format** (every form, 12,000 steps), fixed before the main
run by diagnostics:
- **Uniform draws** learned no push and no door change (0%).
- **Half of each batch from event steps** inflated the base rate:
  blocked pushes fell to 37%.
- **Prioritised replay** (Schaul et al. 2016; draws ∝ last loss^0.6)
  with importance weights learned no push.
- **Kept: the same replay without the weights:** pushes 33%, blocked
  95%, doors 100%, on per-tile tokens at 4,000 steps.

Arm A alone is scored (the vote is the same for every form).

**Prediction:** criterion 1 passes after one move; first sight carries
to new shapes of familiar tiles; colour and touching fails on
multicoloured shapes; Crafter's false links come from creatures moving
together by chance and decay.

**Decision rules:** keep if 1–3 hold; revise if only first-sight
grouping is weak; stop if the gate fails.

**Budget:** grouping about 2 minutes; three training runs about 15
minutes.

## 7. Result

Runs: `tools/card102.2/layers.py` (grouping), then `runs/102.2/chain.sh`
(criterion 2, about 15 minutes), 2026-10-11. Numbers are in
[results.json](results.json). One training seed.

**Criterion 1, objects become tokens: holds.** In world 2's test play,
once an object has moved or changed, its tiles are exactly one token in
99.5% of 50,647 object-states (bar 95%).
- By kind: doors 100%, shapes of several colours 99.5%, of one colour
  99.3%.
- Colour and touching manages 55.7%: 0% for shapes of several colours,
  87.1% for shapes of one colour.

**Criterion 2, effects on the whole object: holds, on blocked pushes.**
The table shows steps where the agent walks into a tile of a shape or a
door, with every tile of that object predicted right.

| Every tile of the object right | One tile per token | Fate groups | True objects (upper bound) | Steps |
|---|---|---|---|---|
| **Objects moved or changed before (judged)** | 86.0% | **94.9%** | 95.2% | 3,168 |
| Pushes that moved the shape | 66.5% | 74.7% | 87.8% | 442 |
| Pushes that were blocked | 86.8% | 97.8% | 95.6% | 2,235 |
| Doors opened or closed | 100% | 100% | 100% | 491 |
| First interactions (no group yet) | 77.4% | 80.3% | 91.6% | 3,622 |

- **Fate groups reach 99.7% of the upper bound's rate** (bar 95%), and
  are 8.9 points above one tile per token.
- **The pass rests on blocked pushes and doors.** On pushes that move
  a shape, fate groups reach 85% of the upper bound's rate, short of the
  bar. When a shape moves, each tile needs the group's extent and the
  cells beyond it, and pooling member states carries only part of that.
- **First interactions** are near one tile per token (80.3% against
  77.4%), as expected: nothing has grouped yet.

**Criterion 3, no false groups: holds.** Crafter's held-out worlds 8–9
have no group of two or more tiles in any of 21,947 states (bar: at
most 1%).
- Per state, 3.65 tracked tiles have moved or changed. The agent's
  move is read in 99.6% of steps.
- Two rules got there, both from the layer model. Before them, 49% of
  states held a false group:
  - A mover must have left its place. A skeleton's new arrow had been
    read as the skeleton moving, which put one tile in two places.
  - A tile that appears on ground or vanishes into it is a new or ended
    tile, not a change in place (Crafter spawns and despawns creatures).
- The learned prior then does the rest. Crafter's training play ends no
  pair as one object, so the prior is −9.6 log-odds, against −0.62 in
  world 2.

**Reported:**
- **First sight is weak.** The pair head groups 33.5% of objects that
  have not yet moved exactly (24,777 object-states), against 30.8% with
  per-tile training and 34.4% with true objects. The prediction ("first
  sight carries to new shapes of familiar tiles") was wrong.
- **Every tile's own fate, at every test step:** 99.89% (fate groups),
  99.86% (per tile), 99.91% (upper bound).
- **Cost:** grouping takes 72 s for world 2's 250,000 steps and 22 s
  for Crafter's 121,000. Each training run takes about 4 minutes.

## 8. Decision

**Revise.** Verdict: pass on all three criteria, but first sight is weak
(34%), and the card's rule for that case is "revise".

What holds:
- **Common fate gives tokens of many tiles.** Moved objects are exactly
  one token in 99.5% of states, against 56% for colour and touching, and
  Crafter forms no false group.
- **Groups help prediction:** 94.9% against 86.0% for one tile per
  token.

The two weak points share a cause: a token does not keep where its tiles
are within it.
- **Pushes that move a shape** reach 85% of the upper bound's rate.
  Averaging member states keeps the group's centre, not its extent or
  the cells beyond it.
- **First sight** is a dot product of two tiles' states, and names
  34% of unmoved shapes.

**The revision, for the user to decide:** the token keeps its tiles'
places within it, as section 2 of the draft specified ("its tiles'
vectors with their places in the group") and the build simplified to a
mean.
- Each member's state gets its offset from the group's other members,
  pooled by attention within the group, not averaged.
- Predicted: pushes at 95% of the upper bound's rate.
- First sight is left as reported, unless the user wants it next.
- Budget: three runs of about 5 minutes; grouping is reused.

**The alternative** is to keep this form and carry both limits into
102.3, which needs groups as tokens, not their extent.

**Also learned:**
- **The training format matters.** Rare effects were learned only with
  loss-prioritised replay without importance weights. This is the
  user's point that the format can fix what the model alone does not.
- **Not tested:** rotation, objects that split or merge, and Crafter's
  carried rows (kept as one tile each).

## 9. Revision 1: a token keeps its tiles' places

Approved by the user on 2026-10-11 ("do the revision").

**What changes (one thing): how a group is read as one token.**
- **Before:** after the two pair layers over every tile, a group's
  members' states were averaged. The mean kept the group's centre but
  not its extent or the cells beyond it.
- **Now, a third pair layer runs within each group.** It has the same
  form (softmax attention, raw positions, the action), so each member
  reads its co-members' states and where they stand.
- **The group's state is pooled by attention** (one learned score per
  member, softmax within the group), not averaged. Its fate is predicted
  once, from that state.
- **Unchanged:** the groups themselves (layers.py's are reused), the
  training format (loss-prioritised replay, 12,000 steps) and the
  first-sight head (which reads the states before the within-group
  layer).
- **One tile per token** also gets the third layer, reading only
  itself, so all three forms have the same network.

**Runs** (`runs/102.2/chain_r1.sh`, about 15 minutes): one tile per
token, fate groups and true objects, each smoke-tested first.

**Judged** by criterion 2 as before, plus the weak point. On pushes
that move a shape, fate groups must reach at least 95% of the upper
bound's rate.

**Prediction:** pushes that move a shape rise from 74.7% to within 95%
of the upper bound; blocked pushes and doors stay where they were.

**Decision rules:**
- **Keep the revision** if criterion 2 holds and pushes reach the bar.
- **Otherwise keep the first form,** with its limits as stated in
  section 8.

### Result of revision 1

Runs: `runs/102.2/chain_r1.sh`, about 16 minutes, one training seed.
The table counts interactions with objects that have moved or changed
before, every tile of the object predicted right; first form →
revision 1.

| | One tile per token | Fate groups | True objects (upper bound) | Steps |
|---|---|---|---|---|
| All (criterion 2) | 86.0% → 87.6% | 94.9% → **96.0%** | 95.2% → 99.0% | 3,168 |
| Pushes that moved the shape | 66.5% → 61.3% | 74.7% → **92.1%** | 87.8% → 100% | 442 |
| Pushes that were blocked | 86.8% → 90.0% | 97.8% → 95.8% | 95.6% → 98.6% | 2,235 |
| Doors | 100% → 100% | 100% → 100% | 100% → 100% | 491 |
| First interactions (no group yet) | 77.4% → 79.7% | 80.3% → 75.5% | 91.6% → 97.5% | 3,622 |

- **Criterion 2 holds:** fate groups reach 96.9% of the upper bound's
  rate (bar 95%).
- **The pushes bar does not:** 92.1% of the upper bound's rate (bar
  95%), though up from 74.7% to 92.1% in absolute terms.
- **With true objects, a pair layer within the token predicts pushes
  perfectly** (100%, from 87.8%). Keeping the tiles' places is the
  right form for a token of many tiles. The 8 points fate groups still
  lose come from grouping and training, not from the token.
- **First interactions with fate groups fell** from 80.3% to 75.5%,
  while one tile per token rose. Singletons get the new layer too; one
  seed does not say whether the drop is real.
- **First sight** is unchanged: 34.0% (fate groups), 32.2% (one tile
  per token), 37.0% (true objects).

## 10. Decision

**Keep** (the card): tiles that share fate become one token; criteria
1–3 hold.

**On the revision, by the rule set before the run, the first form
stands.** Pushes reached 92.1% of the upper bound's rate, against a bar
of 95%. The rule measured against the upper bound, and the revision
raised the upper bound too (87.8% to 100%), so it was stricter than
intended.
- Against the first form, revision 1 is better on pushes (92.1% against
  74.7%) and over all (96.0% against 94.9%).
- It is worse on blocked pushes (95.8% against 97.8%) and on first
  interactions (75.5% against 80.3%).
- **Recommended to the user:** override to revision 1's token. The
  within-group layer brings true objects to 100% on pushes; the mean
  cannot.
- **The user overrode the rule** (2026-10-11, "adopt if it was an
  improvement"). Revision 1's token is kept: a pair layer within each
  group and attention pooling (`predict.py --within`).

**Limits carried forward:**
- First sight names 34% of unmoved shapes.
- Rare effects need loss-prioritised replay.
- Rotation, splitting and merging objects, and Crafter's carried rows
  are untested.

## Appendix A. Breaking world 2, "shapes"

As built (`tools/card102/worlds.py 2`). The draft's 14 × 14 room was cut
to fit Crafter's frame, so Crafter's adapter reads it unchanged.
- **The room:** world 1's 9 × 7, seen whole (the view does not move).
  The agent moves in four directions.
- **Shapes:** 2–3 of 2–5 tiles each (random polyominoes). Half are of
  one colour, half of several, from six colours. Walking into a shape
  pushes it one tile when every cell beyond it is free floor.
- **The door:** 2–3 tiles in the top or bottom wall. Walking into any of
  its tiles opens or closes all of them.
- **Splits:** training worlds draw from 13 free shapes, test worlds from
  the other 7.
- **Play:** 2,000 training and 500 test episodes of 100 random actions
  (noop and four moves).
