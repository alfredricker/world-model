---
id: "098.8"
title: what happens next shapes the tokens — one rule of continuity for every token, with its own motion and the action's change
rung: 1
serves: [P7, P1, P6, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-11
---

# 098.8: what happens next shapes the tokens

Drafted 2026-10-11 at the user's request, after card
[098.7](../098.7-how-wide-is-like-this/card.md) (stop). The user's
direction: one prior for every token, corrected by the agent's own
experience where it is wrong; and "the tracking, what the objects do on
their own, and what the objects do through agent interaction should all
shape the encoder space and the dynamic predictions". Offline; the agent
is not run.

## 1. Question

Card 098.3's encoder gives scenery the same code by day and by night
(95.1% of night tiles), because it is trained on pairs of the same world
place at two times. Creatures move, so that rule mislabels them: cows
keep a code per light level, and one dusk code mixes cows, zombies and
arrows. 098.6's object files then lose 86% of movers' night frames.

Suppose the encoder is trained instead by one rule for every token: a
token in one frame continues as a token in the next. The continuing token
is either the same token, moved as tokens of its kind move (scenery not
at all), or the token the agent's action made of it. Do creatures' night
tokens then join their kind's day tokens? And does recall in the new
space predict what creatures *never followed in training* will do?

The second question is the user's: does following individual cows teach
the concept "cow"? Rung 1. Serves P7 (units from continuity), P1 (a kind
means what its members do), P6 (what actions change) and P2.

## 2. What changes

One component: the encoder's training signal.

```
098.3:      pairs of the same world place: consecutive still frames, and two frames whose
            light (the simulator's) differs by 0.2, places matched by the agent's summed shifts
this card:  a walk over tokens, forward through frames and back (contrastive random walk)
            each step: token i at t continues as token j at t+1 by one of two explanations
              stays itself:  z_i like z_j, and i's kind moves by the offset from i to j
              changed:       z_j is what i becomes under the action a
            training: the walk must come back to the token it started from
```

- **Motion and change are learned with the encoder.** Two small networks
  read the encoder's vectors:
  - **motion:** from z_i, how likely a token like i is to move by each
    offset. Scenery learns "nowhere", cows "one step or none";
  - **change:** from z_i, the action and whether i is in front of the
    agent, how likely i is to change, and the vector it becomes.

  Both train only through the walk, so what a token does decides which
  tokens the encoder puts together. This is the user's coupling: kinds
  with the same motion and changes are pulled together, and a match that
  needs a cow to jump three tiles is unlikely.
- **Correction by experience** (the user, 2026-10-11). A link the walk
  cannot come back through counts little, as in a crossing of two cows.
  So does a link that the token's own kind would not make.
- **In use,** dynamics are still predicted by recall over the encoder's
  space (098.6's files level). Recall is judged in criterion 2. Whether
  the two networks predict better where they are confident is a later
  card, under CHARTER's "recall and networks, each where it predicts
  better".
- **Removed:** 098.3's pairs across light, and with them the simulator's
  light in choosing pairs (a C1 leak, found 2026-10-11). Still pairs and
  variant H's changed place are the walk's special cases: offset zero,
  and "changed".
- **Kept:** the 7 × 7 tile grid (declared), the network's shape, the
  learned view shifts (098.2), the learned front (098.1, 098.3), and the
  code rule: leader clustering with a radius from training's own same
  and different matches.
- **Not in this card:** the agent; 098.4's contrast with doing nothing
  (the change network learns what follows an action, not what the action
  causes); the asleep appearance; noise at the night view's edges, which
  no encoder recovers.

Appendix A gives the walk in full.

### Revision after the gate (the card's one; the user, 2026-10-11)

The gate failed (section 5). Trained through the walk, the motion network
settled on "nothing moves". A moving token covers one tile and uncovers
another, and "they swapped" comes back as surely as the truth. The user
chose two changes. The success criteria are unchanged.

1. **Motion is learned from the links appearance makes.** At each step,
   appearance alone (z_i · z_j / τ) gives each token's likely
   continuations. The motion network learns to predict their offsets,
   each link weighted by how surely appearance's one-step cycle comes
   back. Its loss reaches the encoder through its input, so tokens that
   move alike are still pulled together. In the walk, its prediction is a
   fixed prior, so the walk cannot bend it toward "nothing moves". This
   is the user's rule: one prior (a token continues as a token that looks
   like it), corrected by experience (how tokens like it have been seen to
   move).
2. **Each tile is encoded with its neighbours** (from the ARC note,
   Appendix B). The encoder reads the 3 × 3 tiles around a place
   (21 × 21 pixels, with a channel marking places outside the view), as
   Jabri's patches carry their surroundings. Same-looking tiles then
   differ by what is around them: grass by a tree from grass in a field,
   or in ARC a cell of one colour by the shape it belongs to. Appearance
   links were weakest exactly there: with appearance alone, scenery was
   linked to its own place 4% of the time. The risk is that a token's
   vector also carries where it stands, so a kind may split by its
   surroundings. Criterion 1 measures that.

The change network is trained through the walk as before. Walks run 2 to
40 steps as Appendix A says (the gate's runs used 32 by mistake).

**Arms:** W, W+M and W+M+C, with motion learned this way, and N, which is
W+M+C with neighbours. The gate (tracking check, day purity) is run again
on all four before the main run. Criteria 1–2 are scored for all four.
Criterion 3 (098.6's files level, about 40 minutes each) is scored for
W+M+C and N. The decision follows the better of W+M+C and N by
section 6's rules; where they do equally well, W+M+C (the simpler) is
kept.

## 3. Dependencies

- Data: card 098.5's 300,000 random steps on seeds 2000–2009. These
  worlds were never used to train an encoder. Workers 0–7 train, and
  workers 8–9 (other worlds, so other creatures) test.
- 098.3's network shape and code rule; 098.2's learned view shifts;
  098.1's learned front (100% of awake steps on 098.3's tokens).
- 098.6's files level, unchanged, for criterion 3.
- Jabri, Owens and Efros 2020, contrastive random walk; Wang, Jabri and
  Efros 2019, cycle-consistency of time (LITERATURE). Both learn
  correspondence from video with no labels, the return to the start being
  the only signal.
- LESSONS:
  - A light shift shared by one frame cancels unless both frames' tokens
    are candidates in one softmax (098.2).
  - Predicting one's own latent vectors can erase what actions change
    (cards 052–053). Here every target is an observed token picked
    against the others, so a constant vector cannot satisfy it.

## 4. Data check

Done while drafting (`runs/098.5`, workers 0–9, evaluator's state):

| | Cow | Zombie | Skeleton | Arrow |
|---|---|---|---|---|
| Individuals | 3,776 | 2,568 | 732 | 6,912 |
| Frames in view | 171,480 | 44,257 | 32,270 | 30,711 |
| of them at night (light below 0.3) | 12,820 | 19,681 | 4,387 | 3,400 |
| Unbroken stretches in view across a light change of 0.2 or more | 1,133 | 730 | 278 | 0 |
| of them from day (above 0.7) to night | 118 | 9 | 30 | 0 |
| "Do" while facing it | 183 | 307 | 1 | 43 |

- **Following can tie day to night.** Dusk (light 0.7 to 0.3) lasts a
  median of 35 steps, and a cow stays in view about 45 frames. Over a
  thousand cow stretches and 730 zombie stretches cross a change of 0.2.
  Few go all the way from day to night, so the tie has to chain through
  dusk.
- **No pixel statistic stands in for the light.** Mean brightness
  correlates with it at 0.30 and saturation at 0.48, because scenes
  differ. Walks are drawn without regard to light, and the share that
  crosses dusk is reported.
- **The agent rarely acts on creatures** (183 tries on cows in 300,000
  steps). The change network will shape mostly scenery: trees, grass,
  water and stone. Creatures are shaped by motion.

## 5. Feasibility gate

- **Upper bounds:**
  - criterion 1: a classifier on the evaluator's labels, 98.3% at night
    (098.2);
  - criterion 3: the files level on the true state, IDF1 0.934 (098.6).
    IDF1 is the share of frames in which each object is followed by its
    own track.
- **Trivial baselines:** 098.3's encoder, scored on workers 8–9 in the
  same way; raw-pixel k-means at the same number of codes.
- **Is tracking right?** After a quarter of training, by day on workers
  0–7, the walk's likeliest continuation must be the evaluator's same
  object for at least 90% of creature tokens and 99% of scenery. If not,
  the walk is teaching wrong identities and the main run stops.
- **No collapse:** day purity at least 99% (098.3: 99.9%). Each arm is
  logged term by term.
- Timing: the first 200 training steps are timed before any full run.

Result of the gate, before the main run (`runs/098.8/`, code in
`tools/card098.8/`; 12,000 training steps per arm, about 3 minutes):

**Settled while building:**
1. **The agent's own motion comes from card 098.5's placing step**
   (`place.py`: the view placed against the remembered layout, 98% right
   in every worker). 098.2's two-frame check missed about a fifth of real
   moves, mostly in uniform terrain, where two frames barely differ.
2. **A return counts only at the walk's own start token.** When any
   start-frame tile drawn identically counted, nothing made the walk stay
   on the same tile.
3. **No edge dropout.** Jabri drops 10% of links among image patches,
   where an object has many. A tile has one right continuation, so in a
   walk of about 34 steps the right link was dropped in 97% of walks,
   and the motion network learned to hedge.

**Tracking check** (worker 7 by day; the likeliest continuation against
the evaluator's same object or same place):

| Arm | Creatures (bar 90%) | Scenery (bar 99%) |
|---|---|---|
| W+M+C, as designed | 45% | 86% |
| W+M, as designed | 52% | 94% |
| W (appearance only) | 83% | 4% |
| Motion fitted to the walk's own links, held fixed in the walk | 54% | 96% |
| Motion fitted to the links appearance alone makes, held fixed in the walk | **97.5%** | **99.8%** |

By day, 57% of cows and 75% of skeletons stay put from one step to the
next.

**The design as written fails the check.** A cow that moves covers one
grass tile and uncovers another, and in the walk every token must
continue somewhere. The false story ("nothing moved; the cow became
grass and the grass became a cow") is one-to-one, so it comes back every
time. Trained through the walk, the motion network settles on "cows
never move" (P(stay) 1.0), and the walk then links 284 of 290 moved
cows to the grass left behind. Fitting motion to the walk's own links
confirms the same story. Motion fitted to the links that appearance
alone makes learns P(stay) 0.54 for cows (true 0.57), and the walk
follows them. In W+M+C the change network also took over: the chance
of a change was 0.99 for every token.

**Side readings of the passing variant** (worker 7, a training worker;
098.3 in brackets):
- day purity 99.8% (99.9%) and night purity 96.0% (94.0%);
- night tokens in a code of their class's day tokens: cows 91.8%
  (61.9%), skeletons 72.3% (20.8%), zombies 93.4% (97.3%), all night
  tiles 80.1% (95.0%);
- 17 codes (53), and one code for the agent's four headings (098.3:
  five). With no change network, a turn reads as the same token.

By the card's rule, a failed tracking check means stop. The user chose
a revision instead (section 2).

**The gate after the revision** (`runs/098.8/gate2.sh`; walks of 2 to
40 steps, 12,000 steps per arm; the first gate's runs are kept as
`runs/098.8/gate1_*`):

| Arm | Creatures, after a quarter / at the end (bar 90%) | Scenery, after a quarter / at the end (bar 99%) | Codes |
|---|---|---|---|
| W | 78% / 78% | 4% / 4% | 331 |
| W+M | 96% / 87% | 95% / 90% | 15 |
| W+M+C | 97% / 99% | 94% / 81% | 16 |
| N (W+M+C with neighbours) | 69% / 67% | 90% / 89% | 335 |

- **Motion learned from appearance's links does not hold.** Early in
  training it follows creatures and scenery at about 95%, and the first
  gate's single run (walks up to 32 steps) reached 97.5% and 99.8%. As
  training goes on, scenery falls. Appearance cannot say where a tile of
  grass went when the tiles around it look the same, so motion fitted to
  appearance's links grows unsure for grass, and the walk then picks
  among identical tiles.
- **Neighbours make it worse for movers.** A cow's vector now carries
  the tiles around it, which change at every step it takes, so appearance
  links lose it (67%). Kinds split by surroundings: 335 codes.
- **The change network collapses in every arm.** It gives a change a
  chance of 0.88–0.99 at every place, whatever the action.

The gate fails in every arm. The main run was not made.

## 6. Success criteria and prediction

Judged on workers 8–9, whose worlds and creatures training never saw.

1. **One kind by day and by night, for creatures and scenery alike.**
   For each creature class (cow, zombie, skeleton) and for scenery pooled,
   at least 90% of night tokens must carry a code that holds at least 1%
   of that class's day tokens. Purity must be at least 94% at night and
   99% by day. Compared with: 098.3 (scenery 95.1%; creatures scored in
   the gate). This would show that one rule ties appearances across light
   for tokens that move as well as for those that stay.
2. **The concept carries to new individuals.** Recall over "tokens like
   this" (098.6's recall, at the new identity radius) predicts new
   creatures' next moves:
   - better than with 098.3's encoder, by day and at night (paired
     bootstrap, p < 0.05);
   - at night, within 0.2 nats of day;
   - within 0.1 nats of creatures from the training workers, whom
     training followed.

   Recall's prediction of what the action does to the front token is no
   worse than with 098.3's encoder. This would show that following
   individuals built a kind that predicts for members never followed.
3. **Object files hold on the new tokens.** 098.6's files level, unchanged:
   IDF1 at least 0.85, and movers' night frames followed by their own file
   at least 50% (098.6: 0.717 and 14%). This is below 098.5's 0.90 bar
   because this card judges the encoder; 098.6's day losses have their own
   cause, still to be diagnosed.

**Reported:** three arms on the same data and seeds, to show what each
part adds: walk alone (W), with motion (W+M), and with motion and change
(W+M+C, the card's). After the revision there is also N, W+M+C with
neighbours. Also reported:
- night tokens by distance from the view's centre;
- codes per class, and the share of walks crossing dusk;
- motion and change predicted by the two networks, for comparison only.

**Prediction.**
- By day, cows fall to 1–2 codes.
- At night, creatures reach 85–95% in the inner rings and lower at the
  edge, so criterion 1 may fail for zombies. They are seen mostly at
  night, and their day appearance is the rarer one.
- Criterion 2 holds by day. At night the gap is about 0.2–0.3 nats.
- IDF1 is about 0.80–0.85.
- W+M beats W on creatures. C changes mostly scenery and the front token.

**Decision rules.**
- **Keep** if 1–3 hold. The encoder then becomes the Crafter series'
  tokens.
- **Revise** if creatures pass by day and fail at night only in the outer
  rings, or if a simpler arm does as well as W+M+C (then the simpler arm
  is kept).
- **Stop** if the tracking check fails, or if creatures' night codes are
  no better than 098.3's. Continuity through frames would then not tie
  creatures' appearances on these frames.

**Budget.**
- Training is three arms on 240,000 frames. 098.2's 4,000 steps took
  2 minutes; walks of up to 40 frames make each step about five times
  dearer, so expect about 15 minutes per arm, timed first.
- Evaluation, including 098.6's files on workers 8–9, takes about
  10 minutes per arm.
- In all, about 75 minutes, in runs of 20 minutes or less. Runs over
  30 minutes go to the user as commands.
- *After the revision (timed):*
  - training takes 12,000 steps: about 4 minutes for each tile arm and
    18 minutes for N;
  - criteria 1–2 take about 5 minutes per arm;
  - criterion 3 takes about 40 minutes per arm, in three runs of
    20 minutes or less;
  - in all, about 2 hours 15 minutes.

## 7. Result

Not run; the gate failed before and after the revision (section 5).
Card 098.3 was scored on workers 8–9 for the comparison and is kept in
`runs/098.8/base_098.3/eval.json`. Night tokens in a code of their
class's day tokens: cows 61.8%, zombies 98.0%, skeletons 19.7%, scenery
96.1%. Purity 99.9% by day and 94.5% at night.

## 8. Decision

**Stop.** The tracking check failed in every arm, as designed and after
the card's one revision. What the gate showed:
- **Following by appearance works for creatures, not for scenery.**
  Appearance alone gives the same creature 78–83% of the time, but an
  identical grass tile is as good as any other.
- **A walk over places prefers "they swapped" to "it moved".** Each
  token must continue as some token, so a token that is covered (the
  grass under a cow) has no right continuation. "The cow became grass and
  the grass became a cow" is one-to-one, and the cycle rewards it.
  Trained through the walk, motion settles on "nothing moves". Fitted to
  appearance's links, it drifts toward unsure for grass.
- **Change trained through the walk explains everything.** In every arm
  it took over.

The missing piece is a way for a token to be hidden or revealed. The
object files (098.5–098.6) have that: births, deaths and the pair test
for a token that left one place and appeared at the next. On the true
state they keep identity at IDF1 0.934. A next card could train the
encoder on the links the files level makes, rather than on a walk over
every place. That would be a new card for the user to approve.
098.3's encoder remains the best Crafter tokens.

## Appendix A. The walk

- **Tokens.** Each frame's 63 world tiles, each with its encoder vector
  z and its place in the episode's own coordinates (098.2's summed
  shifts). The agent's tile is the agent and continues as the agent.
- **A walk** starts at a token in frame t0. It steps through consecutive
  frames t0 … tk, with k drawn from 2 to 40, then back to t0 (Jabri's
  palindrome). Each step's actions are known.
- **One step, i at t to j at t+1.** The transition probability is a
  softmax over j of the larger of two explanations:
  - stays: z_i · z_j / τ + log m(offset_ij | z_i). m is the motion
    network, and offsets beyond two places get no weight. *Since the
    revision,* m is fitted to appearance's links and held fixed in the
    walk;
  - changed: c(z_i, a, front_i) · z_j / τ + log π(z_i, a, front_i). c is
    the vector the change network predicts, and π the chance of a change,
    with a prior of 0.05.

  Only the token in front can be changed by the action. Other tokens can
  change by themselves (an arrow vanishing, a sapling growing), with the
  action's input set to none.
- **The return step's candidates** are the start frame's tokens together
  with the far frame's, so that a light shift shared by one frame does not
  cancel. A start-frame token drawn identically to the start token counts
  as a return in the tie, but not in the cycle (settled while building,
  section 5).
- **Loss:** minus the log-probability of returning, summed over start
  tokens. There is no edge dropout (settled while building, section 5).
- **Same and different matches for the code radius:** links taken with
  probability above 0.9 in both directions are "same". Tokens of one
  frame drawn differently are "different".

## Appendix B. Note for later: objects of many cells (ARC-AGI-3)

Added 2026-10-11 at the user's request. Its first remedy, tiles read
with their neighbours, became arm N of the revision (section 2); grouping
by common fate is not part of this card's test.
The question was whether this card, kept as version 28, would help with
objects that span many cells, as in ARC-AGI-3's games (grids up to
64 × 64 of 16 colours, a few move actions and a click at any cell).

- **What would carry over:** identity learned from continuity and from
  what actions change, with no labels; kinds defined by how their tokens
  move (the motion network) and what actions make of them (the change
  network). These are criteria 2 and 3. Criterion 1 (one code by day
  and night) does not apply, since ARC's colours do not vary with light.
- **Why it would not solve objects of many cells:**
  - A token is one tile, encoded alone
    ([`tools/card098/encoder.py`](../../tools/card098/encoder.py), the
    encoder sees a 7 × 7-pixel tile and no neighbours). In ARC a cell is
    one colour value, so every cell of a colour gets the same vector.
  - The walk then cannot tell same-coloured cells apart: when a shape
    shifts, each cell has several equally likely continuations, the
    crossing case that Appendix A down-weights. Jabri's patches avoid
    this by overlapping and carrying their surroundings.
  - Nothing groups tokens into one object. The motion network says how
    a kind of token moves, not which tokens move together.
  - Crafter's declared forms would break: no weight beyond two places
    (ARC shapes can jump far), only the front token changed by an action
    (ARC has no front; a click acts anywhere), and the fixed tile grid.
- **What would help, on top of this card:** tokens that are groups of
  cells, formed by common fate (cells that move by the same offset at
  the same step are one object). The motion network's predicted offsets
  are a natural signal for that grouping, so this card is a step toward
  it rather than the fix. In Crafter the need is smaller: creatures and
  tiles are one tile each, and structures such as lakes and tunnels are
  the main objects of many tiles.
